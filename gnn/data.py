"""
数据加载与批处理（支持 PQ-only 大批次与逐图聚合）

要点
- 支持扩展特征维度：节点特征从4列扩展到20列（基础4列+发电机6列+预计算特征10列，Q先验已移除）
- collate 在跨图合批时对所有与节点/走廊相关的字段做"索引偏移"以避免错位：
  tie_buses、tie_corridors、tie_lines、tie_line2corridor、tie_edge_corridor。
- collate 额外在 Batch 对象上挂四类"前缀和指针（ptr）"：
  node_ptr / edge_ptr / bus_ptr / corr_ptr，用于训练端按"逐图均值→跨图均值"的公平聚合。
- create_dataloader 支持 prefetch_factor / persistent_workers / pin_memory 等吞吐优化参数。
"""

import torch
import os
import glob
import itertools
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader  # PyG's DataLoader (not used for custom collate)


def _collate_with_offsets(batch):
    """
    自定义collate函数，修正跨图批处理时的索引偏移

    必须在模块层级定义以支持multiprocessing pickle序列化
    """
    # 先使用PyG默认方式构建Batch（这样batch属性正确）
    batch_out = Batch.from_data_list(batch)

    # 计算偏移量
    node_counts = []
    corr_counts = []
    edge_counts = []
    bus_counts = []

    node_offset = 0
    corridor_offset = 0

    for d in batch:
        num_nodes = d.x.size(0)
        num_corridors = d.tie_corridors.size(0) if hasattr(d, 'tie_corridors') else 0
        num_edges = d.edge_index.size(1) if hasattr(d, 'edge_index') else 0
        num_buses = d.tie_buses.size(0) if hasattr(d, 'tie_buses') else 0

        node_counts.append(num_nodes)
        corr_counts.append(num_corridors)
        edge_counts.append(num_edges)
        bus_counts.append(num_buses)

        node_offset += num_nodes
        corridor_offset += num_corridors

    # 在Batch构建后，手动修正需要偏移的字段
    # PyG已经正确处理了x, edge_index等，我们只需要修正边界相关字段

    device = batch_out.x.device
    node_offset = 0
    corridor_offset = 0

    # 收集所有需要偏移的字段
    all_tie_buses = []
    all_tie_corridors = []
    all_tie_edge_corridor = []
    all_tie_edge_indices = []

    for i, d in enumerate(batch):
        # tie_buses偏移
        if hasattr(d, 'tie_buses') and d.tie_buses is not None:
            all_tie_buses.append(d.tie_buses + node_offset)

        # tie_corridors偏移
        if hasattr(d, 'tie_corridors') and d.tie_corridors is not None and d.tie_corridors.numel() > 0:
            all_tie_corridors.append(d.tie_corridors + node_offset)

        # tie_edge_corridor偏移（-1保持不变）
        if hasattr(d, 'tie_edge_corridor') and d.tie_edge_corridor is not None and d.tie_edge_corridor.numel() > 0:
            tec = d.tie_edge_corridor.clone()
            valid = tec >= 0
            tec[valid] = tec[valid] + corridor_offset
            all_tie_edge_corridor.append(tec)

        # tie_edge_indices偏移（联络线在全局edge_index中的索引）
        if hasattr(d, 'tie_edge_indices') and d.tie_edge_indices is not None and d.tie_edge_indices.numel() > 0:
            edge_offset = sum(edge_counts[:i])  # 前i个样本的边数总和
            all_tie_edge_indices.append(d.tie_edge_indices + edge_offset)

        node_offset += node_counts[i]
        corridor_offset += corr_counts[i]

    # 拼接并覆盖Batch的字段
    if all_tie_buses:
        batch_out.tie_buses = torch.cat(all_tie_buses, dim=0).to(device)

    if all_tie_corridors:
        batch_out.tie_corridors = torch.cat(all_tie_corridors, dim=0).to(device)

    if all_tie_edge_corridor:
        batch_out.tie_edge_corridor = torch.cat(all_tie_edge_corridor, dim=0).to(device)

    if all_tie_edge_indices:
        batch_out.tie_edge_indices = torch.cat(all_tie_edge_indices, dim=0).to(device)

    # 添加ptr指针
    batch_out.node_ptr = torch.tensor([0] + list(itertools.accumulate(node_counts)), dtype=torch.long, device=device)
    batch_out.corr_ptr = torch.tensor([0] + list(itertools.accumulate(corr_counts)), dtype=torch.long, device=device)
    batch_out.edge_ptr = torch.tensor([0] + list(itertools.accumulate(edge_counts)), dtype=torch.long, device=device)
    batch_out.bus_ptr = torch.tensor([0] + list(itertools.accumulate(bus_counts)), dtype=torch.long, device=device)

    return batch_out



def load_split_data(data_dir, split='train'):
    """
    从已划分的目录加载数据
    
    Args:
        data_dir: 数据根目录，应包含 train/val/test 子目录
        split: 数据集类型 ('train', 'val', 'test')
    
    Returns:
        data_list: 数据对象列表
    """
    assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
    
    # 构建路径
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"未找到数据目录: {split_dir}\n" +
                               "请先运行 prepare_dataset.py 划分数据集")
    
    # 加载所有文件
    pattern = os.path.join(split_dir, '*.pt')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"未找到数据文件: {pattern}")
    
    print(f"从 {split} 集加载 {len(files)} 个文件...")
    
    # 加载所有数据
    all_data = []
    for file in files:
        data = torch.load(file, weights_only=False)
        if isinstance(data, list):
            all_data.extend(data)  # 多个样本的列表
        else:
            all_data.append(data)  # 单个样本
    
    print(f"  {split} 集: {len(all_data)} 个样本")
    
    return all_data


def compute_voltage_stats(data_list):
    """
    统计数据集中电压标签的范围，用于自动调整模型输出映射
    
    Args:
        data_list: 数据列表
        
    Returns:
        dict: 包含v_min, v_max, v_mean, v_std的统计信息
    """
    import numpy as np
    
    all_voltages = []
    for data in data_list:
        if hasattr(data, 'y_bus_V') and data.y_bus_V is not None and data.y_bus_V.numel() > 0:
            all_voltages.extend(data.y_bus_V.detach().cpu().numpy())
    
    if not all_voltages:
        # 默认值，如果没有电压数据
        return {
            'v_min': 0.9, 'v_max': 1.1, 
            'v_mean': 1.0, 'v_std': 0.05
        }
    
    v_array = np.array(all_voltages)
    return {
        'v_min': float(v_array.min()),
        'v_max': float(v_array.max()),
        'v_mean': float(v_array.mean()),
        'v_std': float(v_array.std())
    }


def load_data(data_dir):
    """
    加载所有划分好的数据集，并自动统计电压范围
    
    Args:
        data_dir: 数据根目录，应包含 train/val/test 子目录
    
    Returns:
        train_data, val_data, test_data: 训练、验证和测试数据列表
        voltage_stats: 电压统计信息字典
    """
    train_data = load_split_data(data_dir, 'train')
    val_data = load_split_data(data_dir, 'val')
    test_data = load_split_data(data_dir, 'test')
    
    # 打印统计信息
    total = len(train_data) + len(val_data) + len(test_data)
    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_data)} 样本 ({100*len(train_data)/total:.1f}%)")
    print(f"  验证集: {len(val_data)} 样本 ({100*len(val_data)/total:.1f}%)")
    print(f"  测试集: {len(test_data)} 样本 ({100*len(test_data)/total:.1f}%)")
    print(f"  总计: {total} 样本")
    
    # 统计训练集电压范围（用于模型自适应）
    voltage_stats = compute_voltage_stats(train_data)
    print(f"\n电压标签统计（基于训练集）:")
    print(f"  范围: [{voltage_stats['v_min']:.4f}, {voltage_stats['v_max']:.4f}]")
    print(f"  均值±标准差: {voltage_stats['v_mean']:.4f}±{voltage_stats['v_std']:.4f}")
    
    return train_data, val_data, test_data, voltage_stats


# 注意：已移除 remap_partition_ids 函数
# 原因：data.x[:,4] 是 is_coupling_bus (0/1标记)，不是分区ID
# 重映射会破坏数据语义，没有实际意义


def create_dataloader(data_list, batch_size=32, shuffle=True,
                     sampler=None, num_workers=0, pin_memory=False,
                     persistent_workers=False, prefetch_factor=None,
                     add_ptrs=True):
    """
    创建数据加载器

    Args:
        data_list: Data对象列表
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载进程数
        pin_memory: 是否固定内存（加速GPU传输）
        add_ptrs: 是否添加ptr指针（需要custom collate）

    Returns:
        DataLoader对象
    """
    # 当 sampler 提供时，必须关闭 shuffle 以避免冲突
    effective_shuffle = shuffle and (sampler is None)
    # 构建 DataLoader 关键字参数（样本列表作为第一个位置参数传入）
    kwargs = dict(
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # 仅当启用多进程时，persistent_workers/prefetch_factor 才有效
    if num_workers > 0:
        kwargs['persistent_workers'] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = int(prefetch_factor)

    # PyG's DataLoader removes custom collate_fn, so use PyTorch's native DataLoader when add_ptrs=True
    if add_ptrs:
        loader = TorchDataLoader(
            data_list,
            collate_fn=_collate_with_offsets,
            **kwargs
        )
    else:
        # Use PyG's DataLoader with default Collater
        loader = DataLoader(
            data_list,
            **kwargs
        )

    # collate_fn已经正确添加了ptr和偏移，直接返回loader
    return loader


def analyze_data(data_list):
    """
    分析数据统计信息
    """
    print("\n=== 数据分析 ===")
    
    # 统计基本信息
    n_samples = len(data_list)
    n_nodes = [d.x.shape[0] for d in data_list]
    n_edges = [d.edge_index.shape[1] for d in data_list]
    n_ties = [d.edge_attr[:, 3].sum().item() for d in data_list]
    
    print(f"样本数: {n_samples}")
    print(f"节点数: {min(n_nodes)}-{max(n_nodes)} (平均: {sum(n_nodes)/len(n_nodes):.1f})")
    print(f"边数: {min(n_edges)}-{max(n_edges)} (平均: {sum(n_edges)/len(n_edges):.1f})")
    print(f"联络线数: {min(n_ties)}-{max(n_ties)} (平均: {sum(n_ties)/len(n_ties):.1f})")
    
    # 耦合母线统计
    coupling_bus_stats = []
    for d in data_list[:100]:  # 采样100个
        # V1.1: is_coupling_bus 在 x 的第4列（0-based 索引=3）
        idx = 3 if d.x.size(1) >= 4 else d.x.size(1) - 1
        n_coupling = (d.x[:, idx] > 0.5).sum().item()
        coupling_bus_stats.append(n_coupling)
    
    if coupling_bus_stats:
        print(f"耦合母线数分布: {min(coupling_bus_stats)}-{max(coupling_bus_stats)} (平均: {sum(coupling_bus_stats)/len(coupling_bus_stats):.1f})")
    
    # 标签统计（PQ-only）
    if hasattr(data_list[0], 'y_corridor_pfqt') and data_list[0].y_corridor_pfqt is not None:
        dims = [d.y_corridor_pfqt.shape[0] for d in data_list]
        print(f"走廊端口标签维度: {min(dims)}-{max(dims)}")
    if hasattr(data_list[0], 'y_bus_pq') and data_list[0].y_bus_pq is not None:
        dims = [d.y_bus_pq.shape[0] for d in data_list]
        print(f"母线PQ标签维度: {min(dims)}-{max(dims)}")
    
    # 检查tie_buses字段
    if hasattr(data_list[0], 'tie_buses'):
        n_coupling = [len(d.tie_buses) for d in data_list]
        print(f"tie_buses统计: {min(n_coupling)}-{max(n_coupling)}")
    
    # 检查分区信息（仅用于统计）
    if hasattr(data_list[0], 'partition'):
        partitions = [torch.unique(d.partition).numel() for d in data_list[:100]]
        if partitions:
            print(f"分区数分布: {min(partitions)}-{max(partitions)}")


if __name__ == "__main__":
    # 测试数据加载
    print("测试数据加载...")
    
    # 测试数据目录（根据实际情况修改）
    test_data_dir = '../data/ieee118'  
    
    try:
        # 测试加载所有数据集
        train_data, val_data, test_data = load_data(test_data_dir)
        
        # 分析训练数据
        if train_data:
            analyze_data(train_data[:100])
        
        # 测试数据加载器
        if train_data:
            train_loader = create_dataloader(train_data[:32], batch_size=8)
            
            for batch in train_loader:
                print(f"\n批次信息:")
                print(f"  节点特征: {batch.x.shape}")
                print(f"  边索引: {batch.edge_index.shape}")
                print(f"  边特征: {batch.edge_attr.shape}")
                if hasattr(batch, 'y_corridor_pfqt'):
                    print(f"  走廊端口标签: {batch.y_corridor_pfqt.shape}")
                if hasattr(batch, 'y_bus_pq'):
                    print(f"  母线PQ标签: {batch.y_bus_pq.shape}")
                print(f"  批次大小: {batch.batch.max().item() + 1}")
                break
                
        print("\n数据加载测试通过！")
        
    except Exception as e:
        print(f"错误: {e}")
        print("提示: 请先运行 prepare_dataset.py 来划分数据集")
