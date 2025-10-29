"""
训练数据加载器 - 支持按分区数分组的智能batch策略
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from pathlib import Path
import random
from typing import List, Dict, Optional, Union
from collections import defaultdict

from opfdata.optimizer import OptimizedProcessor


class OPFDataset(Dataset):
    """
    OPFData 训练数据集。
    
    该数据集支持两种模式：
    1. group_by_k=True: 按分区数 k 对样本进行分组。这在训练时非常有用，
       可以确保每个批次都包含来自不同 k 值的样本，从而提高模型的泛化能力。
    2. group_by_k=False: 将所有样本视为一个扁平列表，进行常规加载。
    """
    
    def __init__(self, data_dir: Union[str, List[str]], group_by_k: bool = True):
        """
        初始化数据集。
        
        Args:
            data_dir (str): 包含处理后的 .pt 数据文件的目录。
            group_by_k (bool): 是否按分区数 k 对样本进行分组。
        """
        # 支持单目录或多目录混合
        if isinstance(data_dir, (list, tuple)):
            self.data_dirs = [Path(p) for p in data_dir]
        else:
            self.data_dirs = [Path(data_dir)]
        self.group_by_k = group_by_k
        self.optimizer = OptimizedProcessor()
        
        # 加载所有样本
        self.samples = self._load_all_samples()
        
        if self.group_by_k:
            # 如果启用分组，则将样本按 k 值存入字典
            self.samples_by_k = self._group_samples_by_k()
            self.k_values = sorted(list(self.samples_by_k.keys()))
            print(f"数据已按k值分组: {[f'k={k}:{len(samples)}个样本' for k, samples in self.samples_by_k.items()]}")
        else:
            print(f"已加载 {len(self.samples)} 个样本")
    
    def _load_all_samples(self) -> List:
        """从目录中加载所有数据样本。"""
        all_samples = []
        # 查找所有名为 chunk_*.pt 的文件（支持多个目录）
        chunk_files: List[Path] = []
        for d in self.data_dirs:
            chunk_files.extend(sorted(d.glob('chunk_*.pt')))
        
        for chunk_file in chunk_files:
            try:
                # 使用优化过的加载器批量加载文件中的样本
                samples = self.optimizer.load_batch_optimized(str(chunk_file), format='pt')
                all_samples.extend(samples)
            except Exception as e:
                print(f"加载文件失败 {chunk_file}: {e}")
                continue
                
        return all_samples
    
    def _group_samples_by_k(self) -> Dict[int, List]:
        """根据样本自身的 'k' 属性将它们分组。"""
        samples_by_k = defaultdict(list)
        
        for sample in self.samples:
            # PyG Data 对象中应包含 k 属性
            k = sample.k
            samples_by_k[k].append(sample)
        
        return dict(samples_by_k)
    
    def __len__(self):
        """返回数据集的长度。"""
        if self.group_by_k:
            # 在分组模式下，数据集的长度定义为样本数最少的那个组的长度。
            # 这样可以确保在每个 epoch 中，每个 k 值分组都能被完整地遍历至少一次。
            if not self.samples_by_k:
                return 0
            return min(len(samples) for samples in self.samples_by_k.values())
        else:
            # 在非分组模式下，长度就是总样本数。
            return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个数据项。
        
        Args:
            idx (int): 索引。
        
        Returns:
            - group_by_k=True: 返回一个列表，其中每个元素都来自不同的 k 值分组。
            - group_by_k=False: 返回单个 PyG Data 样本。
        """
        if self.group_by_k:
            # 在分组模式下，为每个 k 值分组都取出一个样本，组成一个 "超级" 样本批次
            batch_samples = []
            for k in self.k_values:
                k_samples = self.samples_by_k[k]
                # 使用模运算确保索引不会越界，实现循环采样
                sample_idx = idx % len(k_samples)
                batch_samples.append(k_samples[sample_idx])
            return batch_samples
        else:
            # 在非分组模式下，直接返回对应索引的样本
            return self.samples[idx]


def collate_fn(batch):
    """
    自定义的 collate 函数，用于将 `__getitem__` 返回的数据项打包成一个批次。
    
    Args:
        batch: 一个列表，其中每个元素都是 `__getitem__` 的返回值。
    
    Returns:
        - group_by_k=True: 返回一个字典，键为 'k_2', 'k_3' 等，值为对应 k 值的样本批次 (Batch 对象)。
        - group_by_k=False: 返回一个标准的 PyG Batch 对象。
    """
    if isinstance(batch[0], list):
        # 此情况对应 group_by_k=True。
        # batch 的结构是 [[k2_sample_0, k3_sample_0, ...], [k2_sample_1, k3_sample_1, ...], ...]
        k_batches = {}
        num_k_groups = len(batch[0])
        
        # 遍历每个 k 值分组
        for k_idx in range(num_k_groups):
            # 提取出所有属于当前 k 值的样本
            k_samples = [item[k_idx] for item in batch]
            k_value = k_samples[0].k # 从第一个样本获取k值
            # 使用 PyG 的 Batch.from_data_list 将它们打包成一个独立的批次
            k_batches[f'k_{k_value}'] = Batch.from_data_list(k_samples)
            
        return k_batches
    else:
        # 此情况对应 group_by_k=False，直接使用标准方法打包。
        return Batch.from_data_list(batch)


def create_dataloader(data_dir: str, 
                     batch_size: int = 32,
                     group_by_k: bool = True,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """
    创建并返回一个用于训练的 DataLoader。
    
    Args:
        data_dir (str): 数据目录。
        batch_size (int): 批次大小。
        group_by_k (bool): 是否按 k 值对样本进行分组。
        shuffle (bool): 是否在每个 epoch 开始时打乱数据。
        num_workers (int): 用于数据加载的子进程数量。
    
    Returns:
        一个配置好的 torch.utils.data.DataLoader 对象。
    """
    dataset = OPFDataset(data_dir, group_by_k=group_by_k)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,  # 使用自定义的 collate 函数
        pin_memory=torch.cuda.is_available()  # 如果使用 GPU，则启用 pin_memory 以加速数据传输
    )


if __name__ == "__main__":
    # --- 数据加载器测试 ---
    print("=== 数据加载器测试 ===")
    
    # 假设数据已经通过 batch_process.py 处理并存放在此目录
    data_dir = "/home/zhangyao/renjiashen/workspace/gnn_data/processed"
    
    if Path(data_dir).exists():
        # --- 测试分组加载 (group_by_k=True) ---
        print("\n🔍 测试按 k 值分组加载:")
        loader = create_dataloader(data_dir, batch_size=2, group_by_k=True)
        
        # 迭代加载器并打印前几个批次的信息以供检查
        for i, batch in enumerate(loader):
            print(f"批次 {i}:")
            # 此时的 batch 是一个字典
            for k_name, k_batch in batch.items():
                print(f"  {k_name}: {k_batch.num_graphs} 个图, {k_batch.num_nodes} 个节点, {k_batch.num_edges} 条边")
            if i >= 2:  # 只查看前3个批次
                break
                
        print(f"\n数据加载器 (分组模式) 已就绪: 每个 epoch 有 {len(loader)} 个批次")
        
    else:
        print(f"数据目录不存在: {data_dir}")
        print("请先运行 batch_process.py 来生成处理好的数据。")
