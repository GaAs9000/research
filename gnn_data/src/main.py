"""
BC-GNN 数据处理脚本（V1.1 边界PQ生成）
=====================================
只负责将 OPFData 原始 JSON 转换为用于边界 P/Q 预测的训练数据。
不包含训练/评估/测试等逻辑，避免引入额外依赖与困惑。

多进程注意：强制使用 'spawn' 启动方式，避免 fork 继承旧模块；并置顶当前 src 目录到 sys.path。
"""

import os
import sys
import argparse
from pathlib import Path
from multiprocessing import get_start_method, set_start_method
from typing import List, Tuple

# ========================================
# 配置区 - 在这里修改您的路径和参数
# ========================================

# 数据集配置
DATASETS = {
    "ieee118": {
        "raw_dir": "/home/zhangyao/renjiashen/workspace/gnn_data/raw/118",
        "processed_dir": "/home/zhangyao/renjiashen/workspace/data/ieee118_mixed",
        "num_samples": 100,  # 处理的样本数量，None表示全部
    },
    "ieee118_n1": {
        "raw_dir": "/home/zhangyao/renjiashen/workspace/gnn_data/raw/118_N_1", 
        "processed_dir": "/home/zhangyao/renjiashen/workspace/data/ieee118_n1_mixed",
        "num_samples": 100,
    },
    "ieee500": {
        "raw_dir": "raw/500/500",  # JSON文件在嵌套的500目录中
        "processed_dir": "../data/ieee500_k456",  # 输出到上级目录的data文件夹
        "num_samples": None,  # None表示处理全部样本
    },
    "ieee500_n1": {
        "raw_dir": "gnn_data/raw/500_N_1",
        "processed_dir": "data/ieee500_n1_k456",
        "num_samples": None,
    },
    "ieee2000": {
        "raw_dir": "/home/zhangyao/renjiashen/workspace/gnn_data/raw/2000",
        "processed_dir": "/home/zhangyao/renjiashen/workspace/data/ieee2000_mixed", 
        "num_samples": 5,   # 仅用5个样本测试泛化能力
    }
}

# 分区配置
PARTITION_CONFIG = {
    "k_values": [5],           # 只生成 k=5 的分区
    "seeds_per_k": 1,          # 每个k值的随机版本数
    "method": "constructive",  # 分区方法（仅保留构造式）
    "size_tolerance": "0.9,1.1"  # 分区大小容差：90%-110%
}

# 处理配置
PROCESS_CONFIG = {
    "num_processes": 8,      # 并行处理进程数
    "chunk_size": 512,       # 每个chunk包含的样本数
    "train_ratio": 0.9,      # 训练集比例
    "val_ratio": 0.05,       # 验证集比例
    "test_ratio": 0.05,      # 测试集比例
}

# 无模型配置（本脚本仅做数据生成）

# ========================================
# 功能函数
# ========================================

def process_dataset(dataset_name: str, overwrite: bool = False):
    """
    处理单个数据集
    
    Args:
        dataset_name: 数据集名称
        overwrite: 是否覆盖已存在的处理结果
    """
    if dataset_name not in DATASETS:
        print(f"错误: 未知数据集 {dataset_name}")
        return False
    
    config = DATASETS[dataset_name]
    output_dir = Path(config["processed_dir"])
    
    # 检查是否已处理
    if output_dir.exists() and not overwrite:
        print(f"数据集 {dataset_name} 已处理，跳过（使用 --overwrite 强制重新处理）")
        return True
    
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")
    print(f"原始数据: {config['raw_dir']}")
    print(f"输出目录: {config['processed_dir']}")
    print(f"样本数量: {config['num_samples'] or '全部'}")
    
    # 调用处理脚本
    from process_opfdata import process_single_json, OPFDataProcessor
    from batch_process import process_chunk
    import json
    import torch
    import hashlib
    from multiprocessing import Pool
    from tqdm import tqdm
    
    # Step 1: 处理JSON文件
    print("\nStep 1: 处理JSON文件...")
    json_dir = Path(config["raw_dir"])
    json_files = sorted([f for f in json_dir.glob("*.json")])
    # 支持通过环境变量快速子样本抽样（离线对比/干跑）
    env_max = os.environ.get("BCGNN_MAX_SAMPLES")
    if env_max:
        try:
            json_files = json_files[:int(env_max)]
        except Exception:
            pass
    if config["num_samples"]:
        json_files = json_files[:config["num_samples"]]
    
    # 创建输出目录
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备处理参数
    args_list = []
    for json_file in json_files:
        args = (
            str(json_file),
            json_file.name,
            str(temp_dir),
            PARTITION_CONFIG["k_values"],
            PARTITION_CONFIG["seeds_per_k"]
        )
        args_list.append(args)
    
    # 设置环境变量
    os.environ["OPFDATA_SIZE_TOL"] = PARTITION_CONFIG["size_tolerance"]
    # 关闭逐文件终端摘要，仅写入日志文件
    os.environ.setdefault("OPFDATA_STDOUT_SUMMARY", "0")
    # 分区方法固定为构造式生长，此处不再设置 OPFDATA_METHOD
    
    # 控制底层并行库线程，避免多进程×多线程竞争
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # 多进程处理（无序返回 + 分块调度提升吞吐）
    chunksize = int(os.environ.get("OPFDATA_IMAP_CHUNKSIZE", "16"))
    with Pool(PROCESS_CONFIG["num_processes"]) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_json, args_list, chunksize=chunksize),
            total=len(args_list),
            desc="处理JSON文件"
        ))

    generated = sum(r[0] for r in results)
    failed = sum(r[1] for r in results)
    # 统计联通性成功率（以每文件 repair_failures==0 为成功）
    logs_dir = temp_dir / "logs"
    success_files = 0
    total_files = len(args_list)
    for _, json_file, _, _, _ in args_list:
        log_path = logs_dir / f"{Path(json_file).stem}.log"
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            last = ""
            # 找到最后一条 [SUMMARY]
            for ln in reversed(lines):
                if ln.find("[SUMMARY]") != -1:
                    last = ln
                    break
            if last:
                # 解析 repair_failures 数值
                import re
                m = re.search(r"repair_failures=(\d+)", last)
                if m and int(m.group(1)) == 0:
                    success_files += 1
        except Exception:
            pass
    success_rate = 100.0 * success_files / max(total_files, 1)
    print(f"联通性成功: {success_files}/{total_files} ({success_rate:.1f}%)")
    
    # 如果只做离线对比，跳过合并分割步骤，保留 temp 目录
    if os.environ.get("OPFDATA_SKIP_CHUNK", "0") == "1":
        print("\n跳过合并与分割（OPFDATA_SKIP_CHUNK=1），已保留临时 .pt 到:")
        print(str(temp_dir))
        return True

    # Step 2: 合并和分割数据集
    print("\nStep 2: 合并和分割数据集...")
    pt_files = sorted(temp_dir.glob("*.pt"))
    
    if not pt_files:
        print("错误: 没有生成.pt文件")
        return False
    
    # 加载所有数据
    all_data = []
    filtered_out = 0
    atol = float(os.environ.get("OPFDATA_MAP_ATOL", "1e-6"))
    # 原始JSON负荷缓存，避免重复读取
    raw_cache = {}
    for pt_file in tqdm(pt_files, desc="加载数据"):
        try:
            # PyTorch 2.6 默认 weights_only=True 会阻止加载包含自定义类的对象（如 PyG Data）
            # 这里明确关闭 weights_only 以加载我们自己生成的 Data 对象
            data = torch.load(pt_file, weights_only=False)
            # 中间检查：走廊→母线映射一致性（通过才纳入合并）
            ok = True
            try:
                if hasattr(data, 'y_end_pq') and hasattr(data, 'tie_corridors') and hasattr(data, 'tie_buses') and hasattr(data, 'y_bus_pq'):
                    C = data.tie_corridors.size(0)
                    B = data.tie_buses.numel()
                    if C > 0 and B > 0:
                        device = torch.device('cpu')
                        yend = data.y_end_pq.to(device)
                        tc = data.tie_corridors.to(device).long()
                        tb = data.tie_buses.to(device).long()
                        # 构造bus索引映射
                        max_bus = int(tb.max().item()) if B > 0 else -1
                        bus2idx = torch.full((max_bus + 1,), -1, dtype=torch.long, device=device)
                        bus2idx[tb] = torch.arange(B, device=device)
                        u = tc[:, 0]
                        v = tc[:, 1]
                        u_idx = bus2idx[u]
                        v_idx = bus2idx[v]
                        Pu = yend[:, 0]; Pv = yend[:, 1]; Qu = yend[:, 2]; Qv = yend[:, 3]
                        mapped = torch.zeros(B, 2, dtype=yend.dtype, device=device)
                        mapped.index_add_(0, u_idx, torch.stack([-Pu, -Qu], dim=1))
                        mapped.index_add_(0, v_idx, torch.stack([Pv, Qv], dim=1))
                        diff = torch.max(torch.abs(mapped - data.y_bus_pq.to(device))).item() if data.y_bus_pq.numel() > 0 else 0.0
                        ok = diff <= atol
                # 结构一致性（快速）：AC边shift≈0、Trafo边is_tie==0、Pdc_ratio≈|Pdc|/Smax裁剪
                if ok and hasattr(data, 'edge_attr'):
                    import numpy as _np
                    ea = data.edge_attr.cpu().numpy()
                    et = ea[:, 4]
                    shift = ea[:, 5]
                    is_tie = ea[:, 3]
                    pdc = ea[:, 6]
                    smax = ea[:, 2]
                    # AC: shift≈0
                    if _np.any(_np.abs(shift[et < 0.5]) > 1e-6):
                        ok = False
                    # Trafo: is_tie==0
                    if _np.any(is_tie[et >= 0.5] > 1e-6):
                        ok = False
                    # Pdc_ratio 近似校验
                    ratio = ea[:, 7]
                    with _np.errstate(divide='ignore', invalid='ignore'):
                        ref = _np.minimum(_np.abs(pdc) / _np.maximum(smax, 1e-8), 2.0)
                    if _np.nanmax(_np.abs(ref - ratio)) > 1e-5:
                        ok = False
                # 负荷一致性检查：data.x[:,0:2] 应与原始 JSON 的 load_link 聚合一致
                if ok and hasattr(data, 'x'):
                    # 从文件名推断原始 JSON 文件
                    base = pt_file.stem
                    if '_k' in base:
                        file_base = base.split('_k')[0]
                        raw_path = Path(config["raw_dir"]) / f"{file_base}.json"
                        if raw_path.exists():
                            if file_base not in raw_cache:
                                import json as _json
                                with open(raw_path, 'r') as f:
                                    j = _json.load(f)
                                buses = j['grid']['nodes']['bus']
                                loads = j['grid']['nodes']['load']
                                ll = j['grid']['edges']['load_link']
                                load_map = {}
                                for li, bi in zip(ll['senders'], ll['receivers']):
                                    p, q = loads[li]
                                    load_map[bi] = (
                                        load_map.get(bi, (0.0, 0.0))[0] + float(p),
                                        load_map.get(bi, (0.0, 0.0))[1] + float(q),
                                    )
                                raw_cache[file_base] = {
                                    'n_buses': len(buses),
                                    'load_map': load_map,
                                }
                            info = raw_cache[file_base]
                            # 节点数一致
                            if data.x.shape[0] == info['n_buses']:
                                # 构造 raw 负荷向量
                                import numpy as _np
                                N = info['n_buses']
                                raw_pq = _np.zeros((N, 2), dtype=float)
                                for bi, (p, q) in info['load_map'].items():
                                    raw_pq[int(bi), 0] = float(p)
                                    raw_pq[int(bi), 1] = float(q)
                                xpq = data.x[:, 0:2].cpu().numpy()
                                diff_load = float(_np.max(_np.abs(xpq - raw_pq)))
                                ok = ok and (diff_load <= atol)
                            else:
                                ok = False
            except Exception:
                ok = False
            if ok:
                all_data.append(data)
            else:
                filtered_out += 1
        except Exception as e:
            print(f"加载失败 {pt_file}: {e}")

    # 随机打乱
    import random
    random.shuffle(all_data)
    
    # 分割数据集
    n_total = len(all_data)
    n_train = int(n_total * PROCESS_CONFIG["train_ratio"])
    n_val = int(n_total * PROCESS_CONFIG["val_ratio"])
    
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    print(f"数据集划分: 训练 {len(train_data)}, 验证 {len(val_data)}, 测试 {len(test_data)}")
    if filtered_out:
        print(f"过滤不一致样本: {filtered_out}")
    
    # Step 3: 保存为chunk文件
    print("\nStep 3: 保存chunk文件...")
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # 分块保存
        chunk_size = PROCESS_CONFIG["chunk_size"]
        n_chunks = (len(split_data) + chunk_size - 1) // chunk_size
        
        for i in range(n_chunks):
            chunk = split_data[i*chunk_size:(i+1)*chunk_size]
            chunk_path = split_dir / f"chunk_{i:04d}.pt"
            torch.save(chunk, chunk_path)
        
        print(f"  {split_name}: {n_chunks} 个chunk文件")
    
    # 清理临时文件
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n数据集 {dataset_name} 处理完成！")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BC-GNN 数据生成工具（边界PQ预测）")
    parser.add_argument("action", choices=["process"],
                       help="执行的操作: process(处理数据)")
    parser.add_argument("--datasets", nargs="+", default=["ieee500"],
                       help="要处理的数据集列表")
    parser.add_argument("--overwrite", action="store_true",
                       help="覆盖已存在的处理结果")
    parser.add_argument("--processes", type=int, default=None,
                       help="并行进程数（默认使用配置或CPU核数上限）")
    
    args = parser.parse_args()
    
    if args.action == "process":
        # 处理数据集
        for dataset in args.datasets:
            if args.processes:
                PROCESS_CONFIG["num_processes"] = int(args.processes)
            success = process_dataset(dataset, args.overwrite)
            if not success:
                print(f"处理 {dataset} 失败，中止")
                return 1
    
    print("\n完成！")
    return 0


if __name__ == "__main__":
    # 置顶当前 src 到 sys.path，避免导入旧模块
    sys.path.insert(0, os.path.dirname(__file__))
    # 强制使用 spawn，避免 fork 继承旧模块缓存
    try:
        if get_start_method(allow_none=True) != 'spawn':
            set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    sys.exit(main())
