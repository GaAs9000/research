"""
OPFData 数据处理流水线（V1.1，PQ-only）

说明
- 仅负责从 OPFData JSON 生成 PyG Data（V1.1 字段），不包含训练/评测逻辑。
- 为避免多进程环境导入到旧模块，强制将当前 src 目录置于 sys.path 首位。
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))  # 确保优先使用当前源码

from opfdata.processor import OPFDataProcessor
import torch
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_start_method, set_start_method
try:
    import orjson as _orjson
except Exception:
    _orjson = None
import json
from typing import Tuple, List
from tqdm import tqdm


def process_single_json(args: Tuple[str, str, str, List[int], int]) -> Tuple[int, int]:
    """
    处理单个 JSON 文件，并为其生成多种分区版本的数据。

    Args:
        args: 一个元组，包含以下参数:
            json_path (str): JSON 文件的完整路径。
            json_file (str): JSON 文件名。
            output_dir (str): 输出目录路径。
            k_values (List[int]): 需要生成的分区数量列表 (例如 [2, 3, 4])。
            seeds_per_k (int): 每个 k 值下要生成的随机版本数量。

    Returns:
        一个元组 (generated, failed)，表示成功生成和失败的文件数量。
    """
    json_path, json_file, output_dir, k_values, seeds_per_k = args
    
    # 为每个子进程设置环境变量，用于控制分区算法的行为
    # 不覆盖父进程设置；若未设置则使用容差 0.8,1.2
    os.environ.setdefault("OPFDATA_SIZE_TOL", "0.8,1.2")
    # 关闭尺寸均衡（只做连通性修复）；如需开启，改为 "1"
    os.environ.setdefault("OPFDATA_BALANCE_SIZES", "0")
    # 分区方法固定为构造式生长，不再设置 OPFDATA_METHOD
    
    processor = OPFDataProcessor()
    # 【关键修复】设置正确的k值列表
    processor.k_values = k_values
    processor.multi_k = True  # 确保使用多k值模式
    processor.reset_run_stats()
    # 为当前 JSON 设置独立日志文件
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_base = json_file.replace('.json', '')
    log_path = os.path.join(log_dir, f"{file_base}.log")
    processor.set_log_file(log_path)
    generated = 0  # 成功计数器
    failed = 0     # 失败计数器
    
    try:
        # 加载 JSON 数据
        with open(json_path, 'rb') as f:
            if _orjson is not None:
                opf_data = _orjson.loads(f.read())
            else:
                # 退回标准库
                opf_data = json.loads(f.read().decode('utf-8'))
        
        # 提取核心的网络拓扑和参数信息
        network_data = processor._extract_network_data(opf_data)
        file_base = json_file.replace('.json', '')
        
        # 遍历指定的 k 值和随机种子，生成多个分区版本
        for k in k_values:
            for seed_idx in range(seeds_per_k):
                try:
                    # 基于文件名、k 和种子索引生成一个确定性的整数种子
                    seed_str = f"{json_file}_{k}_{seed_idx}"
                    seed_int = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**31 - 1)
                    
                    # 创建分区
                    partition = processor._create_partition_dynamic(network_data, opf_data, k, seed_int)
                    if partition is None:
                        failed += 1
                        continue
                    
                    # 基于分区结果创建 PyTorch Geometric 数据对象
                    pyg_data = processor._create_pyg_data(opf_data, network_data, partition, k)
                    if pyg_data is None:
                        failed += 1
                        continue
                    
                    # 保存处理好的数据为 .pt 文件
                    output_filename = f"{file_base}_k{k}_s{seed_idx}.pt"
                    output_filepath = os.path.join(output_dir, output_filename)
                    torch.save(pyg_data, output_filepath)
                    
                    generated += 1
                
                except Exception:
                    # 捕获内层循环的异常，防止单个分区失败导致整个文件处理中断
                    failed += 1
                    continue
                    
    except Exception:
        # 如果文件加载或初步处理失败，则将所有可能的输出都计为失败
        failed += len(k_values) * seeds_per_k
    finally:
        # 每个 JSON 的汇总输出（默认仅写日志文件，不在终端打印）
        stats = processor.get_run_stats()
        summary_line = (f"[SUMMARY] {json_file}: attempts={stats['attempts']} "
                        f"repairs={stats['repairs']} repair_failures={stats['repair_failures']} "
                        f"size_violations={stats['size_violations']}")
        # 仅当 OPFDATA_STDOUT_SUMMARY=1 时输出到终端
        if os.environ.get("OPFDATA_STDOUT_SUMMARY", "0") == "1":
            print(summary_line)
        # 始终写入独立日志文件
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(summary_line + "\n")
        except Exception:
            pass

    return generated, failed


def main():
    """
    数据处理主函数。
    使用多进程并行处理 JSON 文件，为每个文件生成多种分区版本的数据。
    """
    
    # --- 配置区 ---
    # 输入 JSON 文件目录
    json_dir = "/home/zhangyao/renjiashen/workspace/gnn_data/raw/500_N_1"
    # 输出处理后 .pt 文件的目录
    output_dir = "/home/zhangyao/renjiashen/workspace/data/ieee500_n1"
    
    # 多分区生成设置
    k_values = [7, 8, 9]      # 固定为 7/8/9：权衡子问题规模与边界复杂度
    seeds_per_k = 1           # 每个 k 值下生成多少个不同的随机版本
    batch_size = None         # 处理所有15000个文件
    
    # 多进程设置
    num_processes = min(cpu_count(), 16)  # 使用的 CPU 核心数，最多不超过 16
    
    # --- 执行区 ---
    print("=== 多分区 OPF 数据处理 (多进程模式) ===")
    print(f"JSON 输入目录: {json_dir}")
    print(f"数据输出目录: {output_dir}")
    print(f"分区配置 (k值): {k_values}")
    print(f"每个k值的随机版本数: {seeds_per_k}")
    print(f"处理文件批次大小: {batch_size if batch_size else '全部'}")
    print(f"使用进程数: {num_processes}")
    
    # 创建输出目录（如果不存在）
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取待处理的 JSON 文件列表
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
    
    # 支持环境变量控制处理数量（用于调试和测试）
    max_samples = os.environ.get('BCGNN_MAX_SAMPLES')
    if max_samples:
        try:
            max_samples = int(max_samples)
            json_files = json_files[:max_samples]
            print(f"⚠️  环境变量限制: 仅处理前 {max_samples} 个文件")
        except ValueError:
            print(f"⚠️  无效的 BCGNN_MAX_SAMPLES 值: {max_samples}")
    
    if batch_size:
        json_files = json_files[:batch_size]
    
    print(f"准备处理 {len(json_files)} 个 JSON 文件，使用 {num_processes} 个进程...")
    print(f"预计输出文件总数: {len(json_files)} × {len(k_values)} × {seeds_per_k} = {len(json_files) * len(k_values) * seeds_per_k} 个")
    
    # 为多进程任务准备参数
    process_args = []
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        process_args.append((json_path, json_file, output_dir, k_values, seeds_per_k))
    
    # 使用多进程池和 tqdm 进度条来执行处理
    print("\n启动多进程处理...")
    with Pool(processes=num_processes) as pool:
        results = []
        with tqdm(total=len(process_args), desc="处理 JSON 文件", unit="file") as pbar:
            # imap 方法可以按顺序返回结果，便于实时更新进度
            for result in pool.imap(process_single_json, process_args):
                results.append(result)
                pbar.update(1)
                # 在进度条上显示实时的成功/失败统计
                total_gen = sum(r[0] for r in results)
                total_fail = sum(r[1] for r in results)
                pbar.set_postfix({
                    '已生成': total_gen,
                    '已失败': total_fail,
                    '成功率%': f"{total_gen/(total_gen+total_fail)*100:.1f}" if total_gen+total_fail > 0 else "0.0"
                })
    
    # 收集并汇总所有进程的结果
    total_generated = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)
    
    # --- 结果报告 ---
    print(f"\n=== 处理完成 ===")
    print(f"成功生成: {total_generated} 个 PyG 数据文件")
    print(f"处理失败: {total_failed} 次")
    if total_generated + total_failed > 0:
        print(f"成功率: {total_generated/(total_generated+total_failed)*100:.1f}%")
    print(f"数据输出目录: {output_dir}")
    
    # 加载一个样本文件并显示其统计信息，用于快速验证数据格式
    if total_generated > 0:
        sample_file = next(Path(output_dir).glob("*.pt"))
        sample = torch.load(sample_file, weights_only=False)
        print(f"\n样本数据统计:")
        print(f"  节点特征 (Features): {sample.x.shape}")
        print(f"  边索引 (Edges): {sample.edge_index.shape}")
        print(f"  耦合母线 (Tie buses): {len(sample.tie_buses)}")
        print(f"  走廊 (Corridors): {len(sample.tie_corridors)}")
        if hasattr(sample, 'y_corridor_pfqt'):
            print(f"  y_corridor_pfqt shape: {sample.y_corridor_pfqt.shape}")
        if hasattr(sample, 'y_bus_pq'):
            print(f"  y_bus_pq shape: {sample.y_bus_pq.shape}")


if __name__ == "__main__":
    import multiprocessing as mp
    # 在Linux上，默认的'fork'模式可能与CUDA不兼容，'spawn'更安全
    # 必须在任何其他多进程代码之前调用
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    main()
