"""
数据批量后处理与分块存储

本脚本负责将 `process_opfdata.py` 生成的大量独立的 .pt 文件
进行合并、随机打乱，并重新存储为适合 DataLoader 高效读取的、
规模较小的分块（chunk）文件。
"""

import torch
from pathlib import Path
import random
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

from opfdata.optimizer import OptimizedProcessor


def process_chunk(args: Tuple[List, str, int, str]):
    """
    处理单个数据块（chunk）并将其保存到文件。

    Args:
        args (Tuple): 包含以下元素的元组:
            chunk_samples (List): 当前块包含的样本列表。
            output_dir (str): 输出目录路径。
            chunk_idx (int): 当前块的索引号。
            format (str): 保存格式 ('pt' 或 'pkl')。
    """
    chunk_samples, output_dir, chunk_idx, format = args
    optimizer = OptimizedProcessor()
    
    # 构建输出文件名，例如 chunk_0001.pt
    output_path = Path(output_dir) / f"chunk_{chunk_idx:04d}.{format}"
    
    try:
        # 使用优化器来保存数据块
        optimizer.save_batch_optimized(chunk_samples, str(output_path), format=format)
    except Exception as e:
        print(f"保存块 {chunk_idx} 失败: {e}")


def main():
    """
    主函数，执行数据合并、打乱和分块存储。
    """
    # --- 配置区 ---
    # `process_opfdata.py` 生成的数据所在目录
    input_dir = "/home/zhangyao/renjiashen/workspace/data/ieee118"
    # 分块后数据的输出目录
    output_dir = "/home/zhangyao/renjiashen/workspace/data/ieee118_processed"
    # 每个分块文件包含的样本数量
    chunk_size = 512
    # 保存格式 ('pt' for PyTorch, 'pkl' for pickle)
    save_format = 'pt'
    # 使用的 CPU 核心数，最多不超过 16
    num_processes = min(cpu_count(), 16)
    
    # --- 执行区 ---
    print("=== 数据批量后处理与分块 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"分块大小: {chunk_size} 样本/块")
    print(f"保存格式: {save_format}")
    print(f"使用进程数: {num_processes}")
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 加载所有独立的 .pt 文件
    print("\n步骤 1: 加载所有样本...")
    input_files = sorted(list(Path(input_dir).glob("*.pt")))
    
    if not input_files:
        print(f"错误: 在 '{input_dir}' 中未找到 .pt 文件。请先运行 process_opfdata.py。")
        return
        
    all_samples = []
    # 使用 tqdm 显示加载进度
    for file_path in tqdm(input_files, desc="加载 .pt 文件"):
        try:
            sample = torch.load(file_path, weights_only=False)
            all_samples.append(sample)
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            continue
    
    print(f"加载完成，共 {len(all_samples)} 个样本。")
    
    # 2. 随机打乱所有样本
    print("\n步骤 2: 随机打乱样本...")
    random.shuffle(all_samples)
    print("打乱完成。")
    
    # 3. 将样本分割成多个块（chunks）
    print(f"\n步骤 3: 将样本分割成 {chunk_size} 大小的块...")
    chunks = [
        all_samples[i:i + chunk_size]
        for i in range(0, len(all_samples), chunk_size)
    ]
    print(f"分割完成，共 {len(chunks)} 个块。")
    
    # 4. 使用多进程并行保存所有块
    print("\n步骤 4: 使用多进程保存所有块...")
    
    # 为每个块准备参数
    process_args = [
        (chunk, output_dir, idx, save_format)
        for idx, chunk in enumerate(chunks)
    ]
    
    # 使用多进程池执行保存操作
    with Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示保存进度
        list(tqdm(pool.imap(process_chunk, process_args), total=len(chunks), desc="保存块"))

    print("\n=== 全部处理完成 ===")
    print(f"所有数据已成功保存到: {output_dir}")

    # 验证输出
    output_files = list(Path(output_dir).glob(f"*.{save_format}"))
    print(f"验证: 在输出目录中找到 {len(output_files)} 个分块文件。")

    if output_files:
        optimizer = OptimizedProcessor()
        try:
            # 尝试加载第一个块并报告其大小
            first_chunk = optimizer.load_batch_optimized(str(output_files[0]), format=save_format)
            print(f"示例: 第一个块包含 {len(first_chunk)} 个样本。")
        except Exception as e:
            print(f"验证加载第一个块时出错: {e}")


if __name__ == "__main__":
    main()
