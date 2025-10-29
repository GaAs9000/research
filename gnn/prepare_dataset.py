"""
数据集准备脚本

支持两种模式：
1. 从chunk文件划分数据集
2. 从原始样本文件创建混合k值的chunk，然后划分
"""
import os
import glob
import random
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm
import torch

def prepare_mixed_chunks(source_dir, output_dir, samples_per_chunk=256, seed=42):
    """
    将不同k值的样本混合打包成chunk文件
    
    Args:
        source_dir: 包含 example_*_k*_s*.pt 文件的目录
        output_dir: 输出目录
        samples_per_chunk: 每个chunk包含的样本数
        seed: 随机种子
    
    Returns:
        bool: 是否成功创建
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有样本文件
    pattern = os.path.join(source_dir, "example_*_k*_s*.pt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"未找到样本文件: {pattern}")
        return False
    
    print(f"找到 {len(files)} 个样本文件")
    
    # 统计不同k值的分布
    k_stats = {}
    all_samples = []
    
    print("加载样本...")
    for file in tqdm(files):
        try:
            data = torch.load(file, weights_only=False)
            k = data.k if hasattr(data, 'k') else 2  # 默认k=2
            
            if k not in k_stats:
                k_stats[k] = 0
            k_stats[k] += 1
            
            all_samples.append(data)
        except Exception as e:
            print(f"加载 {file} 失败: {e}")
            continue
    
    if not all_samples:
        print("没有成功加载任何样本")
        return False
    
    print(f"\n样本统计:")
    for k, count in sorted(k_stats.items()):
        print(f"  k={k}: {count} 样本 ({100*count/len(all_samples):.1f}%)")
    
    # 设置随机种子并打乱
    random.seed(seed)
    random.shuffle(all_samples)
    print(f"\n已打乱 {len(all_samples)} 个样本")
    
    # 分批保存
    n_chunks = (len(all_samples) + samples_per_chunk - 1) // samples_per_chunk
    print(f"将创建 {n_chunks} 个chunk文件，每个包含最多 {samples_per_chunk} 个样本")
    
    for i in range(0, len(all_samples), samples_per_chunk):
        chunk = all_samples[i:i+samples_per_chunk]
        chunk_id = i // samples_per_chunk
        output_file = os.path.join(output_dir, f"chunk_{chunk_id:04d}.pt")
        torch.save(chunk, output_file)
        
        # 统计该chunk的k值分布
        chunk_k_dist = {}
        for data in chunk:
            k = data.k if hasattr(data, 'k') else 2
            chunk_k_dist[k] = chunk_k_dist.get(k, 0) + 1
        
        k_info = ", ".join([f"k{k}:{n}" for k, n in sorted(chunk_k_dist.items())])
        print(f"  Chunk {chunk_id:04d}: {len(chunk)} 样本 ({k_info})")
    
    print(f"\n混合chunk创建完成！保存到 {output_dir}")
    return True


def split_dataset(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """
    将指定目录下的 .pt 文件划分为 train, val, 和 test 三个子目录。

    Args:
        data_dir (str): 包含 .pt 文件的根目录。
        train_ratio (float): 训练集所占的比例。
        val_ratio (float): 验证集所占的比例。
        seed (int): 用于保证划分可复现的随机种子。
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "训练集和验证集比例之和不能大于等于1"
    
    print(f"开始处理数据集，目标目录: {data_dir}")

    # 检查目录是否存在
    if not os.path.isdir(data_dir):
        print(f"错误: 目录 '{data_dir}' 不存在。")
        return

    # 创建 train, val, test 子目录
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # 清理旧目录，确保从干净状态开始
    for d in [train_dir, val_dir, test_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
    print(f"已创建子目录: 'train', 'val', 'test'")

    # 查找所有 .pt 文件（排除子目录中的文件）
    all_files = glob.glob(os.path.join(data_dir, '*.pt'))
    
    if not all_files:
        print(f"警告: 在 '{data_dir}' 中未找到 '.pt' 文件。")
        return

    print(f"找到 {len(all_files)} 个 .pt 文件。")

    # 设置随机种子并打乱文件列表
    random.seed(seed)
    random.shuffle(all_files)

    # 计算划分点
    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分文件
    train_files = all_files[:n_train]
    val_files = all_files[n_train : n_train + n_val]
    test_files = all_files[n_train + n_val:]

    print(f"计划划分: {len(train_files)}->train, {len(val_files)}->val, {len(test_files)}->test")

    # 移动文件
    def move_files(files, dest_dir, desc):
        for file_path in tqdm(files, desc=desc):
            try:
                shutil.move(file_path, os.path.join(dest_dir, os.path.basename(file_path)))
            except Exception as e:
                print(f"移动文件 {file_path} 失败: {e}")
    
    move_files(train_files, train_dir, "移动训练集")
    move_files(val_files, val_dir, "移动验证集")
    move_files(test_files, test_dir, "移动测试集")

    # 验证划分结果
    print("\n验证划分结果...")
    train_files_after = glob.glob(os.path.join(train_dir, '*.pt'))
    val_files_after = glob.glob(os.path.join(val_dir, '*.pt'))
    test_files_after = glob.glob(os.path.join(test_dir, '*.pt'))
    
    print(f"  训练集: {len(train_files_after)} 文件")
    print(f"  验证集: {len(val_files_after)} 文件")
    print(f"  测试集: {len(test_files_after)} 文件")
    print(f"  总计: {len(train_files_after) + len(val_files_after) + len(test_files_after)} 文件")
    
    # 检查是否有文件丢失
    if len(train_files_after) + len(val_files_after) + len(test_files_after) != n_total:
        print(f"  ⚠️ 警告: 文件数不匹配！原始: {n_total}, 当前: {len(train_files_after) + len(val_files_after) + len(test_files_after)}")
    else:
        print(f"  ✅ 数据划分成功！")
    
    print("\n数据集划分完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="数据准备：混合打包和划分数据集")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['mix', 'split', 'both'],
        default='split',
        help="运行模式: mix(混合打包), split(划分数据集), both(先混合再划分)"
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='/home/zhangyao/renjiashen/workspace/data',
        help="原始样本文件目录（用于mix模式）"
    )
    parser.add_argument(
        '--data-dir', 
        type=str, 
        default="../data/ieee118",
        help="chunk文件目录（用于split模式）或混合后的输出目录（用于mix模式）"
    )
    parser.add_argument(
        '--samples-per-chunk',
        type=int,
        default=256,
        help="每个chunk的样本数（用于mix模式）"
    )
    parser.add_argument(
        '--train-ratio', 
        type=float, 
        default=0.7,
        help="训练集所占的比例"
    )
    parser.add_argument(
        '--val-ratio', 
        type=float, 
        default=0.15,
        help="验证集所占的比例"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="随机种子"
    )
    args = parser.parse_args()
    
    if args.mode == 'mix':
        # 只混合打包
        prepare_mixed_chunks(args.source_dir, args.data_dir, args.samples_per_chunk, args.seed)
    elif args.mode == 'split':
        # 只划分数据集
        split_dataset(args.data_dir, args.train_ratio, args.val_ratio, args.seed)
    elif args.mode == 'both':
        # 先混合打包，再划分
        success = prepare_mixed_chunks(args.source_dir, args.data_dir, args.samples_per_chunk, args.seed)
        if success:
            print("\n开始划分数据集...")
            split_dataset(args.data_dir, args.train_ratio, args.val_ratio, args.seed)
    else:
        print(f"未知模式: {args.mode}")
