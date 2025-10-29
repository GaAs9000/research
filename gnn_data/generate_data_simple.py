"""
简化版数据生成脚本（用于V-θ路线）
"""
import sys
sys.path.insert(0, 'gnn_data/src')
from pathlib import Path
from opfdata.processor import OPFDataProcessor
import torch
from tqdm import tqdm
import os

# 配置
RAW_DIR = Path('gnn_data/raw/500/500')
OUTPUT_DIR = Path('data/ieee500_vtheta')
NUM_SAMPLES = 100  # 先生成100个样本测试
CHUNK_SIZE = 50

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'train').mkdir(exist_ok=True)
(OUTPUT_DIR / 'val').mkdir(exist_ok=True)  
(OUTPUT_DIR / 'test').mkdir(exist_ok=True)

# 初始化处理器
processor = OPFDataProcessor()
processor._log_level = "ERROR"

# 获取JSON文件
json_files = sorted(list(RAW_DIR.glob('*.json')))[:NUM_SAMPLES]
print(f"找到 {len(json_files)} 个JSON文件")

# 生成数据
all_data = []
success = 0
failed = 0

print(f"\n生成PyG数据...")
for json_file in tqdm(json_files):
    try:
        data_list = processor.process_single_json(str(json_file))
        all_data.extend(data_list)
        success += 1
    except Exception as e:
        failed += 1
        if failed <= 3:
            print(f"  失败: {json_file.name} - {e}")

print(f"\n生成统计:")
print(f"  成功: {success}/{len(json_files)}")
print(f"  失败: {failed}/{len(json_files)}")
print(f"  总样本数: {len(all_data)}")

if len(all_data) == 0:
    print("❌ 没有生成任何数据")
    sys.exit(1)

# 划分数据集（90/5/5）
import random
random.shuffle(all_data)
n = len(all_data)
n_train = int(n * 0.9)
n_val = int(n * 0.05)

train_data = all_data[:n_train]
val_data = all_data[n_train:n_train+n_val]
test_data = all_data[n_train+n_val:]

print(f"\n数据集划分:")
print(f"  训练集: {len(train_data)}")
print(f"  验证集: {len(val_data)}")
print(f"  测试集: {len(test_data)}")

# 保存为chunk
def save_chunks(data_list, split_name):
    for i in range(0, len(data_list), CHUNK_SIZE):
        chunk = data_list[i:i+CHUNK_SIZE]
        chunk_path = OUTPUT_DIR / split_name / f'chunk_{i//CHUNK_SIZE}.pt'
        torch.save(chunk, chunk_path)
    print(f"  {split_name}: 保存了 {len(list((OUTPUT_DIR/split_name).glob('*.pt')))} 个chunk")

print(f"\n保存数据...")
save_chunks(train_data, 'train')
save_chunks(val_data, 'val')
save_chunks(test_data, 'test')

print(f"\n✅ 数据生成完成！")
print(f"输出目录: {OUTPUT_DIR}")
