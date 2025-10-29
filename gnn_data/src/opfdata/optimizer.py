"""
OPFData 处理优化

本模块提供了优化的数据处理和存储方案，
旨在提升 GPU 加速效果并减少数据加载时间。
"""

import torch
import pickle
import h5py
import numpy as np
from typing import List, Dict
from torch_geometric.data import Data, Batch
import os
import time


class OptimizedProcessor:
    """
    一个经过优化的数据处理器，支持 GPU 加速和快速 I/O。
    """
    
    def __init__(self):
        """初始化处理器，并检测可用的计算设备（CUDA GPU 或 CPU）。"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"优化处理器使用设备: {self.device}")
    
    def save_batch_optimized(self, samples: List[Data], output_path: str, format='pt'):
        """
        以优化的格式保存一批样本。

        支持的格式：
        - 'pkl': Pickle 格式，序列化 Python 对象，适合小数据集，速度快。
        - 'h5': HDF5 格式，分层数据格式，适合大数据集，支持压缩。
        - 'pt': PyTorch 原生格式，直接保存 Tensor 对象。

        Args:
            samples (List[Data]): 一个包含 PyG Data 对象的列表。
            output_path (str): 输出文件路径。
            format (str): 保存格式。
        """
        if format == 'pkl':
            with open(output_path, 'wb') as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
                
        elif format == 'h5':
            with h5py.File(output_path, 'w') as f:
                for i, sample in enumerate(samples):
                    group = f.create_group(f'sample_{i}')
                    # 将每个 tensor 转换为 numpy array 并使用 gzip 压缩存储
                    group.create_dataset('x', data=sample.x.cpu().numpy(), compression='gzip')
                    group.create_dataset('edge_index', data=sample.edge_index.cpu().numpy(), compression='gzip')
                    group.create_dataset('edge_attr', data=sample.edge_attr.cpu().numpy(), compression='gzip')
                    group.create_dataset('y', data=sample.y.cpu().numpy(), compression='gzip')
                    group.create_dataset('tie_buses', data=sample.tie_buses.cpu().numpy(), compression='gzip')
                    group.create_dataset('tie_lines', data=sample.tie_lines.cpu().numpy(), compression='gzip')
                    
        elif format == 'pt':
            # 直接使用 PyTorch 的 save 函数
            torch.save(samples, output_path)
    
    def load_batch_optimized(self, file_path: str, format='pkl', to_gpu=False) -> List[Data]:
        """
        从优化格式的文件中加载一批样本。

        Args:
            file_path (str): 输入文件路径。
            format (str): 文件格式 ('pkl', 'h5', 'pt')。
            to_gpu (bool): 是否在加载后立即将数据移动到 GPU。

        Returns:
            List[Data]: 一个包含加载的 PyG Data 对象的列表。
        """
        if format == 'pkl':
            with open(file_path, 'rb') as f:
                samples = pickle.load(f)
                
        elif format == 'h5':
            samples = []
            with h5py.File(file_path, 'r') as f:
                for key in sorted(f.keys(), key=lambda x: int(x.split('_')[1])):
                    group = f[key]
                    sample = Data(
                        x=torch.tensor(group['x'][...], dtype=torch.float32),
                        edge_index=torch.tensor(group['edge_index'][...], dtype=torch.long),
                        edge_attr=torch.tensor(group['edge_attr'][...], dtype=torch.float32),
                        y=torch.tensor(group['y'][...], dtype=torch.float32),
                        tie_buses=torch.tensor(group['tie_buses'][...], dtype=torch.long),
                        tie_lines=torch.tensor(group['tie_lines'][...], dtype=torch.long)
                    )
                    samples.append(sample)
                    
        elif format == 'pt':
            # 加载 .pt 文件时，默认先映射到 CPU，避免直接加载到 GPU 导致显存问题
            samples = torch.load(file_path, map_location='cpu')
        
        # 如果需要，将加载的样本批量移动到 GPU
        if to_gpu and torch.cuda.is_available():
            samples = [sample.to(self.device) for sample in samples]
            
        return samples
    
    def create_gpu_batch(self, samples: List[Data]) -> Batch:
        """
        从样本列表创建一个在 GPU 上优化的 PyG Batch 对象。

        Args:
            samples (List[Data]): PyG Data 对象列表。

        Returns:
            Batch: 一个在 GPU 上的 PyG Batch 对象。
        """
        # 先将所有样本移动到 GPU
        gpu_samples = [sample.to(self.device) for sample in samples]
        
        # 使用 from_data_list 方法将它们合并成一个大图 (Batch)
        batch = Batch.from_data_list(gpu_samples)
        
        return batch
    
    def benchmark_formats(self, samples: List[Data], test_file: str = '/tmp/test_format'):
        """
        对不同的存储格式进行基准测试。

        测试指标包括：保存时间、加载时间、文件大小、到 GPU 的传输时间。

        Args:
            samples (List[Data]): 用于测试的样本数据。
            test_file (str): 测试文件的基本路径。
        """
        formats = ['pkl', 'h5', 'pt']
        results = {}
        
        print("=== 存储格式基准测试 ===")
        
        for fmt in formats:
            file_path = f"{test_file}.{fmt}"
            
            # 测试保存时间
            start_time = time.time()
            self.save_batch_optimized(samples, file_path, format=fmt)
            save_time = time.time() - start_time
            
            # 测试文件大小
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为 MB
            
            # 测试加载时间
            start_time = time.time()
            loaded_samples = self.load_batch_optimized(file_path, format=fmt)
            load_time = time.time() - start_time
            
            # 测试到 GPU 的传输时间
            gpu_time = 0
            if torch.cuda.is_available():
                start_time = time.time()
                gpu_samples = [sample.to(self.device) for sample in loaded_samples]
                gpu_time = time.time() - start_time
            
            results[fmt] = {
                'save_time': save_time,
                'load_time': load_time,
                'gpu_time': gpu_time,
                'file_size': file_size
            }
            
            print(f"{fmt.upper()}: 保存 {save_time:.3f}s, 加载 {load_time:.3f}s, "
                  f"GPU传输 {gpu_time:.3f}s, 文件大小 {file_size:.2f}MB")
            
            # 清理测试文件
            os.remove(file_path)
        
        # 推荐综合性能最佳的格式
        best_overall = min(results.keys(), 
                           key=lambda x: results[x]['save_time'] + results[x]['load_time'])
        print(f"\n💡 推荐格式: {best_overall.upper()} (综合读写性能最佳)")
        
        return results


class DatasetOptimizer:
    """Optimize entire dataset processing and storage."""
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.optimizer = OptimizedProcessor()
    
    def process_and_save_chunks(self, json_dir: str, output_dir: str, format: str = 'pkl'):
        """Process dataset in chunks to optimize memory usage.
        
        Args:
            json_dir: Directory with JSON files
            output_dir: Output directory for processed chunks
            format: Storage format ('pkl', 'h5', 'pt')
        """
        from opfdata.processor import OPFDataProcessor
        
        os.makedirs(output_dir, exist_ok=True)
        
        processor = OPFDataProcessor()
        json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
        
        print(f"Processing {len(json_files)} files in chunks of {self.chunk_size}")
        print(f"Output format: {format}")
        
        chunk_id = 0
        total_samples = 0
        
        for i in range(0, len(json_files), self.chunk_size):
            chunk_files = json_files[i:i + self.chunk_size]
            chunk_samples = []
            
            print(f"\nProcessing chunk {chunk_id}: files {i}-{min(i+self.chunk_size-1, len(json_files)-1)}")
            
            # Process files in current chunk
            for json_file in chunk_files:
                json_path = os.path.join(json_dir, json_file)
                try:
                    samples = processor.process_single_json(json_path)
                    chunk_samples.extend(samples)
                except Exception as e:
                    print(f"Failed to process {json_file}: {e}")
                    continue
            
            # Save chunk
            if chunk_samples:
                output_path = os.path.join(output_dir, f'chunk_{chunk_id:04d}.{format}')
                self.optimizer.save_batch_optimized(chunk_samples, output_path, format)
                
                total_samples += len(chunk_samples)
                print(f"Saved {len(chunk_samples)} samples to {output_path}")
            
            chunk_id += 1
        
        print(f"\n✅ Processing complete: {total_samples} samples in {chunk_id} chunks")
        
        # Create metadata file
        metadata = {
            'total_samples': total_samples,
            'num_chunks': chunk_id,
            'chunk_size': self.chunk_size,
            'format': format,
            'files_processed': len(json_files)
        }
        
        metadata_path = os.path.join(output_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Metadata saved to {metadata_path}")
        
        return metadata