"""
BC-GNN 预测器（V-θ 路线）：加载训练好的模型，输出边界功率与母线注入。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import sys
import os

# 导入BC-GNN模型结构（使用仓库内实现）
from gnn.model import BCGNN


class BCGNNPredictor:
    """BC-GNN预测器"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        初始化BC-GNN预测器
        
        Args:
            model_path: 训练好的BC-GNN模型权重路径
            device: 推理设备 ('auto', 'cpu', 'cuda')
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        
        # 加载模型
        self._load_model()
    
    def _get_device(self, device: str) -> torch.device:
        """获取推理设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def _load_model(self) -> None:
        """加载BC-GNN模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        try:
            # 加载模型检查点
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 提取模型状态和配置
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                # 尝试从配置中读取模型参数
                config_obj = checkpoint.get('config', {})
                # 如果config是对象，转换为字典
                if hasattr(config_obj, '__dict__'):
                    config = config_obj.__dict__
                elif isinstance(config_obj, dict):
                    config = config_obj
                else:
                    config = {}
            else:
                # 兼容直接保存模型状态的情况
                model_state = checkpoint
                config = {}
            
            # 从模型状态中推断模型配置
            model_config = self._infer_model_config(model_state, config)
            
            # 实例化模型
            self.model = BCGNN(**model_config)
            
            # 加载权重
            self.model.load_state_dict(model_state)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ 成功加载BC-GNN模型: {self.model_path}")
            print(f"📍 推理设备: {self.device}")
            print(f"⚙️  模型配置: {model_config}")
            
        except Exception as e:
            raise RuntimeError(f"加载BC-GNN模型失败: {e}")
    
    def _infer_model_config(self, model_state: Dict, config: Dict) -> Dict:
        """从模型状态推断模型配置"""
        model_config = {}
        
        # 从state_dict推断参数
        if 'node_encoder.weight' in model_state:
            node_features = model_state['node_encoder.weight'].shape[1]
            hidden_dim = model_state['node_encoder.weight'].shape[0]
            model_config['node_features'] = node_features
            model_config['hidden_dim'] = hidden_dim
        
        # 从配置或默认值设置其他参数
        # V1.1 edge_attr: 10 维
        model_config['edge_features'] = config.get('edge_features', 10)
        model_config['n_mpnn_layers'] = config.get('n_mpnn_layers', 4)
        model_config['jk_mode'] = config.get('jk_mode', 'cat')
        model_config['use_voltage_prior'] = config.get('use_voltage_prior', True)
        model_config['use_global'] = config.get('use_global', False)
        
        # 电压统计量
        voltage_stats = config.get('voltage_stats')
        if voltage_stats:
            model_config['voltage_stats'] = voltage_stats
        
        return model_config
    
    def predict(self, pyg_data: Any) -> Dict:
        """
        对PyG数据进行BC-GNN预测
        
        Args:
            pyg_data: PyG Data对象，包含完整的图结构和特征
            
        Returns:
            预测结果字典：
            {
                'success': bool,
                'boundary_pq': {bus_id: (P_bd, Q_bd)},
                'corridor_ports': {(u,v): (Pu, Pv, Qu, Qv)},
                'error': 错误信息 (如果失败)
            }
        """
        if self.model is None:
            return {
                'success': False,
                'error': "模型未加载，请先调用_load_model()"
            }
        
        try:
            # 将数据移动到推理设备
            pyg_data = pyg_data.to(self.device)
            
            # 前向推理
            with torch.no_grad():
                predictions = self.model(pyg_data)
            
            # 端口级预测
            if 'corridor_pfqt' not in predictions:
                raise RuntimeError('模型未输出 corridor_pfqt（端口级 [pf_u,pt_v,qf_u,qt_v]）')
            end_ports = predictions['corridor_pfqt']  # [C,4]
            # 将端口映射为母线边界注入（按 tie_buses 顺序）：u端 -Pu，v端 +Pv
            if not (hasattr(pyg_data, 'tie_corridors') and hasattr(pyg_data, 'tie_buses')):
                raise RuntimeError('缺少 tie_corridors / tie_buses 字段，无法做母线映射')
            device = end_ports.device
            tie_corr = pyg_data.tie_corridors.to(device)
            tie_buses = pyg_data.tie_buses.to(device)
            B = tie_buses.numel()
            bus2idx = torch.full((int(tie_buses.max().item()) + 1,), -1, dtype=torch.long, device=device)
            bus2idx[tie_buses.long()] = torch.arange(B, device=device)
            agg = torch.zeros(B, 2, dtype=torch.float32, device=device)
            u = tie_corr[:, 0].long(); v = tie_corr[:, 1].long()
            Pu, Pv = end_ports[:, 0], end_ports[:, 1]
            Qu, Qv = end_ports[:, 2], end_ports[:, 3]
            idx_u = bus2idx[u]
            idx_v = bus2idx[v]
            agg.index_add_(0, idx_u, torch.stack([-Pu, -Qu], dim=-1))
            agg.index_add_(0, idx_v, torch.stack([+Pv, +Qv], dim=-1))
            bus_pq_pred = agg  # [B,2]
            # 打包为字典（bus_id -> (P,Q)）
            boundary_pq = {}
            buses = pyg_data.tie_buses.cpu().tolist()
            for i, b in enumerate(buses):
                P, Q = bus_pq_pred[i].detach().cpu().tolist()
                boundary_pq[int(b)] = (float(P), float(Q))

            return {
                'success': True,
                'boundary_pq': boundary_pq,
                'corridor_pfqt': end_ports.detach().cpu().numpy().tolist(),
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"BC-GNN预测失败: {e}"
            }
    
    def predict_batch(self, pyg_data_list) -> list:
        """
        批量预测（暂时简单循环，未来可优化为真正的batch推理）
        
        Args:
            pyg_data_list: PyG Data对象列表
            
        Returns:
            预测结果列表
        """
        results = []
        for pyg_data in pyg_data_list:
            result = self.predict(pyg_data)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {}
        
        # 统计模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_path': self.model_path,
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # 假设float32
        }
