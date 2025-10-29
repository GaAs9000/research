"""V-θ 路线评估脚本"""

import argparse
import os
import sys
from typing import Dict, Tuple

import torch
from tqdm import tqdm

# 允许作为脚本直接运行：把仓库根目录加入 sys.path，使得 'gnn.*' 可导入
if __package__ is None or __package__ == "":
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

from model import BCGNN
from data import create_dataloader, load_split_data


def evaluate_model(model: BCGNN, data_loader, device: torch.device) -> Dict[str, float]:
    """评估 V-θ 模型的核心指标。"""
    model.eval()
    mae_V, mae_theta, mae_P, mae_Q = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating (V-θ)'):
            batch = batch.to(device)
            pred = model(batch)

            if 'V_pred' not in pred or 'sincos_pred' not in pred:
                continue
            if not hasattr(batch, 'y_bus_V') or not hasattr(batch, 'y_edge_sincos'):
                continue

            V_pred = pred['V_pred']
            sincos_pred = pred['sincos_pred']
            edge_pq_pred = pred.get('edge_pq')

            y_V = batch.y_bus_V.to(device)
            y_sincos = batch.y_edge_sincos.to(device)
            y_edge_pq = batch.y_edge_pq.to(device) if hasattr(batch, 'y_edge_pq') else None

            mae_V.append(torch.abs(V_pred - y_V).mean().item())

            cos_err = (sincos_pred * y_sincos).sum(dim=1).clamp(-1, 1)
            mae_theta.append(torch.rad2deg(torch.acos(cos_err).mean()).item())

            if edge_pq_pred is not None and y_edge_pq is not None:
                mae_P.append(torch.abs(edge_pq_pred[:, :2] - y_edge_pq[:, :2]).mean().item())
                mae_Q.append(torch.abs(edge_pq_pred[:, 2:] - y_edge_pq[:, 2:]).mean().item())

    def _avg(values):
        return float(sum(values) / len(values)) if values else float('nan')

    return {
        'mae_V': _avg(mae_V),
        'mae_theta_deg': _avg(mae_theta),
        'mae_P': _avg(mae_P),
        'mae_Q': _avg(mae_Q),
    }


def load_checkpoint(model_path: str, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """加载checkpoint并返回state_dict。"""
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    return state_dict, ckpt


def main():
    parser = argparse.ArgumentParser(description='评估 BC-GNN (V-θ 路线)')
    parser.add_argument('--model', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据目录（包含 train/val/test）')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help='评估划分')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== BC-GNN V-θ 评估 ===")
    print(f"模型: {args.model}")
    print(f"数据集: {args.data_dir} ({args.split})")
    print(f"设备: {device}")

    data_list = load_split_data(args.data_dir, args.split)
    if not data_list:
        raise RuntimeError(f'{args.split} 集为空')

    node_in_dim = int(data_list[0].x.size(1))
    edge_in_dim = int(data_list[0].edge_attr.size(1))

    state_dict, ckpt_meta = load_checkpoint(args.model, device)
    cfg_meta = ckpt_meta.get('config', None)
    if isinstance(cfg_meta, dict):
        hidden_dim = cfg_meta.get('hidden_dim', 64)
    elif hasattr(cfg_meta, 'hidden_dim'):
        hidden_dim = getattr(cfg_meta, 'hidden_dim')
    else:
        hidden_dim = 64

    model = BCGNN(node_features=node_in_dim, edge_features=edge_in_dim, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model = model.to(device)

    loader = create_dataloader(data_list, batch_size=1, shuffle=False)
    metrics = evaluate_model(model, loader, device)

    print("\n=== 指标 ===")
    print(f"电压MAE      : {metrics['mae_V']:.6f} p.u.")
    print(f"相角MAE      : {metrics['mae_theta_deg']:.4f}°")
    print(f"功率MAE (P) : {metrics['mae_P']:.6f} p.u.")
    print(f"功率MAE (Q) : {metrics['mae_Q']:.6f} p.u.")


if __name__ == "__main__":
    main()
