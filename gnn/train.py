"""
训练-评估（支持 PQ 和 V-θ 路线，V1.1 对齐，支持大批次 + 编译加速）

要点
- PQ 路线：端口级监督 + 容量守护 + 母线一致性约束
- V-θ 路线：电压 Huber 损失 + 相角差 MSE 损失 + 可选容量约束
- 大 batch：collate 挂 bus_ptr/corr_ptr；损失按"逐图均值→跨图均值"做公平聚合。
- 加速：AMP(bf16)、非阻塞搬运、pin_memory / persistent_workers / prefetch_factor、torch.compile(mode='reduce-overhead', dynamic=True)。
- 校验：根据路线检查必要字段（PQ: y_bus_pq, V-θ: y_bus_V, y_edge_sincos）
"""

import os
import sys
# 允许作为脚本直接运行：把仓库根目录加入 sys.path，使得 'gnn.*' 可导入
if __package__ is None or __package__ == "":
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext
from torch_geometric.utils import scatter

from config import cfg
from model import BCGNN
from data import load_data, create_dataloader




# 已移除非PQ路线相关的损失与不确定性加权，仅保留 PQ 路线。

"""
注：已删除约170行物理约束相关代码，包括：
- 电压约束计算
- 角度约束计算  
- 走廊级导纳聚合
- 潮流方程计算（P_ij, Q_ij, S_ij）
- 容量约束检查
- 物理权重调度机制
"""

# --- 设备与日志设置 ---


def setup_ddp():
    """DDP初始化，返回 (is_ddp, rank, local_rank, world_size, device)"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        return True, rank, local_rank, world_size, device
    else:
        # 单卡或CPU模式
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return False, 0, 0, 1, device

def setup_device():
    """设置GPU设备"""
    if not torch.cuda.is_available():
        cfg.device = 'cpu'
        cfg.use_multi_gpu = False
        print("⚠️  CUDA不可用，使用CPU训练")
        return cfg.device, None
    
    # GPU信息
    print(f"🚀 检测到 {torch.cuda.device_count()} 块GPU")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} ({mem_gb:.1f}GB)")
    
    # 设置主GPU
    primary_gpu = cfg.gpu_ids[0] if cfg.gpu_ids else 0
    torch.cuda.set_device(primary_gpu)
    print(f"📍 主GPU: GPU {primary_gpu}")
    
    # 清空GPU缓存
    torch.cuda.empty_cache()
    
    device = f'cuda:{primary_gpu}'
    return device, cfg.gpu_ids if cfg.use_multi_gpu else None


def setup_logging():
    """设置日志系统"""
    # 创建logger
    logger = logging.getLogger('BCGNN')
    logger.setLevel(logging.INFO)
    
    # 文件handler
    fh = logging.FileHandler(os.path.join(cfg.log_dir, 'train.log'))
    fh.setLevel(logging.INFO)
    
    # 控制台handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def set_seed(seed):
    """设置随机种子"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss_vtheta(pred, batch, config, rank=0):
    """
    V-θ 路线损失函数（精简版：V + θ + recon）

    Args:
        pred: 模型预测字典 {'V_pred': [B], 'sincos_pred': [E_tie,2], 'edge_pq': [E_tie,4]}
        batch: PyG Batch，包含标签 y_bus_V, y_edge_sincos, y_edge_pq
        config: 配置对象
        rank: DDP rank

    Returns:
        loss: 总损失
        loss_dict: 损失字典（用于记录）
    """
    device = batch.x.device

    # ========== 1. 电压损失：Huber(V_pred, y_bus_V) ==========
    V_pred = pred['V_pred']  # [B]
    y_V = batch.y_bus_V
    if y_V.device != device:
        y_V = y_V.to(device, non_blocking=True)  # [B]
    loss_V = F.huber_loss(V_pred, y_V, delta=0.01)

    # ========== 2. 相角差损失：MSE(sincos_pred, y_edge_sincos) ==========
    sincos_pred = pred['sincos_pred']  # [E_tie, 2]
    y_sincos = batch.y_edge_sincos
    if y_sincos.device != device:
        y_sincos = y_sincos.to(device, non_blocking=True)  # [E_tie, 2]
    loss_theta = F.mse_loss(sincos_pred, y_sincos)

    # ========== 3. 功率重构一致性损失：L1(edge_pq, y_edge_pq) ==========
    edge_pq_pred = pred['edge_pq']  # [E_tie, 4]
    y_edge_pq = batch.y_edge_pq
    if y_edge_pq.device != device:
        y_edge_pq = y_edge_pq.to(device, non_blocking=True)  # [E_tie, 4]
    loss_recon = F.l1_loss(edge_pq_pred, y_edge_pq)

    # ========== 总损失 ==========
    lambda_theta = getattr(config, 'lambda_theta', 1.0)
    lambda_recon = getattr(config, 'lambda_recon', 0.1)

    total_loss = loss_V + lambda_theta * loss_theta + lambda_recon * loss_recon

    loss_dict = {
        'loss_v': float(loss_V.detach().item()),  # 统一使用小写键名
        'loss_theta': float(loss_theta.detach().item()),
        'loss_recon': float(loss_recon.detach().item()),
        'total': float(total_loss.detach().item())
    }

    # 4. 可选：容量约束（按线版本）
    if getattr(config, 'use_capacity_constraint', False) and 'edge_pq' in pred:
        # 使用按线预测的功率
        Pf = edge_pq_pred[:, 0]  # [E_tie]
        Qf = edge_pq_pred[:, 2]  # [E_tie]
        S_pred = torch.sqrt(Pf**2 + Qf**2 + 1e-12)

        # 获取每条线的容量上限
        if hasattr(batch, 'tie_edge_indices') and batch.tie_edge_indices is not None:
            tie_edge_indices = batch.tie_edge_indices
            if tie_edge_indices.device != device:
                tie_edge_indices = tie_edge_indices.to(device, non_blocking=True)
            S_max = batch.edge_attr[tie_edge_indices, 2]  # edge_attr[:, 2] = S_max
        else:
            # 回退：使用默认值
            S_max = torch.ones_like(S_pred) * 1.0

        # 容量守护损失：ReLU(S_pred - alpha * S_max)
        alpha = getattr(config, 'capacity_alpha', 0.95)
        loss_cap = torch.relu(S_pred - alpha * S_max).mean()

        # 加到总损失
        lambda_cap = getattr(config, 'lambda_capacity', 0.1)
        total_loss = total_loss + lambda_cap * loss_cap

        loss_dict['loss_capacity'] = float(loss_cap.detach().item())
        loss_dict['total'] = float(total_loss.detach().item())

    # 5. 计算 MAE 指标（用于监控）
    with torch.no_grad():
        # 电压MAE
        mae_V = torch.abs(V_pred - y_V).mean()
        loss_dict['mae_V'] = float(mae_V.item())

        # 角度误差（单位圆上的角距离）
        cos_err = (sincos_pred * y_sincos).sum(dim=1).clamp(-1, 1)
        theta_err = torch.acos(cos_err)  # radians
        mae_theta_deg = torch.rad2deg(theta_err.mean())
        loss_dict['mae_theta_deg'] = float(mae_theta_deg.item())

        # 功率重构MAE
        mae_P = torch.abs(edge_pq_pred[:, :2] - y_edge_pq[:, :2]).mean()
        mae_Q = torch.abs(edge_pq_pred[:, 2:] - y_edge_pq[:, 2:]).mean()
        loss_dict['mae_P'] = float(mae_P.item())
        loss_dict['mae_Q'] = float(mae_Q.item())

    return total_loss, loss_dict


def train_epoch_simple(model, train_loader, optimizer, supervised_loss_fn, epoch, writer, logger, rank=0):
    """训练一个epoch（支持AMP、梯度累积和DDP）"""
    model.train()
    
    # AMP和梯度累积设置
    use_amp = getattr(cfg, 'use_amp', False)
    amp_dtype = torch.bfloat16 if getattr(cfg, 'amp_dtype', 'fp16') == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    accum_steps = getattr(cfg, 'accum_steps', 1)
    clip_params = [param for group in optimizer.param_groups for param in group['params']]
    enable_param_nan_check = getattr(cfg, 'enable_nan_param_check', False)
    named_params = list(model.named_parameters()) if enable_param_nan_check else ()
    
    losses = []
    v_losses = []
    theta_losses = []
    recon_losses = []
    capacity_losses = []
    mae_V_list = []
    mae_theta_deg_list = []
    mae_P_list = []
    mae_Q_list = []
    
    # 在epoch开始时清空梯度
    optimizer.zero_grad(set_to_none=True)
    
    # 只在rank0显示进度条
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', disable=(rank != 0))
    
    for batch_idx, batch in enumerate(progress_bar):
        # === 数据输入检测 ===
        try:
            # 检测输入数据
            input_issues = []
            if torch.isnan(batch.x).any():
                input_issues.append("x:NaN")
            if torch.isinf(batch.x).any():
                input_issues.append("x:Inf")
            if torch.isnan(batch.edge_attr).any():
                input_issues.append("edge_attr:NaN")
            if torch.isinf(batch.edge_attr).any():
                input_issues.append("edge_attr:Inf")
            
            if input_issues and rank == 0:
                logger.error(f"输入数据异常 batch {batch_idx}: {', '.join(input_issues)}")
                logger.error(f"  - batch size: {batch.num_graphs if hasattr(batch, 'num_graphs') else 'unknown'}")
                logger.error(f"  - num_nodes: {batch.x.shape[0]}, num_edges: {batch.edge_attr.shape[0]}")
                continue  # 跳过异常数据
        except Exception as e:
            if rank == 0:
                logger.error(f"数据检测出错 batch {batch_idx}: {e}")
            continue
        
        batch = batch.to(cfg.device, non_blocking=True)
        
        # DDP no_sync优化：减少梯度同步次数
        is_accumulating = (batch_idx + 1) % accum_steps != 0
        sync_ctx = (model.no_sync() if (hasattr(model, "no_sync") and 
                    is_accumulating and accum_steps > 1) else nullcontext())
        
        with sync_ctx:
            # 混合精度前向传播
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                # === 模型前向传播 ===
                try:
                    pred = model(batch)
                except Exception as e:
                    # 针对 CUDA 稀疏算子不支持 bf16 的退避方案：降级到 fp16 再试一次
                    msg = str(e)
                    tried_fp16 = False
                    if use_amp and amp_dtype == torch.bfloat16 and "addmm_sparse_cuda" in msg:
                        try:
                            tried_fp16 = True
                            with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
                                pred = model(batch)
                        except Exception as e2:
                            if rank == 0:
                                logger.error(f"模型前向传播失败 batch {batch_idx} (fp16 fallback): {e2}")
                            # 保存出错的batch到logs目录
                            if rank == 0:
                                debug_path = os.path.join(cfg.log_dir, f'debug_batch_{batch_idx}_epoch_{epoch}.pt')
                                torch.save(batch, debug_path)
                                logger.error(f"问题batch已保存至: {debug_path}")
                            continue
                    else:
                        if rank == 0:
                            logger.error(f"模型前向传播失败 batch {batch_idx}: {e}")
                            # 保存出错的batch到logs目录
                            debug_path = os.path.join(cfg.log_dir, f'debug_batch_{batch_idx}_epoch_{epoch}.pt')
                            torch.save(batch, debug_path)
                            logger.error(f"问题batch已保存至: {debug_path}")
                        continue
                
                if 'V_pred' not in pred or 'sincos_pred' not in pred:
                    raise RuntimeError("V-θ路线缺少必要字段：V_pred 或 sincos_pred")
                if not hasattr(batch, 'y_bus_V') or not hasattr(batch, 'y_edge_sincos'):
                    raise RuntimeError("V-θ路线缺少标签：y_bus_V 或 y_edge_sincos")

                loss, loss_dict = compute_loss_vtheta(pred, batch, cfg, rank)
                # 梯度累积：损失除以累积步数
                loss = loss / accum_steps
                
                # === 预测值NaN检测 ===
                pred_issues = []
                for key, tensor in pred.items():
                    if isinstance(tensor, torch.Tensor):
                        if torch.isnan(tensor).any():
                            pred_issues.append(f"{key}:NaN")
                        elif torch.isinf(tensor).any():
                            pred_issues.append(f"{key}:Inf")
                
                if pred_issues:
                    if rank == 0:  # 只在主进程记录错误
                        logger.error(f"预测值异常: {', '.join(pred_issues)} 在batch {batch_idx}, epoch {epoch}")
                    # 保存NaN预测的batch
                    if rank == 0:
                        debug_path = os.path.join(cfg.log_dir, f'nan_pred_batch_{batch_idx}_epoch_{epoch}.pt')
                        torch.save(batch, debug_path)
                        logger.error(f"NaN预测batch已保存至: {debug_path}")
                
                # NaN检测和保护
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        logger.warning(f"检测到NaN/Inf损失在batch {batch_idx}, epoch {epoch}，跳过此batch")
                        logger.warning(f"  损失值: {loss.item()}")
                        debug_path = os.path.join(cfg.log_dir, f'nan_loss_batch_{batch_idx}_epoch_{epoch}.pt')
                        torch.save(batch, debug_path)
                        logger.warning(f"NaN损失batch已保存至: {debug_path}")
                    continue
            
            # 反向传播（使用scaler处理混合精度）
            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # 每accum_steps步或最后一批时更新参数
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            if amp_dtype == torch.float16:
                # 梯度裁剪（统一裁剪所有参数组）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
                
                # 移除了physics_alpha相关的梯度限制
                
                # 参数更新
                scaler.step(optimizer)
                scaler.update()
            else:
                # bf16不需要scaler；同样裁剪所有参数组
                torch.nn.utils.clip_grad_norm_(clip_params, cfg.grad_clip)
                
                # 移除了physics_alpha相关的梯度限制
                
                optimizer.step()
            
            # 限制uncertainty参数范围，防止权重失控
            if hasattr(supervised_loss_fn, 'u'):
                with torch.no_grad():
                    supervised_loss_fn.u.clamp_(-3.0, 3.0)
            
            # 移除了physics_alpha参数约束
            
            # === 参数NaN检测 ===
            if enable_param_nan_check:
                nan_params = []
                for name, param in named_params:
                    if torch.isnan(param).any():
                        nan_params.append(name)
                    elif torch.isinf(param).any():
                        nan_params.append(f"{name}(Inf)")

                if nan_params and rank == 0:  # 只在rank0记录
                    logger.error(f"参数变成NaN/Inf: {', '.join(nan_params[:5])}{'...' if len(nan_params) > 5 else ''}")
                    logger.error(f"在 epoch {epoch}, batch {batch_idx} 参数更新后发现异常")
                    # 检查梯度范数
                    grad_norm = torch.nn.utils.clip_grad_norm_(clip_params, float('inf'))
                    logger.error(f"梯度范数: {grad_norm:.6f}")
                    # 可以选择提前终止或者重置参数
                
            optimizer.zero_grad(set_to_none=True)
        
        # 记录
        losses.append(loss_dict['total'])
        v_losses.append(loss_dict['loss_v'])
        theta_losses.append(loss_dict.get('loss_theta', 0.0))
        recon_losses.append(loss_dict.get('loss_recon', 0.0))
        if 'loss_capacity' in loss_dict:
            capacity_losses.append(loss_dict['loss_capacity'])
        if 'mae_V' in loss_dict:
            mae_V_list.append(loss_dict['mae_V'])
        if 'mae_theta_deg' in loss_dict:
            mae_theta_deg_list.append(loss_dict['mae_theta_deg'])
        if 'mae_P' in loss_dict:
            mae_P_list.append(loss_dict['mae_P'])
        if 'mae_Q' in loss_dict:
            mae_Q_list.append(loss_dict['mae_Q'])
        
        # 更新进度条（根据路线显示不同指标）
        postfix_dict = {
            'L_total': f"{loss_dict['total']:.2e}",
            'L_V': f"{loss_dict.get('loss_v', 0.0):.2e}",
            'L_θ': f"{loss_dict.get('loss_theta', 0.0):.2e}",
            'L_recon': f"{loss_dict.get('loss_recon', 0.0):.2e}",
            'MAE_V': f"{loss_dict.get('mae_V', 0.0):.4f}",
            'MAE_θ°': f"{loss_dict.get('mae_theta_deg', 0.0):.2f}",
            'MAE_P': f"{loss_dict.get('mae_P', 0.0):.3f}",
        }
        if 'loss_capacity' in loss_dict:
            postfix_dict['L_cap'] = f"{loss_dict['loss_capacity']:.2e}"
        
        # 定期显示Kendall统计（每100个batch，仅适用于传统路线）
        if batch_idx % 100 == 0 and 'kendall_stats' in loss_dict:
            stats = loss_dict['kendall_stats']
            if 'eff_weight' in stats:
                postfix_dict['w_v'] = f"{stats['eff_weight'][0]:.2f}"
                postfix_dict['w_θ'] = f"{stats['eff_weight'][1]:.2f}"
        
        progress_bar.set_postfix(postfix_dict)
        
        # TensorBoard记录
        global_step = epoch * len(train_loader) + batch_idx
        if batch_idx % cfg.log_interval == 0:
            writer.add_scalar('Train/Loss', loss_dict['total'], global_step)
            writer.add_scalar('Train/Loss_V', loss_dict['loss_v'], global_step)
            writer.add_scalar('Train/Loss_theta', loss_dict.get('loss_theta', 0.0), global_step)
            writer.add_scalar('Train/Loss_recon', loss_dict.get('loss_recon', 0.0), global_step)
            if 'loss_capacity' in loss_dict:
                writer.add_scalar('Train/Loss_capacity', loss_dict['loss_capacity'], global_step)
            if 'mae_V' in loss_dict:
                writer.add_scalar('Train/MAE_V', loss_dict['mae_V'], global_step)
            if 'mae_theta_deg' in loss_dict:
                writer.add_scalar('Train/MAE_theta_deg', loss_dict['mae_theta_deg'], global_step)
            if 'mae_P' in loss_dict:
                writer.add_scalar('Train/MAE_P', loss_dict['mae_P'], global_step)
            if 'mae_Q' in loss_dict:
                writer.add_scalar('Train/MAE_Q', loss_dict['mae_Q'], global_step)
    
    avg_loss = np.mean(losses)
    avg_v_loss = np.mean(v_losses) if v_losses else 0.0
    avg_theta_loss = np.mean(theta_losses) if theta_losses else 0.0
    avg_recon_loss = np.mean(recon_losses) if recon_losses else 0.0
    avg_capacity_loss = np.mean(capacity_losses) if capacity_losses else 0.0
    avg_mae_P = float(np.mean(mae_P_list)) if mae_P_list else 0.0
    avg_mae_Q = float(np.mean(mae_Q_list)) if mae_Q_list else 0.0
    avg_mae_V = float(np.mean(mae_V_list)) if mae_V_list else 0.0
    avg_mae_theta = float(np.mean(mae_theta_deg_list)) if mae_theta_deg_list else 0.0
    logger.info(
        f"Epoch {epoch} - Train Loss: {avg_loss:.2e} "
        f"(L_V: {avg_v_loss:.2e}, L_theta: {avg_theta_loss:.2e}, L_recon: {avg_recon_loss:.2e}, "
        f"L_cap: {avg_capacity_loss:.2e}, MAE_V: {avg_mae_V:.4f}, MAE_θ°: {avg_mae_theta:.2f}, "
        f"MAE_P: {avg_mae_P:.3f}, MAE_Q: {avg_mae_Q:.3f})"
    )
    
    return avg_loss


def validate_simple(model, val_loader, supervised_loss_fn, epoch, writer, logger, rank=0):
    """验证模型（V-θ 路线）"""
    model.eval()

    losses = []
    v_losses = []
    theta_losses = []
    recon_losses = []
    capacity_losses = []
    mae_V_list = []
    mae_theta_deg_list = []
    mae_P_list = []
    mae_Q_list = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation', disable=(rank != 0)):
            batch = batch.to(cfg.device, non_blocking=True)

            pred = model(batch)
            if 'V_pred' not in pred or 'sincos_pred' not in pred:
                continue
            if not hasattr(batch, 'y_bus_V') or not hasattr(batch, 'y_edge_sincos'):
                continue

            loss, loss_dict = compute_loss_vtheta(pred, batch, cfg, rank)

            losses.append(loss_dict['total'])
            v_losses.append(loss_dict['loss_v'])
            theta_losses.append(loss_dict.get('loss_theta', 0.0))
            recon_losses.append(loss_dict.get('loss_recon', 0.0))
            if 'loss_capacity' in loss_dict:
                capacity_losses.append(loss_dict['loss_capacity'])
            if 'mae_V' in loss_dict:
                mae_V_list.append(loss_dict['mae_V'])
            if 'mae_theta_deg' in loss_dict:
                mae_theta_deg_list.append(loss_dict['mae_theta_deg'])
            if 'mae_P' in loss_dict:
                mae_P_list.append(loss_dict['mae_P'])
            if 'mae_Q' in loss_dict:
                mae_Q_list.append(loss_dict['mae_Q'])

    avg_loss = np.mean(losses) if losses else 0.0
    avg_v_loss = np.mean(v_losses) if v_losses else 0.0
    avg_theta_loss = np.mean(theta_losses) if theta_losses else 0.0
    avg_recon_loss = np.mean(recon_losses) if recon_losses else 0.0
    avg_capacity_loss = np.mean(capacity_losses) if capacity_losses else 0.0

    val_select = avg_v_loss + avg_theta_loss + avg_recon_loss + avg_capacity_loss

    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Loss_V', avg_v_loss, epoch)
    writer.add_scalar('Val/Loss_theta', avg_theta_loss, epoch)
    writer.add_scalar('Val/Loss_recon', avg_recon_loss, epoch)
    writer.add_scalar('Val/Loss_capacity', avg_capacity_loss, epoch)
    writer.add_scalar('Val/SelectMetric', val_select, epoch)
    if mae_V_list:
        writer.add_scalar('Val/MAE_V', float(np.mean(mae_V_list)), epoch)
    if mae_theta_deg_list:
        writer.add_scalar('Val/MAE_theta_deg', float(np.mean(mae_theta_deg_list)), epoch)
    if mae_P_list:
        writer.add_scalar('Val/MAE_P', float(np.mean(mae_P_list)), epoch)
    if mae_Q_list:
        writer.add_scalar('Val/MAE_Q', float(np.mean(mae_Q_list)), epoch)

    logger.info(
        f"Epoch {epoch} - Val Loss: {avg_loss:.2e} | SelectMetric: {val_select:.2e} "
        f"(L_V: {avg_v_loss:.2e}, L_theta: {avg_theta_loss:.2e}, L_recon: {avg_recon_loss:.2e}, "
        f"L_cap: {avg_capacity_loss:.2e})"
    )

    return {'total': avg_loss, 'select': val_select}


def train_simple(model, train_loader, val_loader, writer, logger, rank=0, start_epoch=1, resume_path=None):
    """完整训练流程（支持DDP）"""
    # 获取底层模型（处理DDP包装）
    unwrapped_model = model.module if hasattr(model, 'module') else model

    # 仅 PQ 路线，不使用不确定性损失
    supervised_loss_fn = None
    
    # 优化器：两组参数（模型 + Kendall）
    # 优化器参数组：PQ 路线不加入 Kendall 参数
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
    ])
    
    # 学习率调度器
    if cfg.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )
    elif cfg.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay_rate
        )
    elif cfg.lr_scheduler == 'warm':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.warm_period, T_mult=1, eta_min=1e-6
        )
    elif cfg.lr_scheduler == 'cosine_restart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=15, T_mult=1, eta_min=cfg.lr * 0.05
        )
    elif cfg.lr_scheduler == 'warmup_cosine':
        def lr_lambda(epoch):
            if epoch < cfg.warmup_epochs:
                # 预热阶段：从0线性上升到1
                return epoch / cfg.warmup_epochs
            elif epoch < cfg.cosine_start_epoch:
                # 稳定阶段：保持初始学习率
                return 1.0
            else:
                # 余弦退火阶段
                progress = (epoch - cfg.cosine_start_epoch) / (cfg.epochs - cfg.cosine_start_epoch)
                return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
    
    # 训练循环
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, cfg.epochs + 1):
        # 更新DistributedSampler的epoch（重要！）
        if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        elif hasattr(train_loader, 'loader') and hasattr(train_loader.loader, 'sampler') and isinstance(train_loader.loader.sampler, DistributedSampler):
            train_loader.loader.sampler.set_epoch(epoch)
        
        if rank == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{cfg.epochs}")
        
        # 训练
        train_loss = train_epoch_simple(model, train_loader, optimizer, supervised_loss_fn, epoch, writer, logger, rank)
        
        # 验证
        if epoch % cfg.val_interval == 0:
            val_stats = validate_simple(model, val_loader, supervised_loss_fn, epoch, writer, logger, rank)
            val_total = val_stats['total']
            val_select = val_stats['select']
            
            # 直接使用SelectMetric作为评估指标（已经是合理的组合）
            current_metric = val_select
                
            # 保存最佳模型
            if current_metric < best_val_loss:
                best_val_loss = current_metric
                patience_counter = 0
                
                # 处理DataParallel包装
                model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_total,
                    'val_select': val_select,
                    'config': cfg
                }
                
                # 只在rank0保存模型
                if rank == 0:
                    if supervised_loss_fn is not None:
                        checkpoint['uncertainty_state_dict'] = supervised_loss_fn.state_dict()
                    best_path = os.path.join(cfg.checkpoint_dir, 'best_model.pt')
                    torch.save(checkpoint, best_path)
                    logger.info(f"✓ 保存最佳模型 (SelectMetric: {current_metric:.2e})")
            else:
                patience_counter += 1
                
                if patience_counter >= cfg.patience:
                    logger.info(f"早停: 验证损失未改善 {cfg.patience} epochs")
                    break
        
        # 定期保存（只在rank0）
        if epoch % cfg.save_interval == 0 and rank == 0:
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'
            )
            model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            save_obj = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg
            }
            if supervised_loss_fn is not None:
                save_obj['uncertainty_state_dict'] = supervised_loss_fn.state_dict()
            torch.save(save_obj, checkpoint_path)
        
        # 无不确定性权重冻结逻辑（已移除）
        
        # 学习率调整
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Train/LR', new_lr, epoch)
            # 记录学习率变化
            if new_lr > old_lr * 2:  # 学习率突然增大，说明restart了
                if rank == 0:
                    logger.info(f"🔥 学习率重启! {old_lr:.2e} → {new_lr:.2e}")
            elif abs(new_lr - old_lr) > 1e-8:
                if rank == 0:
                    logger.info(f"学习率调整: {old_lr:.2e} → {new_lr:.2e}")
    
    logger.info(f"\n训练完成！最佳验证损失: {best_val_loss:.2e}")
    return best_val_loss


def evaluate_final(model, test_loader, writer, logger):
    """最终测试集评估（V-θ 路线）"""
    logger.info("\n" + "=" * 50)
    logger.info("开始测试集评估 (V-θ)")
    logger.info("=" * 50)

    mae_V_list = []
    mae_theta_deg_list = []
    mae_P_list = []
    mae_Q_list = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(cfg.device, non_blocking=True)
            pred = model(batch)
            if 'V_pred' not in pred or 'sincos_pred' not in pred:
                continue
            if not hasattr(batch, 'y_bus_V') or not hasattr(batch, 'y_edge_sincos'):
                continue

            V_pred = pred['V_pred']
            sincos_pred = pred['sincos_pred']
            edge_pq_pred = pred.get('edge_pq')

            y_V = batch.y_bus_V.to(cfg.device)
            y_sincos = batch.y_edge_sincos.to(cfg.device)
            y_edge_pq = batch.y_edge_pq.to(cfg.device) if hasattr(batch, 'y_edge_pq') else None

            mae_V_list.append(torch.abs(V_pred - y_V).mean().item())

            cos_err = (sincos_pred * y_sincos).sum(dim=1).clamp(-1, 1)
            mae_theta_deg_list.append(torch.rad2deg(torch.acos(cos_err).mean()).item())

            if edge_pq_pred is not None and y_edge_pq is not None:
                mae_P_list.append(torch.abs(edge_pq_pred[:, :2] - y_edge_pq[:, :2]).mean().item())
                mae_Q_list.append(torch.abs(edge_pq_pred[:, 2:] - y_edge_pq[:, 2:]).mean().item())

    def _avg(values):
        return float(sum(values) / len(values)) if values else float('nan')

    metrics = {
        'mae_V': _avg(mae_V_list),
        'mae_theta_deg': _avg(mae_theta_deg_list),
        'mae_P': _avg(mae_P_list),
        'mae_Q': _avg(mae_Q_list),
    }

    logger.info(
        f"  电压MAE: {metrics['mae_V']:.6f} p.u., "
        f"相角MAE: {metrics['mae_theta_deg']:.4f}°, "
        f"P重构MAE: {metrics['mae_P']:.6f} p.u., "
        f"Q重构MAE: {metrics['mae_Q']:.6f} p.u."
    )

    writer.add_scalar('Test/MAE_V', metrics['mae_V'])
    writer.add_scalar('Test/MAE_theta_deg', metrics['mae_theta_deg'])
    writer.add_scalar('Test/MAE_P', metrics['mae_P'])
    writer.add_scalar('Test/MAE_Q', metrics['mae_Q'])


def main(resume_path=None):
    """主函数（支持DDP）"""
    # ===== DDP 初始化 =====
    is_ddp, rank, local_rank, world_size, device = setup_ddp()
    cfg.device = device
    
    # 只在rank0处理日志和配置
    if rank == 0:
        # 确保目录存在
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.tensorboard_dir, exist_ok=True)
        
    # 打印配置与日志（仅rank0）
    if rank == 0:
        cfg.print_config()
        # 保存配置
        cfg.save()
        # 设置日志
        logger = setup_logging()
        logger.info(f"启动BC-GNN训练流水线 (DDP: {is_ddp}, World Size: {world_size})")
        # TensorBoard
        writer = SummaryWriter(cfg.tensorboard_dir)
    else:
        # 其他rank使用虚拟logger和writer
        class _Dummy:
            def add_scalar(self, *args, **kwargs): pass
            def add_scalars(self, *args, **kwargs): pass
            def info(self, *args, **kwargs): pass
            def warning(self, *args, **kwargs): pass
            def close(self): pass
        logger = _Dummy()
        writer = _Dummy()
    
    # 设置随机种子（每个rank不同的种子）
    set_seed(cfg.seed + rank)
    
    # 性能优化设置
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # 加载数据（需要 V1.1 字段）
    logger.info("\n加载数据...")
    # 数据应该已经通过 prepare_dataset.py 划分好
    train_data, val_data, test_data, voltage_stats = load_data(cfg.data_dir)
    
    # 数据泄漏检查与强制去重（全局一致，先检查后修复；以 test > val > train 的优先级保留样本）
    def _make_id_list(dataset):
        """生成与数据列表一一对应的稳定 ID 列表。"""
        import hashlib
        def _hash_parts(parts: list[bytes]) -> str:
            h = hashlib.sha1()
            for p in parts:
                h.update(p)
            return h.hexdigest()
        ids = []
        for data in dataset:
            if hasattr(data, 'sample_id') and data.sample_id is not None:
                ids.append(str(data.sample_id))
                continue
            if hasattr(data, 'idx') and data.idx is not None:
                ids.append(str(data.idx))
                continue
            k_val = int(getattr(data, 'k', 0))
            n_nodes = int(data.x.size(0)) if hasattr(data, 'x') else -1
            n_edges = int(data.edge_index.size(1)) if hasattr(data, 'edge_index') else -1
            parts = [f"k={k_val},n={n_nodes},e={n_edges}".encode('utf-8')]
            if hasattr(data, 'tie_buses') and data.tie_buses is not None and data.tie_buses.numel() > 0:
                parts.append(data.tie_buses.detach().cpu().numpy().tobytes())
            if hasattr(data, 'tie_corridors') and data.tie_corridors is not None and data.tie_corridors.numel() > 0:
                parts.append(data.tie_corridors.detach().cpu().numpy().tobytes())
            # 加入负荷签名（场景级）：x 的前两列 [P_load, Q_load]
            if hasattr(data, 'x') and data.x is not None and data.x.numel() > 0:
                cols = min(2, data.x.size(1))
                parts.append(data.x[:, :cols].detach().cpu().contiguous().numpy().tobytes())
            ids.append(_hash_parts(parts))
        return ids

    # 仅 rank0 打印与修复，其他 rank 同步后使用修复后的列表
    if rank == 0:
        def get_sample_ids(dataset):
            """从数据集中提取稳健的样本ID（避免伪重复误报）。

            优先使用 Data.sample_id；否则基于关键字段内容计算 SHA1：
            - k 值、节点数、边数
            - tie_buses（LongTensor）
            - tie_corridors（LongTensor [C,2]）

            这样能避免使用“首个负荷/节点数”等弱特征导致的碰撞。
            """
            import hashlib
            import numpy as np

            def _hash_parts(parts: list[bytes]) -> str:
                h = hashlib.sha1()
                for p in parts:
                    h.update(p)
                return h.hexdigest()

            sample_ids = set()
            for data in dataset:
                # 1) 明确的 sample_id/idx
                if hasattr(data, 'sample_id') and data.sample_id is not None:
                    sample_ids.add(str(data.sample_id))
                    continue
                if hasattr(data, 'idx') and data.idx is not None:
                    sample_ids.add(str(data.idx))
                    continue

                # 2) 稳健哈希（基于元数据+关键张量）
                k_val = int(getattr(data, 'k', 0))
                n_nodes = int(data.x.size(0)) if hasattr(data, 'x') else -1
                n_edges = int(data.edge_index.size(1)) if hasattr(data, 'edge_index') else -1
                parts = [
                    f"k={k_val},n={n_nodes},e={n_edges}".encode('utf-8')
                ]
                if hasattr(data, 'tie_buses') and data.tie_buses is not None and data.tie_buses.numel() > 0:
                    parts.append(data.tie_buses.detach().cpu().numpy().tobytes())
                if hasattr(data, 'tie_corridors') and data.tie_corridors is not None and data.tie_corridors.numel() > 0:
                    parts.append(data.tie_corridors.detach().cpu().numpy().tobytes())
                sample_ids.add(_hash_parts(parts))
            return sample_ids
        
        logger.info("检查数据集划分...")
        train_ids_list = _make_id_list(train_data)
        val_ids_list = _make_id_list(val_data)
        test_ids_list = _make_id_list(test_data)
        train_ids = set(train_ids_list)
        val_ids = set(val_ids_list)
        test_ids = set(test_ids_list)
        
        # 检查重叠
        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids
        
        # 输出详细信息
        logger.info(f"  训练集唯一样本数: {len(train_ids)}")
        logger.info(f"  验证集唯一样本数: {len(val_ids)}")
        logger.info(f"  测试集唯一样本数: {len(test_ids)}")
        
        if train_val_overlap:
            logger.warning(f"⚠️ 数据泄漏！train/val重叠{len(train_val_overlap)}个样本")
        else:
            logger.info("✓ train/val无重叠")
        
        if train_test_overlap:
            logger.warning(f"⚠️ 数据泄漏！train/test重叠{len(train_test_overlap)}个样本")
        else:
            logger.info("✓ train/test无重叠")
            
        if val_test_overlap:
            logger.warning(f"⚠️ 数据泄漏！val/test重叠{len(val_test_overlap)}个样本")
        else:
            logger.info("✓ val/test无重叠")

        # 强制打散重叠：优先保留 test，其次 val，最后 train
        keep_test = set(test_ids_list)
        keep_val = set(x for x in val_ids_list if x not in keep_test)
        keep_train = set(x for x in train_ids_list if x not in keep_test and x not in keep_val)

        removed_train = len(train_ids_list) - sum(1 for x in train_ids_list if x in keep_train)
        removed_val = len(val_ids_list) - sum(1 for x in val_ids_list if x in keep_val)
        removed_test = len(test_ids_list) - sum(1 for x in test_ids_list if x in keep_test)

        if removed_train or removed_val or removed_test:
            logger.info("执行去重/互斥修复: 优先级 test > val > train")
            if removed_train:
                logger.info(f"  - 从 train 移除 {removed_train} 个重叠样本")
            if removed_val:
                logger.info(f"  - 从 val 移除 {removed_val} 个重叠样本")
            if removed_test:
                logger.info(f"  - 从 test 移除 {removed_test} 个重叠样本")

            # 按保留集合过滤数据列表
            train_data = [d for d, sid in zip(train_data, train_ids_list) if sid in keep_train]
            val_data = [d for d, sid in zip(val_data, val_ids_list) if sid in keep_val]
            test_data = [d for d, sid in zip(test_data, test_ids_list) if sid in keep_test]
            logger.info(f"修复后: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    # 无论是否为 rank0，确保三份数据互斥（与上面相同优先级规则），保证一致性
    train_ids_list_all = _make_id_list(train_data)
    val_ids_list_all = _make_id_list(val_data)
    test_ids_list_all = _make_id_list(test_data)
    keep_test_all = set(test_ids_list_all)
    keep_val_all = set(x for x in val_ids_list_all if x not in keep_test_all)
    keep_train_all = set(x for x in train_ids_list_all if x not in keep_test_all and x not in keep_val_all)
    train_data = [d for d, sid in zip(train_data, train_ids_list_all) if sid in keep_train_all]
    val_data = [d for d, sid in zip(val_data, val_ids_list_all) if sid in keep_val_all]
    test_data = [d for d, sid in zip(test_data, test_ids_list_all) if sid in keep_test_all]

    # 创建DistributedSampler（DDP模式）
    train_sampler = DistributedSampler(train_data, shuffle=True, drop_last=True) if is_ddp else None
    val_sampler = DistributedSampler(val_data, shuffle=False, drop_last=False) if is_ddp else None
    test_sampler = DistributedSampler(test_data, shuffle=False, drop_last=False) if is_ddp else None
    
    # 创建数据加载器（支持大batch）
    actual_batch_size = getattr(cfg, 'batch_size', 1)
    train_loader = create_dataloader(
        train_data, batch_size=actual_batch_size, 
        shuffle=(not is_ddp),  # DDP时由sampler控制
        sampler=train_sampler,
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory,
        persistent_workers=True if cfg.persistent_workers and cfg.num_workers > 0 else False,
        prefetch_factor=getattr(cfg, 'prefetch_factor', None)
    )
    val_loader = create_dataloader(
        val_data, batch_size=actual_batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory,
        persistent_workers=True if cfg.persistent_workers and cfg.num_workers > 0 else False,
        prefetch_factor=getattr(cfg, 'prefetch_factor', None)
    )
    test_loader = create_dataloader(
        test_data, batch_size=actual_batch_size, 
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.num_workers, 
        pin_memory=cfg.pin_memory,
        persistent_workers=True if cfg.persistent_workers and cfg.num_workers > 0 else False,
        prefetch_factor=getattr(cfg, 'prefetch_factor', None)
    )
    
    # 创建模型
    logger.info("\n初始化模型...")
    # 推断特征维度（与实际数据对齐）
    sample0 = train_data[0]
    node_in_dim = int(sample0.x.size(1))
    edge_in_dim = int(sample0.edge_attr.size(1))

    # 检查 V-θ 所需字段
    required = [
        hasattr(sample0, 'tie_buses') and sample0.tie_buses is not None,
        hasattr(sample0, 'tie_corridors') and sample0.tie_corridors is not None,
        hasattr(sample0, 'tie_edge_corridor') and sample0.tie_edge_corridor is not None,
        hasattr(sample0, 'y_bus_V') and sample0.y_bus_V is not None,
        hasattr(sample0, 'y_edge_sincos') and sample0.y_edge_sincos is not None,
    ]
    if not all(required):
        raise RuntimeError('V-θ 路线所需字段缺失：tie_buses/tie_corridors/tie_edge_corridor/y_bus_V/y_edge_sincos')
    model = BCGNN(
        node_features=node_in_dim,
        edge_features=edge_in_dim,
        hidden_dim=cfg.hidden_dim,
        voltage_stats=voltage_stats,
        use_voltage_prior=cfg.use_voltage_prior,
        ring_k=getattr(cfg, 'ring_k', 0),
        ring_decay=(getattr(cfg, 'ring_decay', None) if getattr(cfg, 'ring_decay', None) is not None else None),
        ring_use_decayed=getattr(cfg, 'ring_use_decayed', True),
    ).to(cfg.device)

    # torch.compile 加速（在DDP前）
    if getattr(cfg, 'use_compile', False) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode=getattr(cfg, 'compile_mode', 'reduce-overhead'), dynamic=True, fullgraph=False)
            if rank == 0:
                logger.info("✓ 启用 torch.compile 加速")
        except Exception as e:
            if rank == 0:
                logger.warning(f"torch.compile 启用失败，继续无编译路径: {e}")
    
    # DDP包装
    if is_ddp:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        if rank == 0:
            logger.info(f"✓ 启用DDP，World Size: {world_size}")
    
    # 记录模型结构
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {param_count:,}")
    
    # Resume逻辑（DDP同步）
    start_epoch = 1
    if resume_path:
        if rank == 0:
            logger.info(f"\n从checkpoint恢复: {resume_path}")
        
        checkpoint = torch.load(resume_path, map_location=cfg.device, weights_only=False)
        
        # 恢复模型状态
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 1) + 1
        if rank == 0:
            logger.info(f"从Epoch {start_epoch}开始训练")
    
    # DDP同步
    if is_ddp:
        dist.barrier()
    
    # 训练
    logger.info("\n开始训练...")
    start_time = time.time()
    
    best_val_loss = train_simple(model, train_loader, val_loader, writer, logger, rank, start_epoch, resume_path)
    
    train_time = time.time() - start_time
    logger.info(f"训练耗时: {train_time/60:.1f} 分钟")
    
    # DDP同步
    if is_ddp:
        dist.barrier()
    
    # 加载最佳模型进行评估（只在rank0）
    if cfg.evaluate_after_training and rank == 0:
        logger.info("\n加载最佳模型进行评估...")
        best_checkpoint = torch.load(
            os.path.join(cfg.checkpoint_dir, 'best_model.pt'),
            map_location=device,
            weights_only=False
        )
        # 处理DDP包装
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_checkpoint['model_state_dict'])
        else:
            model.load_state_dict(best_checkpoint['model_state_dict'])
        model.eval()
        
        # 评估
        evaluate_final(model, test_loader, writer, logger)
    
    # 关闭
    writer.close()
    logger.info("\n流水线完成！")
    logger.info(f"日志保存在: {cfg.log_dir}")
    logger.info(f"模型保存在: {cfg.checkpoint_dir}")
    logger.info(f"TensorBoard: tensorboard --logdir {cfg.tensorboard_dir}")
    
    # 销毁DDP进程组
    if is_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint path')
    args = parser.parse_args()
    main(args.resume)
