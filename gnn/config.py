"""
BC-GNN 配置（V-θ 路线专用）
"""

import os
from datetime import datetime


class Config:
    """集中管理所有超参数（简化版）"""
    
    # ========== 数据配置 ==========
    case = 'ieee500_mixed'  # 数据集: ieee57/ieee118/ieee300
    # 指向你的已生成数据集根目录（包含 train/val/test 子目录）
    data_dir = '/Users/jiashen/Desktop/code/data/ieee500_k456'
    
    # ========== 模型配置 ==========
    hidden_dim = 64    # 隐藏层维度
    n_mpnn_layers = 4  # MPNN层数（GAT层数）
    jk_mode = 'cat'    # JK融合模式：'cat', 'max', 'last'
    n_bc_iterations = 2  # BC细化迭代次数
    max_partitions = 20  # 最大分区数
    dropout = 0.2      # Dropout率（增强正则化）
    use_voltage_prior = True  # 是否使用电压先验
    voltage_prior_weight = 0.0  # 电压先验正则权重 (0.0=仅作输入特征,不参与损失)
    use_ring_bias = True  # 是否使用边界注意力（Ring-Bias GATv2）
    ring_K = 3  # hop距离截断（用于边界注意力）

    # ========== V-θ 路线配置 ==========
    lambda_theta = 1.0              # 相角损失权重（相对于电压损失）
    lambda_recon = 0.1              # 功率重构一致性损失权重
    use_capacity_constraint = False  # 是否对重构功率施加容量约束（可选）
    lambda_capacity = 0.1           # 容量约束损失权重
    capacity_alpha = 0.95           # 容量守护的安全退让 α
    ring_k = 3                # 环形汇总K跳（若在模型中使用）
    ring_decay = 0.5          # 环形汇总的衰减因子（None 代表关闭，使用分环拼接）
    ring_use_decayed = True   # 使用 decayed 汇总(2维) 而非分环展开(2*(K+1)维)
    
    # ========== 拉格朗日损失函数配置 ==========
    # 拉格朗日乘子自动学习，无需手动设置权重
    lambda_region = 2e-2      # 区域守恒权重（保留用于兼容旧模型，不再使用）
    
    # ========== 训练配置 ==========
    batch_size = 8  # 目标批大小（每卡），batch collate已修复支持>1
    epochs = 2  # 测试训练：2轮快速验证
    lr = 0.0001  # 提高学习率到1e-4，更合理的起始值
    weight_decay = 5e-4   # 降低权重衰减，避免过度正则化
    patience = 10  # 早停耐心值（更严格）
    grad_clip = 5.0  # 提高梯度裁剪阈值，允许更大的梯度更新
    
    # 混合精度和梯度累积
    use_amp = True  # 启用自动混合精度训练
    amp_dtype = 'bf16'  # 混合精度类型：'fp16' 或 'bf16'（4090推荐bf16）
    accum_steps = 1  # 梯度累积步数（减少到1，更保守）

    # DataLoader配置
    num_workers = 4  # DataLoader进程数
    prefetch_factor = 2  # 预取因子
    persistent_workers = True  # 保持worker进程不销毁（加速epoch切换）
    
    # Kendall不确定性损失配置
    ema_momentum = 0.99        # EMA动量系数
    uncertainty_clamp = 5.0    # log_vars范围限制[-5,5]
    
    # 学习率调度（20轮余弦衰减）
    lr_scheduler = 'cosine'  # 直接余弦衰减，无预热
    lr_decay_epochs = []     # 不使用步进衰减
    lr_decay_rate = 0.1
    warmup_epochs = 0        # 无预热
    cosine_start_epoch = 1   # 从第1轮开始余弦衰减
    
    # ========== 日志配置 ==========
    exp_name = f"bcgnn_simple_{case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f'logs/{exp_name}'
    checkpoint_dir = f'checkpoints/{exp_name}'
    tensorboard_dir = f'runs/{exp_name}'
    
    # 日志频率
    log_interval = 10  # 每N个batch记录一次
    val_interval = 1   # 每N个epoch验证一次
    save_interval = 5   # 每N个epoch保存一次
    
    # ========== 评估配置 ==========
    evaluate_after_training = True  # 训练后自动评估
    test_transfer = True  # 测试跨拓扑迁移
    test_partition_split = True  # 按分区数分组评估
    
    # ========== 随机种子 ==========
    seed = 42  # 随机种子，保证可复现性

    # ========== 设备配置 ==========
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    gpu_ids = [0]  # 使用单GPU
    use_multi_gpu = False  # 暂时禁用多GPU
    pin_memory = True  # 加速GPU数据传输

    # 编译加速
    use_compile = False
    compile_mode = 'reduce-overhead'
    
    # ========== 随机种子 ==========
    seed = 42
    
    def __init__(self):
        """创建必要的目录"""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
    
    def save(self, path=None):
        """保存配置到文件"""
        import json
        if path is None:
            path = os.path.join(self.log_dir, 'config.json')
        
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_')}
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
    
    def print_config(self):
        """打印配置信息"""
        print("\n" + "="*50)
        print("配置信息（简化版）:")
        print("="*50)
        print(f"数据集: {self.case}")
        print(f"模型: hidden_dim={self.hidden_dim}, n_mpnn={self.n_mpnn_layers}, n_bc={self.n_bc_iterations}")
        print(f"训练: epochs={self.epochs}, lr={self.lr}, accum_steps={self.accum_steps}")
        print(f"优化: grad_clip={self.grad_clip}, weight_decay={self.weight_decay}")
        print(f"混合精度: {self.amp_dtype if self.use_amp else 'disabled'}")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.checkpoint_dir}")
        print("="*50 + "\n")


# 创建全局配置实例
cfg = Config()
