"""
OPF (Optimal Power Flow) 相关工具函数

增强版 DCPF 与支路 DC 流计算（统一入口）
- 统一 tap/phase-shift 处理
- 母线相角一律使用增强 DCPF（不读取 AC 解）
"""

import numpy as np
from typing import Dict, Optional, Tuple


class OPFUtils:
    """OPF 相关计算工具（增强版 DCPF + 支路 Pdc）。"""

    def get_bus_angles(self, network_data: Dict, opf_data: Optional[Dict] = None) -> np.ndarray:
        """
        获取母线相角（弧度）。一律基于增强 DCPF 计算；不读取或依赖 AC 解。
        """
        return self.calculate_dcpf(network_data)

    def calculate_dcpf(self, network_data: Dict) -> np.ndarray:
        """
        计算直流潮流（DC Power Flow, DCPF）母线相角（弧度）。

        - Bθ = P + s，s 为相移等效注入（仅来自变压器，相移 shift 见 OPFData 定义）
        - 变压器等效电抗 x_eff = |x|/|tap|
        - 参考母线 θ=0
        """
        n_buses = int(network_data['n_buses'])
        load_map = network_data.get('load_map', {})
        gen_map = network_data.get('gen_map', {})
        slack_bus = int(network_data.get('slack_bus', 0))

        B = np.zeros((n_buses, n_buses), dtype=float)

        # AC 线路对 B 的贡献
        ac = network_data.get('ac_lines', {})
        for fb, tb, feat in zip(ac.get('senders', []), ac.get('receivers', []), ac.get('features', [])):
            try:
                x = float(feat[5])
            except Exception:
                x = 0.0
            if abs(x) > 1e-12:
                bij = -1.0 / x
                B[fb, tb] += bij
                B[tb, fb] += bij
                B[fb, fb] -= bij
                B[tb, tb] -= bij

        # 变压器对 B 与 s 的贡献（tap, shift）
        tr = network_data.get('transformers', {})
        # shift 注入向量（等效注入，维度 [n_buses]）
        s_inj = np.zeros(n_buses, dtype=float)
        for fb, tb, feat in zip(tr.get('senders', []), tr.get('receivers', []), tr.get('features', [])):
            try:
                br_x = float(feat[3])
            except Exception:
                br_x = 0.0
            tap = float(feat[7]) if len(feat) > 7 and abs(feat[7]) > 1e-8 else 1.0
            shift = float(feat[8]) if len(feat) > 8 else 0.0  # 可能为度或弧度
            # 兼容：若绝对值大于 π，视为度，转换为弧度
            if abs(shift) > np.pi:
                shift = np.deg2rad(shift)
            if abs(br_x) > 1e-12:
                # 等效电纳（DC）
                y = -1.0 / br_x
                # B 矩阵（含 tap）
                B[fb, fb] += -y / (tap * tap)
                B[tb, tb] += -y
                B[fb, tb] += y / tap
                B[tb, fb] += y / tap

                # 相移的等效注入：b_eff * shift 分别注入两端，符号相反
                # 近似：b_eff = 1 / (|x|/|tap|) = |tap|/|x|，采用正值（与 DC P 的线性项一致）
                b_eff = abs(tap) / max(abs(br_x), 1e-12)
                s_inj[fb] += b_eff * shift
                s_inj[tb] -= b_eff * shift

        # P_net
        P_net = np.zeros(n_buses, dtype=float)
        for bus in range(n_buses):
            p_load = float(load_map.get(bus, (0.0, 0.0))[0])
            # gen_map现在返回6元组：(P_gen, Q_gen, P_min, P_max, Q_min, Q_max)
            gen_data = gen_map.get(bus, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            p_gen = float(gen_data[0])  # 取P_gen
            P_net[bus] = p_gen - p_load

        # B' θ' = (P + s)'
        keep = [i for i in range(n_buses) if i != slack_bus]
        B_red = B[np.ix_(keep, keep)]
        rhs = (P_net + s_inj)[keep]

        theta = np.zeros(n_buses, dtype=float)
        try:
            theta_red = np.linalg.solve(B_red, rhs)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"DCPF 求解失败：B' 奇异。详情: {e}")
        for i, idx in enumerate(keep):
            theta[idx] = theta_red[i]
        theta[slack_bus] = 0.0
        return theta

    def compute_branch_pdc(self, network_data: Dict, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        基于 θ 计算有向支路的 DC 有功：
        - AC 线：Pdc = (θ_i - θ_j)/x
        - 变压器：Pdc = (θ_i - θ_j - shift)/(x/|tap|)

        返回：
          (ac_from, ac_to, ac_pdc), (tr_from, tr_to, tr_pdc)
        """
        # AC
        ac = network_data.get('ac_lines', {})
        ac_from = np.asarray(ac.get('senders', []), dtype=int)
        ac_to = np.asarray(ac.get('receivers', []), dtype=int)
        ac_feats = np.asarray(ac.get('features', []), dtype=float)
        if ac_feats.size > 0 and ac_feats.shape[1] > 5:
            x = ac_feats[:, 5].astype(float)
            x_eff = np.where(np.abs(x) > 1e-12, x, np.sign(x) * 1e-12 + (x == 0) * 1e-12)
        else:
            x_eff = np.full_like(ac_from, 1.0, dtype=float)
        ac_pdc = (theta[ac_from] - theta[ac_to]) / x_eff

        # Transformer
        tr = network_data.get('transformers', {})
        tr_from = np.asarray(tr.get('senders', []), dtype=int)
        tr_to = np.asarray(tr.get('receivers', []), dtype=int)
        tr_feats = np.asarray(tr.get('features', []), dtype=float)
        if tr_feats.size > 0:
            br_x = tr_feats[:, 3] if tr_feats.shape[1] > 3 else np.full(len(tr_from), 0.1)
            tap = tr_feats[:, 7] if tr_feats.shape[1] > 7 else np.ones(len(tr_from))
            shift = tr_feats[:, 8] if tr_feats.shape[1] > 8 else np.zeros(len(tr_from))
            # 兼容度/弧度：向量化处理
            shift = np.where(np.abs(shift) > np.pi, np.deg2rad(shift), shift)
            x_eff = np.where(np.abs(br_x) > 1e-12, np.abs(br_x) / np.maximum(np.abs(tap), 1e-12), 1.0)
            tr_pdc = (theta[tr_from] - theta[tr_to] - shift) / x_eff
        else:
            tr_pdc = np.zeros(len(tr_from), dtype=float)

        return ac_from, ac_to, ac_pdc, tr_from, tr_to, tr_pdc
