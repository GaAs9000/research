# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a BC-GNN (Boundary Coupling Graph Neural Network) project for power system optimal power flow (OPF) problems. The project uses graph neural networks to predict boundary conditions between partitioned areas, enabling distributed AC optimal power flow solving.

**Dual Prediction Routes:**
- **PQ Route**: Directly predicts corridor-level port flows `[pf_u, pt_v, qf_u, qt_v]` in p.u., with loss-aware corridors (Pu+Pv≠0 due to line losses)
- **V-θ Route**: Predicts boundary bus voltages `V [B]` and corridor angle differences `Δθ [C]` (as sin/cos), then reconstructs power via AC power flow equations. This route is theoretically superior due to fewer dimensions (B+C vs 4C) and stronger physical consistency.

**Key Architecture Components:**
- **V1.1 Data Schema**: Supports both routes with corresponding labels (y_corridor_pfqt for PQ, y_bus_V/y_edge_sincos for V-θ)
- **AC-only tie-lines**: Transformers are not treated as inter-area ties
- **Complete π-model**: AC power flow equations include charging susceptance (b_fr, b_to) for accurate reactive power
- **Physical constraints**: Four-in-one Lagrangian loss with auto-learned weights (port supervision + bus consistency + capacity safety + regional conservation)
- **No ADMM**: Per-area independent OPF with boundary injections

## Project Structure

```
gnn/                  # BC-GNN model (PQ-only): model.py, train.py, evaluate.py, utilities
gnn_data/            # V1.1 data pipeline: JSON → PyG with y_corridor_pfqt, y_bus_pq, ring features
acopf/               # Two-stage ACOPF tooling (uses predicted boundary P/Q, AC-only tie lines)
tools/               # CLI utilities (two_stage_acopf.py for end-to-end demo, data_sanity_scan.py)
data/                # Processed datasets (train/val/test/chunk_*.pt)
```

## Essential Commands

### Data Processing
```bash
# Process OPFData JSON into V1.1 PyG format
python gnn_data/src/main.py process --datasets ieee500 --overwrite

# Validate generated labels
python gnn_data/src/validate_labels.py --processed-dir <dir> --max-samples 50

# Check data integrity
python tools/data_sanity_scan.py --processed-dir <dir> --max-samples 100
```

### Training & Evaluation
```bash
# Configure data path in gnn/config.py first
# Set data_dir = '/path/to/processed/data'

# Smoke test (quick sanity check)
python gnn/smoke_pq.py --processed-dir <dir> --device cuda

# Train model
python gnn/train.py

# Evaluate model
python gnn/evaluate.py --data_dir <dir> --model <checkpoint.pt>
```

### Two-Stage ACOPF Demo
```bash
# End-to-end: partition → predict → per-area OPF
python tools/two_stage_acopf.py --json acopf/raw_data/example_2.json --model <checkpoint.pt>
```

## Data Schema (V1.1)

**Node features** `x [N,20]`:
- Base 4 cols: `[P_load, Q_load, theta_dcpf, is_coupling_bus]` (p.u.)
- Generator 6 cols: `[P_gen, Q_gen, P_gen_min, P_gen_max, Q_gen_min, Q_gen_max]` (aggregated per bus)
- Precomputed 10 cols: `[B_decayed, S_decayed, feat_B_0..3, feat_S_0..3]` (Q prior removed)

**Edge features** `edge_attr [E,10]`:
- `[R, X_eff, S_max, is_tie, edge_type(0=AC/1=trafo), shift, Pdc, Pdc_ratio, b_fr, b_to]`
- AC lines: `edge_type=0, shift=0, X_eff=|x|, b_fr/b_to=充电电纳`
- Transformers: `edge_type=1, X_eff=|x|/|tap|`, shift from JSON, b_fr/b_to=充电电纳

**Boundary objects**:
- `tie_buses [B]`: Coupling bus IDs
- `tie_corridors [C,2]`: Inter-area corridors `(u<v)` (canonical direction)
- `tie_edge_corridor [E]`: Corridor ID for each directed edge (-1 for non-tie edges)

**Labels** (route-dependent):
- PQ route: `y_corridor_pfqt [C,4]` (port flows `[pf_u, pt_v, qf_u, qt_v]` in p.u.), `y_bus_pq [B,2]` (bus net injections)
- V-θ route: `y_bus_V [B]` (boundary bus voltages in p.u.), `y_edge_sincos [E_tie,2]` (tie-line angle differences as [sin(Δθ), cos(Δθ)]), `y_edge_pq [E_tie,4]` (for reconstruction validation)

**Prior**:
- `pq_prior [B,2]`: P prior from DCPF (column 2 Q_prior=0), used with residual + dropout

## Training Configuration (gnn/config.py)

**Route selection** (choose one):
- `use_pq_route=True, use_vtheta_route=False`: PQ route (direct power prediction)
- `use_pq_route=False, use_vtheta_route=True`: V-θ route (voltage/angle prediction + reconstruction)

**Core settings**:
- `data_dir`: Path to processed dataset directory (must contain train/val/test)
- `batch_size=1, accum_steps=1`: Conservative batch settings for stability
- `use_compile=True, compile_mode='reduce-overhead'`: torch.compile acceleration (default enabled for PyTorch ≥2.1)

**Physical constraints** (Lagrangian losses with auto-learned weights):
- `pq_capacity_alpha=0.95`: Capacity safety margin α (penalize if S > α·Smax)
- Port supervision: Huber loss on `[pf_u, pt_v, qf_u, qt_v]`
- Bus consistency: Aggregate port predictions to bus-level and compare with y_bus_pq
- Regional conservation: Internal generation + boundary net injection ≈ 0

**Prior and features**:
- `pq_use_p_prior=True, pq_use_q_prior=False`: Use P prior only (residual mode)
- `pq_prior_dropout=0.2`: Dropout on prior channel to prevent overfitting
- `ring_k=3, ring_decay=0.5, ring_use_decayed=True`: Ring-aggregated features (K=3 hops)

**Acceleration** (default enabled):
- AMP with bf16
- `pin_memory=True, persistent_workers=True, prefetch_factor=4`
- torch.compile with `reduce-overhead` mode
- Set `use_compile=False` to disable compilation if third-party operator compatibility issues occur

## Important Conventions

**Units**: All power values in p.u. (divided by baseMVA from JSON)

**Corridor direction**: Fixed as `(u<v)`; if physical line direction is `(v→u)`, aggregation uses `pt/qt` appropriately

**AC-only ties**: Transformers are NOT inter-area ties (edge_type=1 should have is_tie=0)

**Loss aggregation**: Uses per-graph mean → cross-graph mean (fair aggregation) via bus_ptr/corr_ptr attached to Batch during collate

**Capacity constraint**: Corridor-level S_max aggregated from tie_edge_corridor; penalizes `max(0, (S-α·Smax)/(α·Smax))` automatically divided by 2 to account for bidirectional edges

## Model Architecture (gnn/model.py)

**BC-GNN Components**:
- GAT-based MPNN layers (n_mpnn_layers=4)
- JumpingKnowledge fusion (mode='cat')
- BC refinement iterations (n_bc_iterations=2)
- PQ port head: outputs per-corridor `[Pu, Pv, Qu, Qv]` (4 values per corridor)
- Optional: Pairwise projection for capacity-aware output (pairwise_projection_corrend utility)

**Input handling**:
- Supports extended node features (x[N,20]) with generator info and precomputed ring features
- Edge features include DCPF-based Pdc and Pdc_ratio for better initialization
- Ring features are precomputed and appended to x (rely on gnn_data APIs)

**Output dictionary** (route-dependent):
```python
# PQ route
{
    'corridor_pfqt': torch.Tensor[C,4],  # [pf_u, pt_v, qf_u, qt_v] in p.u.
}

# V-θ route
{
    'V_pred': torch.Tensor[B],          # Boundary bus voltages (p.u.)
    'sincos_pred': torch.Tensor[C,2],   # Corridor angle diff [sin(Δθ), cos(Δθ)]
    'edge_pq': torch.Tensor[E_tie,4],   # Reconstructed power [Pf, Pt, Qf, Qt] via AC equations
}
```

**AC Power Flow Reconstruction** (V-θ route):

The model reconstructs power from predicted voltages and angles using complete π-model AC power flow equations:
```python
# Series admittance
G = R / (R² + X²)
B = -X / (R² + X²)

# Power equations (P unaffected by charging susceptance)
Pf = V_u² G - V_u·V_v·(G·cos(Δθ) + B·sin(Δθ))
Pt = V_v² G - V_v·V_u·(G·cos(Δθ) - B·sin(Δθ))

# Reactive power (includes charging susceptance b_fr, b_to)
Qf = -V_u² (B + b_fr) - V_u·V_v·(G·sin(Δθ) - B·cos(Δθ))
Qt = -V_v² (B + b_to) - V_v·V_u·(-G·sin(Δθ) - B·cos(Δθ))
```

**Critical**: The charging susceptance (b_fr, b_to) is essential for accurate reactive power. Without it, Q errors can be ~0.015 p.u. With it, errors drop to ~3e-6 p.u. (4400x improvement).

## Two-Stage ACOPF Pipeline (tools/two_stage_acopf.py)

**Stage 1**: Partition via OPFDataProcessor (constructive growth) + BC-GNN prediction of corridor port flows

**Stage 2**: Build per-area pandapower network:
- Keep internal elements (loads, generators, shunts, internal branches)
- Cut inter-area tie lines
- Inject predicted boundary P/Q at boundary buses as fixed sgen (can be negative)
- Add one ext_grid as angle reference
- Run per-area OPF independently (no ADMM by default)

**Usage notes**:
- Predictions are converted from p.u. to MW/MVAr (multiply by baseMVA)
- No voltage anchoring or ADMM consensus by default (minimal baseline)
- Refer to acopf/src/dist_acopf_pipeline.py for more detailed pipeline

## Coding Conventions

- Python 3.11+, 4-space indent, f-strings, snake_case for files/identifiers
- Type hints encouraged for public APIs
- Device-safe operations: `tensor.to(device, non_blocking=True)`
- Avoid creating modules inside `forward` (use lazy init if needed)
- Prefer vectorized scatter/segment operations over loops
- Keep functions pure and side-effect free where possible

## Testing

- Primary validation: smoke tests + label validators
- Place unit tests under `tests/` with names `test_*.py`
- Target critical utilities: parsers, mappers, loss functions
- Keep tests small, deterministic, and fast

## Commit Guidelines

- Use conventional commits (imperative mood)
- Examples: `feat: add port-end head`, `fix: AC-only tie mask`, `docs: update V1.1 schema`
- Do not commit large datasets/checkpoints
- PRs should include: summary, rationale, reproduction commands, risks

## Critical Known Issues & Fixes

### AC Power Flow Reconstruction Q Error

**Problem**: Large Q (reactive power) reconstruction errors (~0.015 p.u.) despite correct P errors (<1e-8).

**Root Cause**: Missing charging susceptance (b_fr, b_to) in AC power flow equations. The standard π-model requires:
- Q_from = -V_from² (B + b_fr) - V_from·V_to·(...)
- Q_to = -V_to² (B + b_to) - V_to·V_from·(...)

**Solution**:
1. Add b_fr, b_to to edge_attr (expand from 8D to 10D)
2. Extract from JSON: `ac_line.features[2]` (b_fr), `ac_line.features[3]` (b_to)
3. Update both reconstruction functions in model.py:
   - `power_flow_reconstruction()` (corridor aggregated version)
   - `power_flow_reconstruction_per_edge()` (per-edge version)
4. Regenerate all data with new edge_attr schema

**Verification**: With fix, Q error drops from ~0.015 to ~3e-6 (4400x improvement).

### Direction Convention: Pf/Pt Swap

**Problem**: Model reconstruction and labels have opposite definitions of "from" and "to".

**Root Cause**:
- Labels use edge direction: Pf flows along edge (u→v), Pt flows reverse (v→u)
- Reconstruction calculates: P_u (power from u), P_v (power from v)
- These are swapped: P_u = Pt_label, P_v = Pf_label

**Solution**: In reconstruction functions, swap output: `[Pt, Pf, Qt, Qf]` instead of `[Pf, Pt, Qf, Qt]`

**Verification**: Both P and Q errors should be <1e-6 after fix.

## Quick Verification Commands

```bash
# Verify data schema (should show edge_attr[E,10])
python3 -c "
from gnn.data import load_split_data
data = load_split_data('data/ieee500_k456', 'test')
print(f'edge_attr shape: {data[0].edge_attr.shape}')
print(f'Has b_fr/b_to: {data[0].edge_attr.shape[1] == 10}')
"

# Verify AC reconstruction accuracy
python3 -c "
import torch
from gnn.model import BCGNN
from gnn.data import load_split_data

data = load_split_data('data/ieee500_k456', 'test')[0]
model = BCGNN(node_features=20, edge_features=10, hidden_dim=64)
model.eval()

with torch.no_grad():
    recon = model.power_flow_reconstruction_per_edge(
        data.y_bus_V, data.y_edge_sincos, data, data.tie_edge_indices
    )

P_err = torch.abs(recon[:,:2] - data.y_edge_pq[:,:2]).mean()
Q_err = torch.abs(recon[:,2:] - data.y_edge_pq[:,2:]).mean()
print(f'P error: {P_err:.2e}')
print(f'Q error: {Q_err:.2e}')
print(f'✓ Pass' if Q_err < 1e-5 else '✗ Fail: Q error too large')
"
```