# ACOPF Helpers (PQ-only)

This folder now contains only PQ-only helpers consistent with the current data/model stack.

Status
- Deprecated helpers: kept for reference. Prefer `tools/two_stage_acopf.py` for a minimal, supported flow.
- New training/data live under `gnn/` and `gnn_data/` (V1.1 schema).
- ADMM consensus removed. Only independent per-area OPF is supported here.

What’s here (PQ-only)
- `src/dist_acopf_pipeline.py`: End-to-end flow using `gnn_data` OPFDataProcessor APIs and `gnn.model.BCGNN`. It now predicts bus-level boundary P/Q (p.u.), cuts inter-area links, and injects P/Q directly at boundary buses for each area. ADMM code removed; independent per-area OPF only.
- `src/bcgnn_predictor.py`: Thin wrapper to load `BCGNN` and produce `corridor_pfqt` ([pf_u,pt_v,qf_u,qt_v]), with canonical mapping to bus-level boundary P/Q.
- `src/pyg/` (removed): Legacy PQ utilities were deleted; use `gnn_data` APIs instead.

Removed legacy pieces
- Older label builders and prototypes not aligned with PQ-only have been deleted.
- Angle-synchronization-related utilities used by retired flows have been removed.

Recommended usage
- Data generation: `gnn_data/` (V1.1) → produces `x`, `edge_attr`, `tie_*`, `y_bus_pq`, `y_corridor_pfqt`, and priors.
- Training/inference: `gnn/` (PQ-only) → bus/corridor P/Q heads with capacity-aware projection utilities.
- Minimal runner: `tools/two_stage_acopf.py` → partition + predict + per-area OPF, no ADMM by default.

Notes
- This directory now relies on `gnn_data/src/opfdata/processor.py` directly to build PyG data; legacy local copies under `src/pyg/` have been removed.
