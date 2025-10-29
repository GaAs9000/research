"""
Utility helpers for BC-GNN PQ training.

Modules:
 - bpq: boundary PQ aggregation, K-hop ring features, pairwise projection, capacity guard.
"""

from .bpq import (
    build_pair2cid,
    aggregate_boundary_pq_from_solution,
    assert_labels_consistency_with_ycorr,
    # ring_sums_khop,  # 已废弃 - frontier-based环汇总
    map_corridor_to_bus_preds,
    pairwise_projection_corrend,
    pairwise_penalty_from_bus,
    corridor_Smax_from_edges,
    capacity_guard_loss,
    bpq_training_step,
)

__all__ = [
    "build_pair2cid",
    "aggregate_boundary_pq_from_solution",
    "assert_labels_consistency_with_ycorr",
    # "ring_sums_khop",  # 已废弃
    "map_corridor_to_bus_preds",
    "pairwise_projection_corrend",
    "pairwise_penalty_from_bus",
    "corridor_Smax_from_edges",
    "capacity_guard_loss",
    "bpq_training_step",
]
