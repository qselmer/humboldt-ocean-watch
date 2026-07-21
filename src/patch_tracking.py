"""Lineage graph construction, node classification, tracks, and families.

The accepted temporal links form a directed acyclic graph (DAG).  Track IDs
identify maximal non-branching paths, while event-family IDs identify weakly
connected DAG components.  Neither identifier reuses the date-local patch ID.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.patch_linking import node_identifier


OBSERVATION_COLUMNS = [
    "node_id", "date", "local_patch_id", "track_id", "event_family_id",
    "node_event_type", "source_variable", "threshold_type", "direction",
    "area_km2", "centroid_latitude", "centroid_longitude",
    "mean_source_value", "maximum_source_value", "mean_exceedance",
    "maximum_exceedance", "parent_node_ids", "child_node_ids",
    "parent_track_ids", "child_track_ids", "predecessor_count",
    "successor_count", "continuation_backbone_parent",
    "continuation_backbone_child", "link_crossed_temporal_gap",
    "valid_coverage", "climatology_method", "data_mode", "status", "reason",
    "touches_north_boundary", "touches_south_boundary",
    "touches_east_boundary", "touches_west_boundary",
]

EDGE_COLUMNS = [
    "predecessor_node_id", "successor_node_id", "predecessor_date",
    "successor_date", "predecessor_patch_id", "successor_patch_id",
    "elapsed_days", "intersection_cell_count", "intersection_area_km2",
    "union_area_km2", "area_weighted_iou",
    "predecessor_overlap_fraction", "successor_overlap_fraction",
    "centroid_distance_km", "area_ratio", "link_score", "link_basis",
    "is_continuation_backbone", "lineage_relation",
    "crosses_temporal_gap", "status", "reason",
]

NODE_EVENT_CODES = {
    "appearance": 1,
    "continuation": 2,
    "split_parent": 3,
    "split_child": 4,
    "merge_parent": 5,
    "merge_child": 6,
    "complex_branch": 7,
    "termination": 8,
    "isolated_single_day": 9,
}


def _sorted_nodes(nodes: Iterable[str], node_rows: dict[str, pd.Series]) -> list[str]:
    return sorted(
        nodes,
        key=lambda node: (
            pd.Timestamp(node_rows[node]["date"]),
            float(node_rows[node]["centroid_latitude"]),
            float(node_rows[node]["centroid_longitude"]),
            int(node_rows[node]["local_patch_id"]),
        ),
    )


def validate_lineage_dag(node_ids: Iterable[str], edges: pd.DataFrame) -> None:
    """Raise a clear error unless edges form a forward-directed acyclic graph."""
    nodes = set(node_ids)
    incoming: dict[str, int] = {node: 0 for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    seen_edges: set[tuple[str, str]] = set()
    for row in edges.itertuples(index=False):
        predecessor = str(row.predecessor_node_id)
        successor = str(row.successor_node_id)
        if predecessor not in nodes or successor not in nodes:
            raise ValueError("A lineage edge references an unknown patch observation")
        edge = (predecessor, successor)
        if edge in seen_edges:
            raise ValueError("Duplicate lineage edges are not allowed")
        seen_edges.add(edge)
        if pd.Timestamp(row.successor_date) <= pd.Timestamp(row.predecessor_date):
            raise ValueError("Lineage edges must point strictly forward in time")
        children[predecessor].append(successor)
        incoming[successor] += 1
    queue = deque(sorted(node for node, degree in incoming.items() if degree == 0))
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    if visited != len(nodes):
        raise ValueError("Accepted lineage links contain a directed cycle")


def _node_type(
    node: str,
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> str:
    predecessors = parents[node]
    successors = children[node]
    predecessor_count = len(predecessors)
    successor_count = len(successors)
    split_parent = successor_count >= 2
    split_child = any(len(children[parent]) >= 2 for parent in predecessors)
    merge_child = predecessor_count >= 2
    merge_parent = any(len(parents[child]) >= 2 for child in successors)
    if (split_parent or split_child) and (merge_parent or merge_child):
        return "complex_branch"
    if split_parent:
        return "split_parent"
    if merge_child:
        return "merge_child"
    if split_child:
        return "split_child"
    if merge_parent:
        return "merge_parent"
    if predecessor_count == 0 and successor_count == 0:
        return "isolated_single_day"
    if predecessor_count == 0:
        return "appearance"
    if successor_count == 0:
        return "termination"
    if (
        predecessor_count == 1
        and successor_count == 1
        and len(children[next(iter(predecessors))]) == 1
        and len(parents[next(iter(successors))]) == 1
    ):
        return "continuation"
    return "complex_branch"


def _lineage_relation(
    predecessor: str,
    successor: str,
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> str:
    splits = len(children[predecessor]) >= 2
    merges = len(parents[successor]) >= 2
    if splits and merges:
        return "complex"
    if splits:
        return "split"
    if merges:
        return "merge"
    return "continuation"


class _UnionFind:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def _assign_families(
    nodes: list[str],
    node_rows: dict[str, pd.Series],
    edge_pairs: Iterable[tuple[str, str]],
) -> dict[str, str]:
    union_find = _UnionFind(nodes)
    for predecessor, successor in edge_pairs:
        union_find.union(predecessor, successor)
    components: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        components[union_find.find(node)].append(node)
    ordered = sorted(
        components.values(),
        key=lambda members: (
            pd.Timestamp(node_rows[_sorted_nodes(members, node_rows)[0]]["date"]),
            float(node_rows[_sorted_nodes(members, node_rows)[0]]["centroid_latitude"]),
            float(node_rows[_sorted_nodes(members, node_rows)[0]]["centroid_longitude"]),
            int(node_rows[_sorted_nodes(members, node_rows)[0]]["local_patch_id"]),
        ),
    )
    result: dict[str, str] = {}
    for number, members in enumerate(ordered, start=1):
        family_id = f"FAM{number:06d}"
        for node in members:
            result[node] = family_id
    return result


def _assign_tracks(
    nodes: list[str],
    node_rows: dict[str, pd.Series],
    edges: pd.DataFrame,
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> dict[str, str]:
    continuation_parent: dict[str, str] = {}
    continuation_child: dict[str, str] = {}
    for row in edges.itertuples(index=False):
        predecessor = str(row.predecessor_node_id)
        successor = str(row.successor_node_id)
        if (
            bool(row.is_continuation_backbone)
            and len(children[predecessor]) == 1
            and len(parents[successor]) == 1
        ):
            continuation_child[predecessor] = successor
            continuation_parent[successor] = predecessor
    starts = _sorted_nodes(
        (node for node in nodes if node not in continuation_parent), node_rows
    )
    result: dict[str, str] = {}
    for number, start in enumerate(starts, start=1):
        track_id = f"TRK{number:06d}"
        node = start
        while True:
            if node in result:
                raise ValueError("A patch observation was assigned to more than one track")
            result[node] = track_id
            if node not in continuation_child:
                break
            node = continuation_child[node]
    if set(result) != set(nodes):
        raise ValueError("Every patch observation must be assigned to exactly one track")
    return result


def build_lineage(
    patches: pd.DataFrame,
    pair_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign node types, deterministic track IDs, and event-family IDs."""
    required = {
        "date", "patch_id", "source_variable", "threshold_type", "direction",
        "area_km2", "centroid_latitude", "centroid_longitude",
        "mean_source_value", "maximum_source_value", "mean_exceedance",
        "maximum_exceedance", "valid_coverage", "climatology_method", "data_mode",
    }
    missing = sorted(required.difference(patches.columns))
    if missing:
        raise ValueError("Daily patch table is missing: " + ", ".join(missing))
    canonical = patches.copy()
    canonical["date"] = pd.to_datetime(canonical["date"], errors="raise").dt.normalize()
    canonical["local_patch_id"] = canonical["patch_id"].astype(int)
    canonical["node_id"] = [
        node_identifier(date, patch_id)
        for date, patch_id in zip(canonical.date, canonical.local_patch_id, strict=True)
    ]
    if canonical.node_id.duplicated().any():
        raise ValueError("Patch node identifiers must be unique")
    canonical = canonical.sort_values(
        ["date", "centroid_latitude", "centroid_longitude", "local_patch_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    nodes = canonical.node_id.astype(str).tolist()
    node_rows = {
        str(row.node_id): pd.Series(row._asdict())
        for row in canonical.itertuples(index=False)
    }
    parents: dict[str, set[str]] = {node: set() for node in nodes}
    children: dict[str, set[str]] = {node: set() for node in nodes}
    edge_rows: list[dict[str, Any]] = []
    for row in pair_edges.itertuples(index=False):
        predecessor = node_identifier(row.predecessor_date, row.predecessor_patch_id)
        successor = node_identifier(row.successor_date, row.successor_patch_id)
        if predecessor not in parents or successor not in parents:
            raise ValueError("Accepted link does not correspond to a patch-table node")
        parents[successor].add(predecessor)
        children[predecessor].add(successor)
        edge_rows.append({
            "predecessor_node_id": predecessor,
            "successor_node_id": successor,
            "predecessor_date": pd.Timestamp(row.predecessor_date).normalize(),
            "successor_date": pd.Timestamp(row.successor_date).normalize(),
            "predecessor_patch_id": int(row.predecessor_patch_id),
            "successor_patch_id": int(row.successor_patch_id),
            "elapsed_days": int(row.elapsed_days),
            "intersection_cell_count": int(row.intersection_cell_count),
            "intersection_area_km2": float(row.intersection_area_km2),
            "union_area_km2": float(row.union_area_km2),
            "area_weighted_iou": float(row.area_weighted_iou),
            "predecessor_overlap_fraction": float(row.predecessor_overlap_fraction),
            "successor_overlap_fraction": float(row.successor_overlap_fraction),
            "centroid_distance_km": float(row.centroid_distance_km),
            "area_ratio": float(row.area_ratio),
            "link_score": float(row.link_score),
            "link_basis": row.link_basis,
            "is_continuation_backbone": bool(row.is_continuation_backbone),
            "lineage_relation": None,
            "crosses_temporal_gap": bool(row.crosses_temporal_gap),
            "status": "valid",
            "reason": None,
        })
    edges = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)
    validate_lineage_dag(nodes, edges)
    if not edges.empty:
        edges["lineage_relation"] = [
            _lineage_relation(row.predecessor_node_id, row.successor_node_id, parents, children)
            for row in edges.itertuples(index=False)
        ]
    pairs = [
        (str(row.predecessor_node_id), str(row.successor_node_id))
        for row in edges.itertuples(index=False)
    ]
    family_ids = _assign_families(nodes, node_rows, pairs)
    track_ids = _assign_tracks(nodes, node_rows, edges, parents, children)
    backbone_parents: dict[str, list[str]] = defaultdict(list)
    backbone_children: dict[str, list[str]] = defaultdict(list)
    crossed: dict[str, bool] = defaultdict(bool)
    for row in edges.itertuples(index=False):
        predecessor, successor = str(row.predecessor_node_id), str(row.successor_node_id)
        if bool(row.is_continuation_backbone):
            backbone_parents[successor].append(predecessor)
            backbone_children[predecessor].append(successor)
        if bool(row.crosses_temporal_gap):
            crossed[predecessor] = True
            crossed[successor] = True

    observations: list[dict[str, Any]] = []
    for row in canonical.itertuples(index=False):
        node = str(row.node_id)
        parent_nodes = _sorted_nodes(parents[node], node_rows)
        child_nodes = _sorted_nodes(children[node], node_rows)
        reasons = [] if pd.isna(getattr(row, "reason", None)) else [str(row.reason)]
        observations.append({
            "node_id": node,
            "date": pd.Timestamp(row.date).normalize(),
            "local_patch_id": int(row.local_patch_id),
            "track_id": track_ids[node],
            "event_family_id": family_ids[node],
            "node_event_type": _node_type(node, parents, children),
            "source_variable": row.source_variable,
            "threshold_type": row.threshold_type,
            "direction": row.direction,
            "area_km2": float(row.area_km2),
            "centroid_latitude": float(row.centroid_latitude),
            "centroid_longitude": float(row.centroid_longitude),
            "mean_source_value": float(row.mean_source_value),
            "maximum_source_value": float(row.maximum_source_value),
            "mean_exceedance": float(row.mean_exceedance),
            "maximum_exceedance": float(row.maximum_exceedance),
            "parent_node_ids": parent_nodes,
            "child_node_ids": child_nodes,
            "parent_track_ids": sorted({track_ids[value] for value in parent_nodes}),
            "child_track_ids": sorted({track_ids[value] for value in child_nodes}),
            "predecessor_count": len(parent_nodes),
            "successor_count": len(child_nodes),
            "continuation_backbone_parent": (
                sorted(backbone_parents[node])[0] if backbone_parents[node] else None
            ),
            "continuation_backbone_child": (
                sorted(backbone_children[node])[0] if backbone_children[node] else None
            ),
            "link_crossed_temporal_gap": bool(crossed[node]),
            "valid_coverage": float(row.valid_coverage),
            "climatology_method": row.climatology_method,
            "data_mode": row.data_mode,
            "status": "warning" if reasons else "valid",
            "reason": "; ".join(reasons) if reasons else None,
            "touches_north_boundary": bool(getattr(row, "touches_north_boundary", False)),
            "touches_south_boundary": bool(getattr(row, "touches_south_boundary", False)),
            "touches_east_boundary": bool(getattr(row, "touches_east_boundary", False)),
            "touches_west_boundary": bool(getattr(row, "touches_west_boundary", False)),
        })
    observation_frame = pd.DataFrame(observations, columns=OBSERVATION_COLUMNS)
    if observation_frame.track_id.isna().any() or observation_frame.event_family_id.isna().any():
        raise ValueError("Every patch observation requires one track and one event family")
    return observation_frame, edges[EDGE_COLUMNS]
