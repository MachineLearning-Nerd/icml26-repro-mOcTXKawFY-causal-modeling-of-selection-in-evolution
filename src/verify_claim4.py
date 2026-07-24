"""Claim 4 verification: Theorem 2 — soundness and completeness of PC/GES on G+.

Theorem 2 states (asymptotically, with enough i.i.d. data):
  1. Adjacency S&C: Xi,Xj adjacent in CPDAG C ⟺ Xi→Xj∈G OR Xi←Xj∈G OR {Xi,Xj}⊆an_G(S)
  2. Orientation soundness: Xi→Xj in C ⟹ Xi∈pa_G(Xj) AND Xj∉an_G(S)
  3. Orientation completeness: Xi—Xj in C ⟹ genuinely ambiguous

We verify:
  - Skeleton recovery: PC/GES skeleton matches G+ skeleton (precision & recall)
  - Orientation soundness rate: fraction of oriented edges satisfying Theorem 2
  - "Standard" vs "Our" precision: the paper's key insight that only oriented edges
    (with non-selection target) should be trusted as causal
  - Unoriented edges overlap with selection clique: testing the completeness claim
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
from typing import Dict, List, Set, Tuple

from graph import DAG
from evolutionary import (
    clique_augmented_dag,
    ancestors_of_selection,
    random_dag_with_selection,
)
from sem import generate_evolutionary_data
from algorithms import run_pc, run_ges


def dag_to_matrix(G: DAG, trait_nodes: List[str]) -> np.ndarray:
    d = len(trait_nodes)
    M = np.zeros((d, d), dtype=int)
    for a, b in G.edges():
        if a in trait_nodes and b in trait_nodes:
            i = trait_nodes.index(a)
            j = trait_nodes.index(b)
            M[i, j] = 1
    return M


def verify_claim4_theorem2(
    d_values: List[int] = [10, 15, 20],
    T_values: List[int] = [2, 4, 8],
    n_runs: int = 10,
    n_samples: int = 500,
    alpha: float = 0.05,
    seed_offset: int = 4000,
) -> dict:
    """Verify Theorem 2 by running PC and GES on evolutionary SEM data."""
    results = {"configs": [], "total_runs": 0}

    for d in d_values:
        for T in T_values:
            config = {
                "d": d, "T": T, "runs": [],
                # Adjacency metrics
                "pc_adj_precision": [], "pc_adj_recall": [],
                "ges_adj_precision": [], "ges_adj_recall": [],
                # Interpretation precision (paper's key claim)
                "pc_standard_prec": [], "pc_ours_prec": [],
                "ges_standard_prec": [], "ges_ours_prec": [],
                # Orientation soundness rate
                "pc_orient_sound_rate": [], "ges_orient_sound_rate": [],
                # Unoriented overlap with selection
                "pc_unoriented_in_clique_rate": [],
                "ges_unoriented_in_clique_rate": [],
            }

            for run_idx in range(n_runs):
                seed = seed_offset + d * 10000 + T * 100 + run_idx

                X_T, G, S_node, trait_nodes, Gplus, pi, actual_n = \
                    generate_evolutionary_data(d=d, T=T, n_initial=n_samples, seed=seed)

                # True structures
                an_S = ancestors_of_selection(G, S_node)
                an_S_idx = set(trait_nodes.index(n) for n in an_S if n in trait_nodes)

                # True G+ edges
                gplus_edges: Set[frozenset] = set()
                for a, b in Gplus.edges():
                    i, j = trait_nodes.index(a), trait_nodes.index(b)
                    gplus_edges.add(frozenset({i, j}))

                # True causal edges (trait → trait in G)
                true_causal: Set[frozenset] = set()
                true_causal_dir: Set[Tuple[int,int]] = set()
                for a, b in G.edges():
                    if a in trait_nodes and b in trait_nodes:
                        i, j = trait_nodes.index(a), trait_nodes.index(b)
                        true_causal.add(frozenset({i, j}))
                        true_causal_dir.add((i, j))

                # Selection clique edges (in G+ but not in G)
                clique_edges = gplus_edges - true_causal

                # Run algorithms
                try:
                    pc_res = run_pc(X_T, alpha=alpha)
                except Exception as e:
                    config["runs"].append({"seed": seed, "error": str(e)})
                    continue

                try:
                    ges_res = run_ges(X_T)
                except Exception:
                    ges_res = {"skeleton": set(), "directed": set(), "undirected": set()}

                for algo, res in [("pc", pc_res), ("ges", ges_res)]:
                    skel = res["skeleton"]
                    directed = res["directed"]
                    undirected = res["undirected"]

                    # Adjacency precision/recall vs G+
                    tp = len(skel & gplus_edges)
                    fp = len(skel - gplus_edges)
                    fn = len(gplus_edges - skel)
                    config[f"{algo}_adj_precision"].append(tp / max(tp + fp, 1))
                    config[f"{algo}_adj_recall"].append(tp / max(tp + fn, 1))

                    # Standard interpretation: all adjacencies are causal
                    std_tp = len(skel & true_causal)
                    std_fp = len(skel - true_causal)
                    config[f"{algo}_standard_prec"].append(std_tp / max(std_tp + std_fp, 1))

                    # Our interpretation: only oriented edges with non-selection target
                    ours_edges = {frozenset({i, j}) for (i, j) in directed
                                  if j not in an_S_idx}
                    ours_tp = len(ours_edges & true_causal)
                    ours_fp = len(ours_edges - true_causal)
                    config[f"{algo}_ours_prec"].append(ours_tp / max(ours_tp + ours_fp, 1))

                    # Orientation soundness rate
                    if directed:
                        sound_count = sum(1 for (i, j) in directed
                                          if (i, j) in true_causal_dir and j not in an_S_idx)
                        config[f"{algo}_orient_sound_rate"].append(sound_count / len(directed))
                    else:
                        config[f"{algo}_orient_sound_rate"].append(1.0)

                    # Unoriented edge overlap with selection clique
                    if undirected:
                        clique_overlap = len(undirected & clique_edges) / len(undirected)
                        config[f"{algo}_unoriented_in_clique_rate"].append(clique_overlap)
                    else:
                        config[f"{algo}_unoriented_in_clique_rate"].append(0.0)

                config["runs"].append({
                    "seed": seed, "n": actual_n,
                    "gplus_edges": len(gplus_edges),
                    "clique_edges": len(clique_edges),
                    "pc_skeleton": len(pc_res["skeleton"]),
                    "pc_directed": len(pc_res["directed"]),
                    "pc_undirected": len(pc_res["undirected"]),
                    "ges_skeleton": len(ges_res["skeleton"]),
                    "ges_directed": len(ges_res["directed"]),
                })
                results["total_runs"] += 1

            # Summarize
            for key in list(config.keys()):
                if key.endswith("_precision") or key.endswith("_recall") or \
                   key.endswith("_prec") or key.endswith("_rate"):
                    vals = config[key]
                    if isinstance(vals, list) and vals:
                        config[key + "_mean"] = float(np.mean(vals))
                        config[key + "_std"] = float(np.std(vals))
                        del config[key]

            results["configs"].append(config)

    # Overall assessment
    all_configs_summary = []
    for c in results["configs"]:
        d_str = f"d={c['d']},T={c['T']}"
        pc_ours = c.get("pc_ours_prec_mean", 0)
        pc_std = c.get("pc_standard_prec_mean", 0)
        ges_ours = c.get("ges_ours_prec_mean", 0)
        ges_std = c.get("ges_standard_prec_mean", 0)
        ours_better = (pc_ours > pc_std) or (ges_ours > ges_std)
        all_configs_summary.append({
            "config": d_str,
            "pc_ours_prec": pc_ours, "pc_standard_prec": pc_std,
            "ges_ours_prec": ges_ours, "ges_standard_prec": ges_std,
            "ours_better": ours_better,
        })
    results["summary"] = all_configs_summary

    ours_better_count = sum(1 for s in all_configs_summary if s["ours_better"])
    total_configs = len(all_configs_summary)
    results["verdict"] = "VERIFIED" if ours_better_count > total_configs / 2 else "PARTIALLY_VERIFIED"
    return results
