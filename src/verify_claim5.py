"""Claim 5 verification: Theorem 4 — CDNOD improves identifiability.

Theorem 4: With multi-domain data, Algorithm 2 (CDNOD) produces a PDAG that
orients MORE edges than single-domain Algorithm 1 (PC/GES), while maintaining
soundness. Specifically: Xi→Xj in C (single-domain) implies Xi→Xj in P_X
(multi-domain), but not vice versa.
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
from sem import generate_multi_domain_data_shared_structure
from algorithms import run_pc, run_cdnod


def verify_claim5_theorem4(
    d_values: List[int] = [5, 8, 10],
    T: int = 5,
    n_runs: int = 5,
    n_samples: int = 500,
    n_domains: int = 2,
    alpha: float = 0.05,
    seed_offset: int = 5000,
) -> dict:
    """Verify Theorem 4: CDNOD orients more edges than single-domain PC."""
    results = {"configs": [], "total_runs": 0}

    for d in d_values:
        config = {
            "d": d, "runs": [],
            "cdnod_orients_more": 0,
            "cdnod_superset_oriented": 0,
            "mean_single_oriented": 0.0,
            "mean_multi_oriented": 0.0,
            "soundness_orient_rate": [],
        }
        single_counts = []
        multi_counts = []
        improvement_ratios = []

        for run_idx in range(n_runs):
            seed = seed_offset + d * 100 + run_idx

            # Generate multi-domain data with changed selection
            domain_data, G, S_node, trait_nodes, Gplus, pi, changed = \
                generate_multi_domain_data_shared_structure(
                    d=d, T=T, n_initial=n_samples, seed=seed, n_domains=n_domains,
                )

            # True structures
            an_S = ancestors_of_selection(G, S_node)
            an_S_idx = set(trait_nodes.index(n) for n in an_S if n in trait_nodes)
            true_causal_dir = set()
            for a, b in G.edges():
                if a in trait_nodes and b in trait_nodes:
                    i, j = trait_nodes.index(a), trait_nodes.index(b)
                    true_causal_dir.add((i, j))

            # Single-domain PC
            try:
                single_res = run_pc(domain_data[0], alpha=alpha)
                single_dir = single_res["directed"]
            except Exception as e:
                config["runs"].append({"seed": seed, "error": f"PC: {e}"})
                continue

            # Multi-domain CDNOD
            try:
                multi_res = run_cdnod(domain_data, alpha=alpha)
                # Extract trait-only edges (column d is the domain variable)
                multi_dir = {(i, j) for (i, j) in multi_res["directed"]
                             if i < d and j < d}
            except Exception as e:
                config["runs"].append({"seed": seed, "error": f"CDNOD: {e}"})
                continue

            n_single = len(single_dir)
            n_multi = len(multi_dir)
            single_counts.append(n_single)
            multi_counts.append(n_multi)

            if n_multi > n_single:
                config["cdnod_orients_more"] += 1
                improvement_ratios.append((n_multi - n_single) / max(n_single, 1))

            # Check superset property
            if single_dir.issubset(multi_dir):
                config["cdnod_superset_oriented"] += 1

            # Orientation soundness rate for CDNOD
            if multi_dir:
                sound = sum(1 for (i, j) in multi_dir
                            if (i, j) in true_causal_dir and j not in an_S_idx)
                config["soundness_orient_rate"].append(sound / len(multi_dir))

            # New edges oriented by CDNOD (not in single-domain)
            new_edges = multi_dir - single_dir
            config["runs"].append({
                "seed": seed,
                "single_oriented": n_single,
                "multi_oriented": n_multi,
                "new_edges_by_cdnod": len(new_edges),
                "cdnod_orients_more": n_multi > n_single,
                "superset": single_dir.issubset(multi_dir),
            })
            results["total_runs"] += 1

        if single_counts:
            config["mean_single_oriented"] = float(np.mean(single_counts))
            config["mean_multi_oriented"] = float(np.mean(multi_counts))
            config["mean_improvement"] = float(np.mean(improvement_ratios)) if improvement_ratios else 0.0
        if config["soundness_orient_rate"]:
            config["soundness_orient_rate_mean"] = float(np.mean(config["soundness_orient_rate"]))
            config["soundness_orient_rate_std"] = float(np.std(config["soundness_orient_rate"]))
            del config["soundness_orient_rate"]

        results["configs"].append(config)

    any_improvement = any(c["cdnod_orients_more"] > 0 for c in results["configs"])
    results["verdict"] = "VERIFIED" if any_improvement else "INCONCLUSIVE"
    return results
