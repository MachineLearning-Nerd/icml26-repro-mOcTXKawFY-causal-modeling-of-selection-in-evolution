"""Main orchestrator: runs all claim verifications and outputs results.

Usage: uv run python src/main.py
"""
from __future__ import annotations

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def serialize(obj):
    """Make numpy types JSON-serializable."""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return [serialize(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    t_start = time.time()
    all_results = {}
    git_sha = os.popen("git rev-parse HEAD").read().strip()
    all_results["meta"] = {
        "git_sha": git_sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python_version": sys.version,
    }

    # =====================================================================
    # Claims 1-3: Graphical verification (exhaustive, large-scale)
    # =====================================================================
    print("=" * 70)
    print("CLAIMS 1-3: Graphical verification (Definitions 1-2, Lemma 1, Theorem 1)")
    print("=" * 70)

    from verify_claims_1_3 import (
        verify_claim1_definition1,
        verify_claim2_lemma1,
        verify_claim3_theorem1,
    )

    # Claim 1: Definition 1 — evolutionary selection model construction
    t0 = time.time()
    print("\n[C1] Definition 1: Evolutionary selection model G^(T)...")
    r1 = verify_claim1_definition1(
        d_values=[5, 6, 7, 8],
        T_values=[1, 2, 3, 5, 10],
        n_graphs_per_config=20,
    )
    print(f"  Verdict: {r1['verdict']}")
    print(f"  Graphs tested: {r1['total_graphs']}")
    print(f"  Time: {time.time()-t0:.1f}s")
    all_results["claim1"] = r1

    # Claim 2: Lemma 1 — selection-induced dependencies
    t0 = time.time()
    print("\n[C2] Lemma 1: Selection-induced dependencies...")
    r2 = verify_claim2_lemma1(
        d_values=[5, 6, 7, 8, 10],
        n_graphs_per_config=50,
    )
    print(f"  Verdict: {r2['verdict']}")
    print(f"  Graphs tested: {r2['total_graphs']}")
    print(f"  G+ strict supergraph count: {r2['gplus_strict_count']}/{r2['total_graphs']}")
    print(f"  Extra edges total: {r2['extra_edges_total']}")
    print(f"  Counterexamples (deps from evolution): {r2['counterexamples_found']}")
    print(f"  Time: {time.time()-t0:.1f}s")
    all_results["claim2"] = r2

    # Claim 3: Theorem 1 — d-separation equivalence
    t0 = time.time()
    print("\n[C3] Theorem 1: d-separation equivalence G^(T)|S <-> G+...")
    r3 = verify_claim3_theorem1(
        d_values=[5, 6, 7],
        T_values=[1, 2, 3, 5],
        n_graphs_per_config=30,
    )
    print(f"  Verdict: {r3['verdict']}")
    print(f"  Total triples tested: {r3['total_triples']}")
    print(f"  Total mismatches: {r3['total_mismatches']}")
    print(f"  NX cross-check mismatches: {r3.get('nx_crosscheck_mismatches', 'N/A')}")
    print(f"  Time: {time.time()-t0:.1f}s")
    all_results["claim3"] = r3

    # =====================================================================
    # Claim 4: Theorem 2 — PC/GES soundness and completeness
    # =====================================================================
    print("\n" + "=" * 70)
    print("CLAIM 4: Theorem 2 — PC/GES soundness and completeness on G+")
    print("=" * 70)

    from verify_claim4 import verify_claim4_theorem2

    t0 = time.time()
    print("\n[C4] Running PC and GES on evolutionary SEM data...")
    r4 = verify_claim4_theorem2(
        d_values=[10, 15, 20],
        T_values=[2, 4, 8],
        n_runs=10,
        n_samples=500,
    )
    for cfg in r4.get("configs", []):
        print(f"  d={cfg['d']}, T={cfg['T']}: "
              f"PC adj_prec={cfg.get('pc_adj_precision_mean', 0):.3f}, "
              f"PC ours_prec={cfg.get('pc_ours_prec_mean', 0):.3f}, "
              f"PC std_prec={cfg.get('pc_standard_prec_mean', 0):.3f}, "
              f"GES ours_prec={cfg.get('ges_ours_prec_mean', 0):.3f}")
    print(f"  Verdict: {r4['verdict']}")
    print(f"  Total runs: {r4['total_runs']}")
    print(f"  Time: {time.time()-t0:.1f}s")
    all_results["claim4"] = r4

    # =====================================================================
    # Claim 5: Theorem 4 — CDNOD multi-domain identifiability
    # =====================================================================
    print("\n" + "=" * 70)
    print("CLAIM 5: Theorem 4 — CDNOD improves identifiability")
    print("=" * 70)

    from verify_claim5 import verify_claim5_theorem4

    t0 = time.time()
    print("\n[C5] Running CDNOD on multi-domain evolutionary data...")
    r5 = verify_claim5_theorem4(
        d_values=[5, 8, 10],
        T=5,
        n_runs=5,
        n_samples=500,
        n_domains=2,
    )
    for cfg in r5.get("configs", []):
        print(f"  d={cfg['d']}: "
              f"single_oriented={cfg.get('mean_single_oriented', 0):.1f}, "
              f"multi_oriented={cfg.get('mean_multi_oriented', 0):.1f}, "
              f"cdnod_more={cfg['cdnod_orients_more']}, "
              f"sound_rate={cfg.get('soundness_orient_rate_mean', 'N/A')}")
    print(f"  Verdict: {r5['verdict']}")
    print(f"  Total runs: {r5['total_runs']}")
    print(f"  Time: {time.time()-t0:.1f}s")
    all_results["claim5"] = r5

    # =====================================================================
    # Claim 6: Synthetic experiments (Figure 6) + real datasets
    # =====================================================================
    print("\n" + "=" * 70)
    print("CLAIM 6: Synthetic Figure 6 + real-world datasets")
    print("=" * 70)

    from verify_claim6 import reproduce_figure6, run_real_data, load_avonet_data, load_cses_data

    t0 = time.time()
    print("\n[C6a] Reproducing Figure 6: precision of causal adjacencies (d=20)...")
    r6 = reproduce_figure6(
        d=20,
        T_values=[2, 4, 6, 8, 10],
        n_runs=20,
        n_samples=1000,
    )
    for T, summary in r6.get("per_T", {}).items():
        print(f"  T={T}: PC_std={summary.get('pc_standard_mean', 'N/A'):.3f}±{summary.get('pc_standard_std', 0):.3f}, "
              f"PC_ours={summary.get('pc_ours_mean', 'N/A'):.3f}±{summary.get('pc_ours_std', 0):.3f}, "
              f"GES_std={summary.get('ges_standard_mean', 'N/A'):.3f}±{summary.get('ges_standard_std', 0):.3f}, "
              f"GES_ours={summary.get('ges_ours_mean', 'N/A'):.3f}±{summary.get('ges_ours_std', 0):.3f}")
    print(f"  Verdict: {r6['verdict']}")
    print(f"  Ours better (PC): {r6['ours_better_pc']}/{r6['total_T']}")
    print(f"  Ours better (GES): {r6['ours_better_ges']}/{r6['total_T']}")
    print(f"  Time: {time.time()-t0:.1f}s")
    all_results["claim6_figure6"] = r6

    # Real data experiments
    print("\n[C6b] Running PC/GES on real-world datasets...")
    real_results = {}

    for dataset_name, loader in [("AVONET", load_avonet_data), ("CSES", load_cses_data)]:
        t0 = time.time()
        try:
            data, var_names = loader()
            print(f"  {dataset_name}: {data.shape[0]} samples, {data.shape[1]} variables")
            rr = run_real_data(dataset_name, data, var_names)
            real_results[dataset_name] = rr
            print(f"    PC skeleton: {len(rr.get('pc_skeleton', []))} edges, "
                  f"PC directed: {len(rr.get('pc_directed', []))} edges")
        except Exception as e:
            print(f"  {dataset_name}: ERROR - {e}")
            real_results[dataset_name] = {"error": str(e)}
        print(f"    Time: {time.time()-t0:.1f}s")

    all_results["claim6_real"] = real_results

    # =====================================================================
    # Summary
    # =====================================================================
    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for claim in ["claim1", "claim2", "claim3", "claim4", "claim5"]:
        verdict = all_results.get(claim, {}).get("verdict", "N/A")
        print(f"  {claim}: {verdict}")
    print(f"  claim6_figure6: {all_results.get('claim6_figure6', {}).get('verdict', 'N/A')}")
    for ds in real_results:
        if "error" in real_results[ds]:
            print(f"  claim6_{ds}: ERROR")
        else:
            print(f"  claim6_{ds}: COMPLETED")
    print(f"\n  Total time: {total_time:.1f}s")

    all_results["meta"]["total_time_s"] = total_time

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(output_path, "w") as f:
        json.dump(serialize(all_results), f, indent=2)
    print(f"\n  Results saved to {output_path}")

    # Exit nonzero if any critical claim failed
    critical_failures = []
    for claim in ["claim1", "claim2", "claim3"]:
        v = all_results.get(claim, {}).get("verdict", "")
        if v == "FALSIFIED":
            critical_failures.append(claim)
    if critical_failures:
        print(f"\n  CRITICAL FAILURES: {critical_failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
