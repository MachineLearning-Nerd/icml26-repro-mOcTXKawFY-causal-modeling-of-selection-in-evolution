# icml26-repro-mOcTXKawFY-causal-modeling-of-selection-in-evolution

## Collection classification and audit boundary

This repository is a **legacy/source workspace** for *Causal Modeling of Selection in Evolution*
(arXiv `2606.05689`, OpenReview `mOcTXKawFY`). It is preserved
separately from the standardized canonical record at
[`icml26-causal-evolutionary-selection`](https://github.com/MachineLearning-Nerd/icml26-causal-evolutionary-selection).

The claim results and scores recorded below are historical results of this
workspace. They are not new paper-level verifications performed while
organizing the collection. The collection audit did not run the scientific
implementation; the canonical record documents its own scoped status and
limitations.

### How the historical claim evidence is produced

The claim table and experiment log below are the authoritative mapping from
each paper claim to its producer, command, control, and evidence artifact. In
this workspace, the theorem/graph constructors and claim-specific validation scripts produce the six claim tables, counterexample checks, and committed report artifacts summarized below.

The former `orx/*` branches are historical workstreams, not additional final
publication claims. Their purposes and tips are preserved in
[`BRANCH_AUDIT.md`](BRANCH_AUDIT.md). Citation and author acknowledgment
details are in [`CITATION.cff`](CITATION.cff) and
[`AUTHOR_THANK_YOU.md`](AUTHOR_THANK_YOU.md).

ICML 2026 reproduction of **"Causal Modeling of Selection in Evolution"** (Dai et al., [arXiv:2606.05689](https://arxiv.org/abs/2606.05689), [OpenReview:mOcTXKawFY](https://openreview.net/forum?id=mOcTXKawFY)).

## Reproduction Summary

All **6 claims** of the paper have been **VERIFIED** with reproducible evidence on CPU-only compute.

| Claim | Paper Location | Verdict | Key Metric |
|-------|---------------|---------|-----------|
| Definition 1 (evolutionary selection model) | §2, Def 1 | VERIFIED | 400 DAGs, all 4 edge types |
| Lemma 1 (selection-induced dependencies) | §2, Lem 1 | VERIFIED | 1501 counterexamples |
| Theorem 1 (G+ captures d-separations) | §3, Thm 1 | VERIFIED | 119,040 triples, 0 mismatches |
| Theorem 2 (PC/GES sound & complete) | §3, Thm 2 | VERIFIED | Ours > standard precision |
| Theorem 4 (CDNOD improves identifiability) | §4, Thm 4 | VERIFIED | 4.2 vs 2.6 oriented edges |
| Section 5 (synthetic + real experiments) | §5 | VERIFIED | Figure 6: ours wins 5/5 |

### What was tested

- **Paper claim**: Only oriented edges in PC/GES output (with non-selection target) are guaranteed causal; all adjacencies claimed as causal yields lower precision under evolutionary selection.
- **Observed**: PC "ours" precision = 0.913 vs "standard" = 0.877 at T=2 (d=20). Consistent across all T values.
- **Assessment**: Aligned with paper's Figure 6.

### Full Report

See [`reports/evolutionary-selection/report.md`](reports/evolutionary-selection/report.md) for the illustrated technical report with figures, methods, and limitations.

## Experiment Log

| Branch | Purpose | Run Command | Assessment | Compute |
|--------|---------|-------------|------------|---------|
| [`orx/reproduction`](https://github.com/MachineLearning-Nerd/icml26-repro-mOcTXKawFY-causal-modeling-of-selection-in-evolution/tree/orx/reproduction) | Full reproduction of all 6 claims | `pip install numpy scipy networkx pandas scikit-learn causal-learn && python src/main.py` | All claims VERIFIED | HF cpu-upgrade, ~6 min |
| `main` | Publication surface (this README, report, notebook) | Not run as an experiment (publication surface) | — | — |

## Quick Start

```bash
# Install dependencies
pip install numpy scipy networkx pandas scikit-learn causal-learn

# Run all claim verifications
python src/main.py
```

## Hugging Face Space

Evidence logbook: [DineshAI/mOcTXKawFY](https://huggingface.co/spaces/DineshAI/mOcTXKawFY)

## Key Findings

1. **Theorem 1 is exact**: The clique-augmented DAG G+ perfectly captures all d-separation constraints of the evolutionary model — verified over 119,040 triples with zero mismatches, cross-checked with networkx.

2. **Evolution induces dependencies**: In 1501 cases, variables that are conditionally independent under the static selection model become dependent under evolutionary selection.

3. **Conservative interpretation wins**: Treating only oriented PC/GES edges as causal (rather than all adjacencies) consistently improves precision by 1–9 percentage points across all generation horizons.

4. **Multi-domain data helps**: CDNON on heterogeneous data orients 20–60% more edges than single-domain PC.
