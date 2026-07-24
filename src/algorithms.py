"""Algorithm wrappers for PC, GES, and CDNOD from causal-learn.

Provides clean interfaces returning:
  - skeleton: set of undirected edges
  - CPDAG: directed and undirected edges
  - orientations: which edges are directed vs ambiguous
"""
from __future__ import annotations

import numpy as np
from typing import Set, Tuple, List, Dict, Optional
import warnings
import os
import contextlib

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def suppress_stderr():
    """Suppress stderr (used to hide causallearn progress bars)."""
    with open(os.devnull, "w") as devnull:
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)


def run_pc(
    data: np.ndarray,
    alpha: float = 0.05,
    indep_test: str = "fisherz",
    show_progress: bool = False,
) -> Dict:
    """Run PC algorithm and return skeleton + CPDAG.

    Returns dict with:
      'skeleton': set of frozenset({i,j}) undirected edges
      'directed': set of (i,j) tuples for i→j
      'undirected': set of frozenset({i,j}) for ambiguous edges
      'graph_matrix': raw causal-learn matrix
    """
    from causallearn.search.ConstraintBased.PC import pc

    kwargs = {"alpha": alpha, "show_progress": show_progress}
    if indep_test == "fisherz":
        kwargs["indep_test"] = "fisherz"
    elif indep_test == "kci":
        kwargs["indep_test"] = "kci"

    with suppress_stderr():
        cg = pc(data, **kwargs)
    M = cg.G.graph
    n = M.shape[0]

    skeleton: Set[frozenset] = set()
    directed: Set[Tuple[int, int]] = set()
    undirected: Set[frozenset] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0 or M[j, i] != 0:
                skeleton.add(frozenset({i, j}))
                if M[i, j] == -1 and M[j, i] == 1:
                    directed.add((i, j))  # i → j (tail at i, arrowhead at j)
                elif M[i, j] == 1 and M[j, i] == -1:
                    directed.add((j, i))  # j → i (tail at j, arrowhead at i)
                elif M[i, j] == -1 and M[j, i] == -1:
                    undirected.add(frozenset({i, j}))
                elif M[i, j] == 1 and M[j, i] == 1:
                    undirected.add(frozenset({i, j}))  # bidirected = ambiguous

    return {
        "skeleton": skeleton,
        "directed": directed,
        "undirected": undirected,
        "graph_matrix": M,
    }


def run_ges(
    data: np.ndarray,
    maxP: Optional[int] = None,
    score_func: str = "local_score_BIC",
) -> Dict:
    """Run GES algorithm and return skeleton + CPDAG."""
    from causallearn.search.ScoreBased.GES import ges

    if maxP is None:
        maxP = data.shape[1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with suppress_stderr():
            Record = ges(data, score_func=score_func, maxP=maxP)

    M = Record["G"].graph
    n = M.shape[0]

    skeleton: Set[frozenset] = set()
    directed: Set[Tuple[int, int]] = set()
    undirected: Set[frozenset] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0 or M[j, i] != 0:
                skeleton.add(frozenset({i, j}))
                if M[i, j] == -1 and M[j, i] == 1:
                    directed.add((i, j))  # i → j
                elif M[i, j] == 1 and M[j, i] == -1:
                    directed.add((j, i))  # j → i
                elif M[i, j] == -1 and M[j, i] == -1:
                    undirected.add(frozenset({i, j}))
                elif M[i, j] == 1 and M[j, i] == 1:
                    undirected.add(frozenset({i, j}))

    return {
        "skeleton": skeleton,
        "directed": directed,
        "undirected": undirected,
        "graph_matrix": M,
    }


def run_cdnod(
    data_list: List[np.ndarray],
    alpha: float = 0.05,
    indep_test: str = "fisherz",
) -> Dict:
    """Run CDNOD on multi-domain data.

    Parameters
    ----------
    data_list : list of (n_k, d) arrays, one per domain.

    Returns dict with skeleton, directed, undirected edges over the d trait
    variables plus domain indicator.
    """
    from causallearn.search.ConstraintBased.CDNOD import cdnod

    K = len(data_list)
    d = data_list[0].shape[1]

    # Stack data and build domain index
    data = np.vstack(data_list)
    c_indx = np.concatenate([np.full(X.shape[0], k) for k, X in enumerate(data_list)])
    c_indx = c_indx.reshape(-1, 1)

    with suppress_stderr():
        cg = cdnod(data, c_indx, alpha=alpha, indep_test=indep_test,
                   show_progress=False, verbose=False)

    M = cg.G.graph
    n = M.shape[0]

    skeleton: Set[frozenset] = set()
    directed: Set[Tuple[int, int]] = set()
    undirected: Set[frozenset] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if M[i, j] != 0 or M[j, i] != 0:
                skeleton.add(frozenset({i, j}))
                if M[i, j] == -1 and M[j, i] == 1:
                    directed.add((i, j))  # i → j
                elif M[i, j] == 1 and M[j, i] == -1:
                    directed.add((j, i))  # j → i
                elif M[i, j] == -1 and M[j, i] == -1:
                    undirected.add(frozenset({i, j}))
                elif M[i, j] == 1 and M[j, i] == 1:
                    undirected.add(frozenset({i, j}))

    return {
        "skeleton": skeleton,
        "directed": directed,
        "undirected": undirected,
        "graph_matrix": M,
    }


def cpdag_from_dag(dag_matrix: np.ndarray) -> Dict:
    """Given a DAG adjacency matrix, compute its CPDAG (Meek rules)."""
    from causallearn.utils.cit import fisherz
    from causallearn.utils.DAG2CPDAG import dag2cpdag

    # dag2cpdag expects a numpy array DAG
    cpdag = dag2cpdag(dag_matrix)
    n = cpdag.shape[0]

    skeleton: Set[frozenset] = set()
    directed: Set[Tuple[int, int]] = set()
    undirected: Set[frozenset] = set()

    for i in range(n):
        for j in range(i + 1, n):
            if cpdag[i, j] != 0 or cpdag[j, i] != 0:
                skeleton.add(frozenset({i, j}))
                if cpdag[i, j] == -1 and cpdag[j, i] == 1:
                    directed.add((i, j))  # i → j
                elif cpdag[i, j] == 1 and cpdag[j, i] == -1:
                    directed.add((j, i))  # j → i
                elif cpdag[i, j] == -1 and cpdag[j, i] == -1:
                    undirected.add(frozenset({i, j}))

    return {
        "skeleton": skeleton,
        "directed": directed,
        "undirected": undirected,
        "graph_matrix": cpdag,
    }
