"""Evolutionary selection model constructions.

Implements:
  - Definition 1: evolutionary selection DAG G^(T)
  - Definition 2: clique-augmented DAG G+
  - Theorem 3:    multi-domain clique-augmented DAG G+I
  - Lemma 1:      selection-induced dependencies
"""
from __future__ import annotations

from typing import Dict, Set, List, Tuple, Optional
import itertools
import random

from graph import DAG


# ---------------------------------------------------------------------------
# Definition 1 — Evolutionary Selection Model G^(T)
# ---------------------------------------------------------------------------

def construct_evolutionary_dag(
    G: DAG,
    selection_node: str,
    T: int,
    trait_nodes: Optional[List[str]] = None,
) -> Tuple[DAG, List[str]]:
    """Build the evolutionary selection DAG G^(T) per Definition 1.

    Parameters
    ----------
    G : DAG over vertices X ∪ {selection_node}.
    selection_node : the selection variable S in G.
    T : number of generations (≥ 1).
    trait_nodes : ordered list of trait variable names. If None, inferred from G.

    Returns
    -------
    (G^(T), trait_nodes) where G^(T) has the four edge types:
      1. X_i^(t) → X_j^(t)   (causal effects within generation)
      2. X_i^(t) → S^(t)      (trait effects on reproduction)
      3. ε_i^(t) → X_i^(t)    (exogenous → trait)
      4. ε_i^(t) → ε_i^(t+1)  (inheritance)
    """
    if trait_nodes is None:
        trait_nodes = sorted(G.nodes - {selection_node})

    d = len(trait_nodes)
    assert d > 0, "Need at least one trait node"

    # Enumerate all nodes in G^(T)
    all_nodes: Set[str] = set()
    for t in range(T + 1):
        for i in range(d):
            all_nodes.add(f"X{i}@{t}")
            all_nodes.add(f"eps{i}@{t}")
        if t < T:
            all_nodes.add(f"S@{t}")

    evo = DAG(all_nodes)

    for t in range(T + 1):
        # Edge type 1: trait → trait within generation
        for xi in trait_nodes:
            idx_i = trait_nodes.index(xi)
            for xj in trait_nodes:
                if G.has_edge(xi, xj):
                    idx_j = trait_nodes.index(xj)
                    evo.add_edge(f"X{idx_i}@{t}", f"X{idx_j}@{t}")

        # Edge type 2: trait → S (only for t < T)
        if t < T:
            for xi in trait_nodes:
                if G.has_edge(xi, selection_node):
                    idx_i = trait_nodes.index(xi)
                    evo.add_edge(f"X{idx_i}@{t}", f"S@{t}")

        # Edge type 3: ε → X
        for i in range(d):
            evo.add_edge(f"eps{i}@{t}", f"X{i}@{t}")

        # Edge type 4: ε^(t) → ε^(t+1) (inheritance, only for t < T)
        if t < T:
            for i in range(d):
                evo.add_edge(f"eps{i}@{t}", f"eps{i}@{t+1}")

    return evo, [f"X{i}@{T}" for i in range(d)]


def get_selection_nodes(T: int) -> List[str]:
    """Return the list of selection nodes S^(0), ..., S^(T-1) to condition on."""
    return [f"S@{t}" for t in range(T)]


# ---------------------------------------------------------------------------
# Definition 2 — Clique-Augmented DAG G+
# ---------------------------------------------------------------------------

def ancestors_of_selection(G: DAG, selection_node: str) -> Set[str]:
    r"""Return an_G(S) \ {S}: ancestors of S excluding S itself."""
    return G.ancestors({selection_node}) - {selection_node}


def clique_augmented_dag(
    G: DAG,
    selection_node: str,
    pi: Optional[List[str]] = None,
    trait_nodes: Optional[List[str]] = None,
) -> DAG:
    """Build the clique-augmented DAG G+ per Definition 2.

    G+ is a DAG over X where X_i → X_j ∈ G+ iff:
      (a) X_i → X_j ∈ G  (original causal edge), OR
      (b) {X_i, X_j} ⊆ an_G(S) and π(X_i) < π(X_j)  (clique augmentation).
    """
    if trait_nodes is None:
        trait_nodes = sorted(G.nodes - {selection_node})

    if pi is None:
        pi = G.topological_order()

    an_S = ancestors_of_selection(G, selection_node)
    an_S_traits = an_S & set(trait_nodes)

    # Position map for topological ordering
    pos = {n: i for i, n in enumerate(pi)}

    gplus = DAG(trait_nodes)

    for xi in trait_nodes:
        for xj in trait_nodes:
            if xi == xj:
                continue
            # Condition (a): original edge in G
            is_original = G.has_edge(xi, xj)
            # Condition (b): both are ancestors of S and xi precedes xj in pi
            is_clique = (
                xi in an_S_traits
                and xj in an_S_traits
                and pos.get(xi, 0) < pos.get(xj, 0)
            )
            if is_original or is_clique:
                gplus.add_edge(xi, xj)

    return gplus


# ---------------------------------------------------------------------------
# Theorem 3 — Multi-domain clique-augmented DAG G+I
# ---------------------------------------------------------------------------

def multi_domain_clique_augmented_dag(
    G: DAG,
    selection_node: str,
    changed_vars: Set[str],
    pi: Optional[List[str]] = None,
    trait_nodes: Optional[List[str]] = None,
) -> DAG:
    """Build the multi-domain clique-augmented DAG G+I per Theorem 3.

    Adds auxiliary vertex ζ to G+ and draws ζ → X_i for each changed X_i.
    If selection is changed (an_G(S) ∩ I ≠ ∅), adds ζ → all an_G(S)\\{S}.
    """
    if trait_nodes is None:
        trait_nodes = sorted(G.nodes - {selection_node})

    gplus = clique_augmented_dag(G, selection_node, pi, trait_nodes)
    gplus_I = gplus.copy()
    zeta = "zeta"
    gplus_I.add_edge  # ensure node exists
    # Add zeta as a node
    gplus_I._children[zeta] = set()
    gplus_I._parents[zeta] = set()

    an_S = ancestors_of_selection(G, selection_node)

    # Changed trait variables
    for xi in changed_vars:
        if xi != selection_node and xi in trait_nodes:
            gplus_I.add_edge(zeta, xi)

    # If selection is changed, add edges from zeta to all an_G(S) \ {S}
    if selection_node in changed_vars or (an_S & changed_vars):
        for node in an_S - {selection_node}:
            if node in trait_nodes:
                gplus_I.add_edge(zeta, node)

    return gplus_I


# ---------------------------------------------------------------------------
# Random graph generation
# ---------------------------------------------------------------------------

def random_dag_with_selection(
    d: int,
    avg_degree: float = 2.0,
    n_selection_parents: Optional[int] = None,
    seed: int = 0,
) -> Tuple[DAG, str, List[str]]:
    """Generate a random DAG G over d trait variables plus a selection node S.

    Uses an Erdős–Rényi model: for each ordered pair (i,j) with i < j in a random
    topological order, add edge with probability avg_degree / (d-1).
    Then connect n_selection_parents random traits as parents of S.
    """
    rng = random.Random(seed)
    trait_names = [f"X{i}" for i in range(d)]
    perm = trait_names.copy()
    rng.shuffle(perm)

    S = "S"
    all_nodes = trait_names + [S]
    G = DAG(all_nodes)

    edge_prob = avg_degree / max(d - 1, 1)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < edge_prob:
                G.add_edge(perm[i], perm[j])

    # Selection parents
    if n_selection_parents is None:
        n_selection_parents = max(1, d // 5)
    n_selection_parents = min(n_selection_parents, d)
    sel_parents = rng.sample(trait_names, n_selection_parents)
    for p in sel_parents:
        G.add_edge(p, S)

    return G, S, trait_names
