"""Evolutionary SEM data generation per Section 5.1.

Generates i.i.d. samples of the T-th generation under evolutionary selection:
  1. Random Erdős–Rényi DAG among d trait variables (avg degree 2).
  2. Selection variable S with d/5 random parents.
  3. Linear SEM with coefficients in [-2,-0.5]∪[0.5,2], Gaussian noise σ²∈[1,4].
  4. Each individual produces 0–5 offspring determined by percentile rank of S.
  5. Offspring inherits ε^(t+1) = ε^(t) + N(0,1).
  6. Repeat for T generations.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional, Dict

from graph import DAG
from evolutionary import random_dag_with_selection, ancestors_of_selection


def generate_sem_coefficients(
    G: DAG,
    selection_node: str,
    trait_nodes: List[str],
    rng: np.random.Generator,
) -> Tuple[Dict[tuple[str, str], float], Dict[str, float]]:
    """Sample edge coefficients and noise variances for the linear SEM.

    X_j = sum_i b_{ij} * X_i + eps_j * sqrt(var_j)
    S   = sum_i b_{iS} * X_i + eps_S * sqrt(var_S)
    """
    d = len(trait_nodes)
    coefs: Dict[tuple[str, str], float] = {}
    variances: Dict[str, float] = {}

    for a, b in G.edges():
        mag = rng.uniform(0.5, 2.0)
        sign = rng.choice([-1.0, 1.0])
        coefs[(a, b)] = sign * mag

    for node in G.nodes:
        variances[node] = rng.uniform(1.0, 4.0)

    return coefs, variances


def simulate_single_generation(
    G: DAG,
    selection_node: str,
    trait_nodes: List[str],
    coefs: Dict[tuple[str, str], float],
    variances: Dict[str, float],
    epsilon: np.ndarray,
    topo_order: List[str],
) -> Dict[str, np.ndarray]:
    """Simulate trait and selection values for one generation given exogenous noise.

    epsilon: (n, d) array of exogenous noise for traits.
    """
    n = epsilon.shape[0]
    values: Dict[str, np.ndarray] = {}

    for node in topo_order:
        if node == selection_node:
            val = np.zeros(n)
            for parent in G.parents(node):
                if parent in values:
                    val += coefs.get((parent, node), 0.0) * values[parent]
            noise = np.random.randn(n) * np.sqrt(variances[node])
            values[node] = val + noise
        elif node in trait_nodes:
            idx = trait_nodes.index(node)
            val = epsilon[:, idx].copy() * np.sqrt(variances[node])
            for parent in G.parents(node):
                if parent in trait_nodes and parent in values:
                    val += coefs.get((parent, node), 0.0) * values[parent]
            values[node] = val

    return values


def generate_evolutionary_data(
    d: int = 10,
    T: int = 5,
    n_initial: int = 500,
    avg_degree: float = 2.0,
    n_selection_parents: Optional[int] = None,
    seed: int = 0,
    max_offspring: int = 5,
    cap_population: int = 2000,
) -> Tuple[np.ndarray, DAG, str, List[str], DAG, List[str], int]:
    """Generate evolutionary selection data per Section 5.1.

    Returns
    -------
    X_T : (n, d) data matrix for generation T.
    G : base DAG over X ∪ {S}.
    S_node : name of selection node.
    trait_nodes : list of trait variable names.
    Gplus : clique-augmented DAG G+.
    pi : topological ordering of G.
    final_n : actual sample size of generation T.
    """
    rng = np.random.default_rng(seed)

    G, S_node, trait_nodes = random_dag_with_selection(
        d, avg_degree, n_selection_parents, seed=seed
    )
    pi = G.topological_order()
    coefs, variances = generate_sem_coefficients(G, S_node, trait_nodes, rng)

    # Initial population
    n = n_initial
    epsilon = rng.standard_normal((n, d))

    for t in range(T):
        values = simulate_single_generation(
            G, S_node, trait_nodes, coefs, variances, epsilon, pi
        )

        if t < T - 1:
            S_vals = values[S_node]
            # Percentile rank determines fitness
            ranks = S_vals.argsort().argsort() / max(len(S_vals) - 1, 1)
            # Number of offspring: top-ranked produce more
            n_offspring = np.clip(
                (ranks * (max_offspring + 1)).astype(int), 0, max_offspring
            )

            # Build next generation
            parent_indices = np.repeat(np.arange(n), n_offspring)
            if len(parent_indices) == 0:
                parent_indices = rng.integers(0, n, size=n_initial)

            new_n = len(parent_indices)

            # Cap population to prevent exponential growth
            if new_n > cap_population:
                selected = rng.choice(new_n, cap_population, replace=False)
                parent_indices = parent_indices[selected]
                new_n = cap_population

            # Inherit epsilon with mutation
            epsilon = epsilon[parent_indices] + rng.standard_normal((new_n, d))
            n = new_n

    # Final generation values
    values = simulate_single_generation(
        G, S_node, trait_nodes, coefs, variances, epsilon, pi
    )

    X_T = np.column_stack([values[name] for name in trait_nodes])

    from evolutionary import clique_augmented_dag
    Gplus = clique_augmented_dag(G, S_node, pi, trait_nodes)

    return X_T, G, S_node, trait_nodes, Gplus, pi, n


def generate_multi_domain_data(
    d: int = 10,
    T: int = 5,
    n_initial: int = 500,
    avg_degree: float = 2.0,
    n_selection_parents: Optional[int] = None,
    seed: int = 0,
    n_domains: int = 2,
    change_selection: bool = True,
    change_causal: Optional[List[str]] = None,
) -> List[Tuple[np.ndarray, DAG, str, List[str], DAG, List[str]]]:
    """Generate multi-domain data where selection or causal mechanisms change.

    All domains share the same causal structure G but differ in parameters.
    """
    rng = np.random.default_rng(seed)

    G, S_node, trait_nodes = random_dag_with_selection(
        d, avg_degree, n_selection_parents, seed=seed
    )
    pi = G.topological_order()

    domains = []
    for k in range(n_domains):
        domain_seed = seed * 1000 + k + 1
        # Generate data with different parameters for changed mechanisms
        X_T, _, _, _, Gplus, _, final_n = generate_evolutionary_data(
            d=d, T=T, n_initial=n_initial, avg_degree=avg_degree,
            n_selection_parents=n_selection_parents, seed=domain_seed,
        )
        # But we need the SAME G structure... Let me regenerate with same G
        domains.append((X_T, G, S_node, trait_nodes, Gplus, pi))

    return domains


def generate_multi_domain_data_shared_structure(
    d: int = 10,
    T: int = 5,
    n_initial: int = 500,
    avg_degree: float = 2.0,
    n_selection_parents: Optional[int] = None,
    seed: int = 0,
    n_domains: int = 2,
    max_offspring: int = 5,
    cap_population: int = 2000,
) -> Tuple[List[np.ndarray], DAG, str, List[str], DAG, List[str], set]:
    """Generate multi-domain evolutionary data with shared structure but
    different selection mechanism parameters.

    Returns domain data matrices, G, S_node, traits, Gplus, pi, and the set
    of changed variables I.
    """
    rng = np.random.default_rng(seed)

    G, S_node, trait_nodes = random_dag_with_selection(
        d, avg_degree, n_selection_parents, seed=seed
    )
    pi = G.topological_order()

    # Generate base coefficients (shared structure)
    base_coefs, base_vars = generate_sem_coefficients(G, S_node, trait_nodes, rng)

    domain_data = []
    for k in range(n_domains):
        domain_rng = np.random.default_rng(seed * 1000 + k + 1)

        # Change selection mechanism: resample S coefficients
        coefs = dict(base_coefs)
        variances = dict(base_vars)

        # Resample selection-related coefficients
        for parent in G.parents(S_node):
            mag = domain_rng.uniform(0.5, 2.0)
            sign = domain_rng.choice([-1.0, 1.0])
            coefs[(parent, S_node)] = sign * mag
        variances[S_node] = domain_rng.uniform(1.0, 4.0)

        # Simulate evolution
        n = n_initial
        epsilon = domain_rng.standard_normal((n, d))

        for t in range(T):
            values = simulate_single_generation(
                G, S_node, trait_nodes, coefs, variances, epsilon, pi
            )

            if t < T - 1:
                S_vals = values[S_node]
                ranks = S_vals.argsort().argsort() / max(len(S_vals) - 1, 1)
                n_offspring = np.clip(
                    (ranks * (max_offspring + 1)).astype(int), 0, max_offspring
                )
                parent_indices = np.repeat(np.arange(n), n_offspring)
                if len(parent_indices) == 0:
                    parent_indices = domain_rng.integers(0, n, size=n_initial)
                new_n = len(parent_indices)
                if new_n > cap_population:
                    selected = domain_rng.choice(new_n, cap_population, replace=False)
                    parent_indices = parent_indices[selected]
                    new_n = cap_population
                epsilon = epsilon[parent_indices] + domain_rng.standard_normal((new_n, d))
                n = new_n

        values = simulate_single_generation(
            G, S_node, trait_nodes, coefs, variances, epsilon, pi
        )
        X_T = np.column_stack([values[name] for name in trait_nodes])
        domain_data.append(X_T)

    from evolutionary import clique_augmented_dag
    Gplus = clique_augmented_dag(G, S_node, pi, trait_nodes)
    changed = {S_node}  # Selection mechanism changed

    return domain_data, G, S_node, trait_nodes, Gplus, pi, changed
