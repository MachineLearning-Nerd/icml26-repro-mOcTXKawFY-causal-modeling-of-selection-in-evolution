"""Graph utilities: DAG operations, d-separation, topological ordering.

Independent implementation of d-separation via the ancestral moral graph
algorithm (Lauritzen et al., 1990). Cross-checked against networkx.d_separated.
"""
from __future__ import annotations

from typing import Set, Dict, List, Optional, Iterable
import itertools


class DAG:
    """Directed acyclic graph backed by an adjacency-list representation."""

    def __init__(self, nodes: Iterable[str], edges: Optional[Iterable[tuple[str, str]]] = None):
        self._children: Dict[str, Set[str]] = {n: set() for n in nodes}
        self._parents: Dict[str, Set[str]] = {n: set() for n in nodes}
        if edges:
            for a, b in edges:
                self.add_edge(a, b)

    @classmethod
    def from_dict(cls, adj: Dict[str, Set[str]]) -> "DAG":
        nodes = set(adj.keys())
        for children in adj.values():
            nodes.update(children)
        g = cls(nodes)
        for a, children in adj.items():
            for b in children:
                g.add_edge(a, b)
        return g

    def add_edge(self, a: str, b: str) -> None:
        self._children.setdefault(a, set()).add(b)
        self._children.setdefault(b, set())
        self._parents.setdefault(b, set()).add(a)
        self._parents.setdefault(a, set())

    @property
    def nodes(self) -> Set[str]:
        return set(self._children.keys())

    def children(self, n: str) -> Set[str]:
        return self._children.get(n, set())

    def parents(self, n: str) -> Set[str]:
        return self._parents.get(n, set())

    def neighbors(self, n: str) -> Set[str]:
        return self.children(n) | self.parents(n)

    def has_edge(self, a: str, b: str) -> bool:
        return b in self._children.get(a, set())

    def edges(self) -> List[tuple[str, str]]:
        return [(a, b) for a, children in self._children.items() for b in children]

    def ancestors(self, nodes: Iterable[str]) -> Set[str]:
        """Return all ancestors of *nodes* (including themselves)."""
        result: Set[str] = set()
        stack = list(nodes)
        while stack:
            n = stack.pop()
            if n in result:
                continue
            result.add(n)
            for p in self._parents.get(n, set()):
                if p not in result:
                    stack.append(p)
        return result

    def descendants(self, nodes: Iterable[str]) -> Set[str]:
        result: Set[str] = set()
        stack = list(nodes)
        while stack:
            n = stack.pop()
            if n in result:
                continue
            result.add(n)
            for c in self._children.get(n, set()):
                if c not in result:
                    stack.append(c)
        return result

    def topological_order(self) -> List[str]:
        """Return a topological ordering of all nodes."""
        in_deg = {n: len(self._parents.get(n, set())) for n in self.nodes}
        queue = sorted([n for n in self.nodes if in_deg[n] == 0])
        order: List[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in sorted(self._children.get(n, set())):
                in_deg[c] -= 1
                if in_deg[c] == 0:
                    queue.append(c)
            queue.sort()
        if len(order) != len(self.nodes):
            raise ValueError("Graph has a cycle")
        return order

    def copy(self) -> "DAG":
        g = DAG(self.nodes)
        for a, b in self.edges():
            g.add_edge(a, b)
        return g

    def subgraph(self, nodes: Iterable[str]) -> "DAG":
        keep = set(nodes)
        g = DAG(keep)
        for a, b in self.edges():
            if a in keep and b in keep:
                g.add_edge(a, b)
        return g

    # -- d-separation -------------------------------------------------------

    def d_separated(self, x: Set[str], y: Set[str], z: Set[str]) -> bool:
        """Test whether *x* is d-separated from *y* given *z*.

        Uses the ancestral moral graph algorithm:
        1. Build ancestral graph of x ∪ y ∪ z.
        2. Moralise (connect co-parents, drop directions).
        3. Remove z.
        4. Check connectivity between x and y.
        """
        x, y, z = set(x), set(y), set(z)
        relevant = self.ancestors(x | y | z)
        sub = self.subgraph(relevant)
        # Moralise: connect parents of common children, then make undirected
        undirected: Dict[str, Set[str]] = {n: set() for n in relevant}
        for a, b in sub.edges():
            undirected[a].add(b)
            undirected[b].add(a)
        for n in relevant:
            pars = sub.parents(n)
            for p1, p2 in itertools.combinations(pars, 2):
                undirected[p1].add(p2)
                undirected[p2].add(p1)
        # Remove conditioning set
        for n in z:
            if n in undirected:
                for nb in undirected[n]:
                    undirected[nb].discard(n)
                undirected[n] = set()
        # BFS from x; if we reach y, they are d-connected
        visited: Set[str] = set()
        queue = list(x - z)
        visited.update(queue)
        while queue:
            n = queue.pop(0)
            if n in y:
                return False
            for nb in undirected.get(n, set()):
                if nb not in visited and nb not in z:
                    visited.add(nb)
                    queue.append(nb)
        return True

    def all_d_separation_triples(self, nodes: List[str]) -> List[tuple[str, str, frozenset[str]]]:
        """Enumerate all (a, b, C) triples where a, b are distinct nodes and
        C is any subset of the remaining nodes, recording the d-separation verdict."""
        results = []
        nodes = list(nodes)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                remaining = [n for n in nodes if n != a and n != b]
                for r in range(len(remaining) + 1):
                    for C in itertools.combinations(remaining, r):
                        C_set = frozenset(C)
                        sep = self.d_separated({a}, {b}, set(C))
                        results.append((a, b, C_set, sep))
        return results


def cross_check_dsep(dag: DAG, x: Set[str], y: Set[str], z: Set[str]) -> bool:
    """Cross-check d-separation using networkx."""
    from networkx.algorithms.d_separation import is_d_separator
    import networkx as nx

    g = nx.DiGraph()
    for n in dag.nodes:
        g.add_node(n)
    for a, b in dag.edges():
        g.add_edge(a, b)
    return is_d_separator(g, x, y, z)
