#emerge_phenotype.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np
from collections import defaultdict
from numpy.typing import NDArray
from scipy.stats import binomtest, betabinom, permutation_test
from scipy.optimize import minimize

from emerge_data import MotifNode, MotifEdge
from emerge_forest import MotifForest

def bh_fdr(pvals: NDArray[np.float64], q: float = 0.05) -> Optional[float]:
    pvals = pvals[np.isfinite(pvals)]
    m = pvals.size
    if m == 0:
        return None

    order = np.argsort(pvals)
    sorted_p = pvals[order]

    thresholds = q * np.arange(1, m + 1) / m
    ok = sorted_p <= thresholds
    if not ok.any():
        return None

    k = np.max(np.where(ok)[0])
    cutoff = float(sorted_p[k])

    return cutoff

# add capability for different type of pvals later
# add 5to3, mle col support later
class ForestEdges:
    def __init__(self, forest: MotifForest, q: float = 0.05) -> None:
        self.forest = forest
        self.q = q

    @staticmethod
    def _perm_pval(
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        *,
        n_resamples: int = 1000,
        seed: int = 0
    ) -> float:

        if X.size < 2 or Y.size < 2:
            return np.nan
        def stat(X, Y):
            return float(np.mean(X) - np.mean(Y))

        res = permutation_test(
            data=(X, Y),
            statistic=stat,
            permutation_type='independent',
            alternative='greater',
            n_resamples=n_resamples,
            random_state=seed,
            vectorized=False
        )
        return float(res.pvalue)

    # change 'mle' and '5to3' to be seq_col, edit_col args from EmergeHandler
    @staticmethod
    def _get_not_child_edits(
        child: MotifNode,
        parent: MotifNode
    ) -> NDArray[np.float64]:
        child_seqs = set(child.seqs['5to3'])
        mask = ~parent.seqs['5to3'].isin(child_seqs)
        return parent.seqs.loc[mask, 'mle'].to_numpy(dtype=float)

    def _compute_pval(self, child: MotifNode, parent: MotifNode) -> float:
        child_edits = child.seqs['mle'].to_numpy(dtype=float)
        not_child_edits = self._get_not_child_edits(child, parent)
        return self._perm_pval(X=child_edits, Y=not_child_edits)

    def compute_pvals(self, with_canopy: bool = True) -> None:
        for edge in self.forest.iter_edges(with_canopy=with_canopy):
            pval = self._compute_pval(edge.child, edge.parent)
            edge.pval = pval

    # sp = supported positive
    def assign_sp(self, with_canopy: bool = True) -> None:
        p_list = []
        for e in self.forest.iter_edges(with_canopy=with_canopy):
            if not hasattr(e, 'pval'):
                raise RuntimeError(
                    'Edge p-values not computed; run compute_pvals() first.'
                )
                p_list.append(e.pval)
        pvals = np.array(p_list, dtype=float)
        cutoff = bh_fdr(pvals, q=self.q)
        for edge in self.forest.iter_edges(with_canopy=with_canopy):
            edge.sp = (cutoff is not None) and (edge.pval <= cutoff)

# add capability for different type of pvals later
# currently deprecating append_edits_bb (beta-binomial modeling of node editing)
# please see archived emerge_phenotype for old beta-binomial code
class ForestNodes:
    def __init__(self, forest: MotifForest, q: float = 0.05) -> None:
        self.forest = forest
        self.q = q

    def _compute_edit(self, node: MotifNode) -> float:
        if node.seqs is None or node.seqs.empty:
            return 0.0
        mle = node.seqs['mle'].to_numpy(dtype=float)
        if mle.size == 0:
            return 0.0
        return float(np.mean(mle))

    # in the future maybe make sure motif_len is actually "number of
    # constrained positions;" it isn't now, strictly. just taking length of
    # motif sequence. Z, for example, is unconstrained and shouldn't contribute
    # to 0.25^L
    def _compute_pval(self, node: MotifNode) -> float:
        if node.seqs is None:
            return np.nan
        p0 = 0.25 ** node.motif_len
        k = int(node.seqs.shape[0])
        n = int(self.forest.df_len)
        res = binomtest(k=k, n=n, p=p0, alternative='greater')
        return float(res.pvalue)

    def compute_edits(self, with_canopy: bool = True) -> None:
        for node in self.forest.flatten(with_canopy=with_canopy):
            node.edit = self._compute_edit(node)

    def compute_pvals(self, with_canopy: bool = True) -> None:
        for node in self.forest.flatten(with_canopy=with_canopy):
            pval = self._compute_pval(node)
            node.pval = pval

    # sp = supported positive
    def assign_sp(self, with_canopy: bool = True) -> None:
        pvals = np.array(
            [n.pval for n in self.forest.flatten(with_canopy=with_canopy)],
            dtype=float
        )
        cutoff = bh_fdr(pvals, q=self.q)
        for node in self.forest.flatten(with_canopy=with_canopy):
            node.sp = (cutoff is not None) and (node.pval <= cutoff)

# need to add: depth statistic in BPE merger
# forest.iter_edges() method
class ForestPruner:
    def __init__(self, forest: MotifForest, q: float = 0.05) -> None:
        self.forest = forest
        self.over_nodes = ForestNodes(forest=forest, q=q)
        self.over_edges = ForestEdges(forest=forest, q=q)

    def prune_by_delta(
        self,
        with_canopy: bool = True,
        with_parents: bool = True
    ) -> None:
        self.over_edges.compute_pvals(with_canopy=with_canopy)
        self.over_edges.assign_sp(with_canopy=with_canopy)

        out = defaultdict(list)
        for e in self.forest.iter_edges(with_canopy=with_canopy):
            out[id(e.parent)].append(e)

        keep: set[int] = set()
        nodes = list(self.forest.flatten(with_canopy=with_canopy))

        for node in nodes:
            edges = out.get(id(node), [])
            s = sum(bool(getattr(e, 'sp', False)) for e in edges)

            node.motif_state = 2 if s >= 2 else 1 if s == 1 else 0
            if node.motif_state > 0:
                keep.add(id(node))

        if with_parents:
            for node in nodes:
                if id(node) in keep:
                    cur = getattr(node, 'parent', None)
                    while cur is not None:
                        keep.add(id(cur))
                        cur = getattr(cur, 'parent', None)

        for node in nodes:
            if id(node) in keep:
                continue
            parent = getattr(node, 'parent', None)
            if parent is None or id(parent) in keep:
                self.forest.prune(node)

    def prune_by_enrichment(
        self,
        with_canopy: bool = True,
        with_parents: bool = True
    ) -> None:
        self.over_nodes.compute_pvals(with_canopy=with_canopy)
        self.over_nodes.assign_sp(with_canopy=with_canopy)

        keep: set[int] = set()
        nodes = list(self.forest.flatten(with_canopy=with_canopy))

        for node in nodes:
            if getattr(node, 'sp', False):
                keep.add(id(node))

        if with_parents:
            for node in nodes:
                if id(node) in keep:
                    cur = getattr(node, 'parent', None)
                    while cur is not None:
                        keep.add(id(cur))
                        cur = getattr(cur, 'parent', None)

        for node in nodes:
            if id(node) in keep:
                continue
            parent = getattr(node, 'parent', None)
            if parent is None or id(parent) in keep:
                self.forest.prune(node)

