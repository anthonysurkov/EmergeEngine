#emerge_phenotype.py
from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
from scipy.stats import binomtest, betabinom
from scipy.optimize import minimize
from emerge_data import MotifNode
from emerge_forest import MotifForest

class ForestPhenotyper():
    def __init__(
        self,
        forest: MotifForest,
    ):
        if not isinstance(forest, MotifForest):
            raise ValueError(
                'arg `forest` must be a MotifForest. '
                f'got: {type(forest)}'
            )
        self.forest = forest
        self.edits_appended: bool = False

    def append_edits_edgewise(self):
        self.forest.traverse(func=_compute_deltas)

    # delta := Y_S(v) - Y_S(k), Y := avg edit. see p.123 of notebook
    def _append_node_deltas(self):
        # encompass _compute_deltas(self) in this function. have it act on one
        # node. use append_edits_edgewise to traverse forest and compute
        # deltas for all. then, extend to do permutation p-vals in one pass,
        # instead of two traversals. 11:55am

    def _compute_deltas(self):
        if not self.edits_appended:
            self.append_edits()
        for node in forest:
            if len(node.parents) != 2:
                seq = node.motif_seq
                warnings.warn(
                    'length of parent is not 2 for node with seq {seq}'
                )
            l_parent, r_parent = node.parents
            # K := S(p) \ S(v), S(x) := seqs matching restriction x. see p. 123
            v_seqs = set(node.seqs)
            l_seqs = l_parent.seqs
            r_seqs = r_parent.seqs

            l_K = [seq for seq in l_seqs if seq not in v_seqs]
            r_K = [seq for seq in r_seqs if seq not in v_seqs]
            l_K_edit = self._get_avg_edit(l_K)
            r_K_edit = self._get_avg_edit(r_K)

            node.l_delta = node.avg_edit - l_K_edit
            node.r_delta = node.avg_edit - r_K_edit

    def prune_by_enrichment(
        self,
        q: float = 0.05
    ) -> None:
        def _compute_p(node: MotifNode):
            node.p = self._binom_test(node=node)

        self.forest.traverse(func=_compute_p)
        self.bh_fdr(forest=self.forest, q=q)

    def append_edits(self):
        self.forest.traverse(func=self._append_node_editing)
        self.edits_appended: bool = True

    def _append_node_editing(self, node: MotifNode) -> None:
        node.avg_edit = self._get_avg_edit(node.seqs)

        J = node.seqs['mle'].shape[0]
        node.var_edit = (
            (1 / (J - 1) *
            ((node.seqs['mle'] - node.avg_edit)**2).sum()
        )

    def _get_avg_edit(seqs: pd.DataFrame) -> float:
        J = seqs['mle'].shape[0]
        if J == 0:
            return 0
        if J == 1:
            return seqs['mle'].iloc[0]
        return seqs['mle'].sum() / J

    def append_edits_bb(
        self,
        node: MotifNode,
        correct_if_bb_bad: bool = True
    ) -> None:
        a, b, res = self._fit_beta_binom(node)
        if not res.success:
            if correct_if_bb_bad:
                self.append_node_editing(node)
                return
            warnings.warn(
                f'Beta-binomial fit did not converge for node ID {node.node_id}'
                f' / motif {node.motif_seq}, and correct_if_bb_bad behavior'
                f' is off. Appending node seqs below:\n {node.seqs}'
            )
        node.avg_edit = a / (a + b)
        node.var_edit = (a * b) / (((a + b)**2) * (a + b + 1))

    def bh_fdr(
        self,
        forest: MotifForest,
        q: float = 0.05
    ) -> None:
        # Benjamini-Hochberg false discovery rate control
        nodes = sorted(self.forest.flatten(), key=lambda nd: nd.node_id)
        if not nodes:
            return

        pvals = np.asarray([node.p for node in nodes], dtype=float)
        m = pvals.size

        order = np.argsort(pvals)
        sorted_p = pvals[order]

        thresholds = q * np.arange(1, m+1) / m
        below = sorted_p <= thresholds

        if not np.any(below):
            # nothing survives; prune everything
            for node in nodes:
                self.forest.prune(node)
            return

        k = np.max(np.where(below)[0])
        cutoff = sorted_p[k]
        print(f'FDR p-value cutoff: {cutoff}')

        for node, p in zip(nodes, pvals):
            if p > cutoff:
                self.forest.prune(node)

    def _binom_test(self, node: MotifNode) -> float:
        p0 = 0.25 ** node.motif_len
        k = node.seqs.shape[0]
        res = binomtest(k=k, n=self.forest.df_len, p=p0, alternative='greater')
        return res.pvalue # might want to expose rest of res later

    @staticmethod
    def _fit_beta_binom(
        node: MotifNode,
        alpha0: float = 1.0,
        beta0: float = 1.0
    ):
        k = np.asarray(node.seqs['k'])
        n = np.asarray(node.seqs['n'])

        def _neg_loglik(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e9
            ll = betabinom.logpmf(k, n, a, b)
            if not np.all(np.isfinite(ll)):
                return 1e9
            return -np.sum(ll)

        res = minimize(
            _neg_loglik,
            x0=np.array([alpha0, beta0]),
            method="L-BFGS-B",
            bounds=[(1e-6,None), (1e-6,None)]
        )

        alpha_hat, beta_hat = res.x
        return alpha_hat, beta_hat, res

