#emerge_phenotype.py
from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
from scipy.stats import binomtest, betabinom
from scipy.optimize import minimize
from emerge_data import MotifForest, MotifNode

class ForestPhenotype(MotifForest):
    def __init__(
        self,
        forest: list[MotifNode],
        df_emerge: pd.DataFrame,
        seq_col: str = '5to3',
        n_col: str = 'n',
        k_col: str = 'k',
        to_rna: bool = True
    ):
        super().__init__(
            forest = forest,
            df_emerge = df_emerge,
            seq_col = seq_col,
            n_col = n_col,
            k_col = k_col,
            to_rna = to_rna
        )

    def prune_by_enrichment(
        self,
        q: float = 0.05
    ) -> None:
        def _compute_p(node: MotifNode):
            node.p = self._binom_test(node=node)

        self.traverse(func=_compute_p)
        self._bh_fdr(q=q)

    def append_edits(self):
        self.traverse(function = self.append_node_editing)

    def append_node_editing(self, node: MotifNode) -> None:
        J = node.seqs['mle'].shape[0]
        if J == 0:
            node.avg_edit = np.nan
            node.var_edit = np.nan
            return
        if J == 1:
            node.avg_edit = node.seqs['mle'].iloc[0]
            node.var_edit = np.nan
            return
        node.avg_edit = node.seqs['mle'].sum() / J
        node.var_edit = (
            (1 / (J - 1))
            * ((node.seqs['mle'] - node.avg_edit)**2).sum()
        )

    def append_edits_bb(
        self,
        node: MotifNode,
        correct_if_bb_bad: bool = True
    ) -> None:
        a, b, res = self._fit_beta_binom(node)
        if not res.success:
            warnings.warn(
                f'Beta-binomial fit did not converge for node ID {node.node_id}'
                f' / motif {node.motif_seq}. Appending node seqs below:'
                f'{node.seqs}'
            )
            if correct_if_bb_bad:
                self.append_node_editing(node)
                return
        node.avg_edit = a / (a + b)
        node.var_edit = (a * b) / (((a + b)**2) * (a + b + 1))

    def _bh_fdr(
        self,
        q: float = 0.05
    ) -> None:
        # Benjamini-Hochberg false discovery rate control
        nodes = list(self.flatten())
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
                self.prune(node)
            return

        k = np.max(np.where(below)[0])
        cutoff = sorted_p[k]
        print(f'FDR p-value cutoff: {cutoff}')

        for node, p in zip(nodes, pvals):
            if p > cutoff:
                self.prune(node)

    def _binom_test(self, node: MotifNode) -> float:
        p0 = 0.25 ** node.motif_len
        k = node.seqs.shape[0]
        res = binomtest(k=k, n=self.df_len, p=p0, alternative='greater')
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

