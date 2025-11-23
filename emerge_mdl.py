from __future__ import annotations
from dataclasses import dataclass, field
import warnings

import pandas as pd
import numpy as np
from scipy.stats import binomtest, betabinom
from scipy.optimize import minimize
from collections import Counter, deque, defaultdict
from collections.abc import Callable
from typing import Any
import re

import networkx as nx
from pyvis.network import Network
from matplotlib import cm
from matplotlib import colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout

# Helper functions (not public rn so not completely type-annotated)
def fit_beta_binom(
    n: pd.Series,
    k: pd.Series,
    alpha0: float = 1.0,
    beta0: float = 1.0
):
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)

    def neg_loglik(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e9
        ll = betabinom.logpmf(k, n, a, b)
        if not np.all(np.isfinite(ll)):
            return 1e9
        return -np.sum(ll)

    res = minimize(
        neg_loglik,
        x0=np.array([alpha0, beta0]),
        method="L-BFGS-B",
        bounds=[(1e-6,None), (1e-6,None)]
    )

    alpha_hat, beta_hat = res.x
    return alpha_hat, beta_hat, res

def enrichment_test(n, k, p0=0.25):
    """
    k  = number of times the motif/base was observed
    n  = total number of sequences in corpus
    p0 = null probability under 'no enrichment'
    """
    res = binomtest(k, n, p=p0, alternative='greater')
    return res.statistic, res.pvalue

def non_enriched(n, node):
    length = len(node.motif_seq) / 2
    p0 = 0.25 ** length
    k = node.seqs.shape[0]
    stat, p = enrichment_test(n=n, k=k, p0=p0) # n = corpus.shape[0]
    node.p = p
    return p

def find_freq_pair(
    token_ids: list[int],
    vocab: dict[int, str],
    mode: str = 'most'
) -> tuple[int, int]:
    pairs = Counter(zip(token_ids, token_ids[1:]))
    if vocab is not None:
        pairs = {
            p: c for p, c in pairs.items()
            if 'E' not in vocab[p[0]] and 'E' not in vocab[p[1]]
        }
    if mode == 'most':
        return max(pairs.items(), key = lambda x: x[1])[0]
    elif mode == 'least':
        return min(pairs.items(), key = lambda x: x[1])[0]
    else:
        raise ValueError('invalid mode (`most` or `least`)')

def replace_pair(
    token_ids: list[int],
    pair_id: tuple[int, int],
    new_id: int
):
    dq = deque(token_ids)
    replaced = []
    while dq:
        current = dq.popleft()
        if dq and (current, dq[0]) == pair_id:
            replaced.append(new_id)
            dq.popleft()
        else:
            replaced.append(current)
    return replaced

def count_kmers(
    seqs: list[str],
    kmax: int
):
    counts = Counter()
    for seq in seqs:
        tokens = [f"{char}{i}" for i, char in enumerate(seq)]
        for k in range(2, kmax + 1):
            for i in range(len(tokens) - k + 1):
                kmer = "".join(tokens[i:i+k])
                counts[kmer] += 1
    return counts

token_re = re.compile(r'([ACGU])(\d+)')
def parse_token(seq: str) -> dict[int, str]:
    return {int(pos): base for base, pos in token_re.findall(seq)}

def token_to_mask(mapping: dict[int, str], L: int) -> str:
    mask = ["X"] * L
    for pos, base in mapping.items():
        mask[pos] = base
    return "".join(mask)

def matches_mask(seq: str, mask: str) -> bool:
    return all(m == "X" or s == m for s, m in zip(seq, mask))

def find_matching_seqs(
    node: MotifNode,
    df: pd.DataFrame,
    seq_col: str = 'seq'
):
    token = node.motif_seq
    mapping = parse_token(token)
    L = len(df[seq_col].iloc[0])
    mask = token_to_mask(mapping, L)
    def match(s):
        s = s.replace('T', 'U')
        return matches_mask(s, mask)
    sub = df[df[seq_col].apply(match)]
    return sub

# public
class MotifNode:
    def __init__(
        self,
        node_id: int | None = None,
        motif_seq: str | None = None,
        seqs: pd.DataFrame = None,
        rank: int | None = None,
        prevalence: int | None = None,
        avg: float | None = None,
        var: float | None = None,
        parents: list[MotifNode] | None = None,
    ):
        self.node_id = node_id
        self.motif_seq = motif_seq
        self.seqs = seqs
        self.rank = rank
        self.prevalence = prevalence

        self.parents = parents or []
        self.children = []

        self.avg = avg
        self.var = var
        self.p = None # used for enrichment hypothesis testing

    # if you modify what equality between motifs means, please also add a
    # __hash__ to EmergeTokenizer to avoid a nasty-to-find unhashable type
    # error :)

    def add_child(self, child: MotifNode):
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)
        return self

    def add_parent(self, parent: MotifNode):
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)
        return self

class EmergeMDL:
    def __init__(
        self,
        df_emerge: pd.DataFrame,
        kmax: int = 6,
        seq_col: str = '5to3',
        n_col: str = 'n',
        k_col: str = 'k',
        to_rna: bool = True
    ):
        if not isinstance(df_emerge, pd.DataFrame):
            raise TypeError(
                'arg `df_emerge` must be a pandas DataFrame. '
                f'is: {type(df_emerge)}'
            )
        df_emerge = df_emerge.copy(deep=True)

        if not isinstance(seq_col, str):
            raise TypeError(
                'arg `seq_col` must be a string'
                f'is: {type(seq_col)}'
            )
        if not isinstance(n_col, str):
            raise TypeError(
                'arg `n_col` must be a string'
                f'is: {type(n_col)}'
            )
        if not isinstance(k_col, str):
            raise TypeError(
                'arg `k_col` must be a string'
                f'is: {type(k_col)}'
            )
        if df_emerge.shape[0] == 0:
            raise ValueError(
                '`df_emerge` must have more than one entry associated with it'
            )
        elif seq_col not in df_emerge.columns:
            raise KeyError('arg `seq_col` is not a valid col in `df_emerge`')
        elif k_col not in df_emerge.columns:
            raise KeyError('arg `k_col` is not a valid col in `df_emerge`')
        elif n_col not in df_emerge.columns:
            raise KeyError('arg `n_col` is not a valid col in `df_emerge`')
        if df_emerge[[seq_col, n_col, k_col]].isnull().any().any():
            raise ValueError(
                '`df_emerge` cannot contain NaN values` in neither `edit_col`'
                ' nor `seq_col`.'
            )
        if not all(isinstance(x, str) for x in df_emerge[seq_col]):
            raise TypeError('Every element of `seq_col` must be a string')
        if not all(isinstance(x, int) for x in df_emerge[k_col]):
            raise TypeError('Every element of `k_col` must be an int')
        if not all(isinstance(x, int) for x in df_emerge[n_col]):
            raise TypeError('Every element of `n_col` must be an int')

        # configure seq_len
        seq_len = len(df_emerge[seq_col].iloc[0])
        if df_emerge[seq_col].str.len().nunique() != 1:
            raise ValueError(
                'Every element of `seq_col` must have the same length'
            )
        if seq_len == 0:
            raise ValueError('sequences cannot be empty (seq_len == 0)')

        # convert to RNA sequences
        df_emerge[seq_col] = df_emerge[seq_col].str.upper()
        if to_rna:
            df_emerge[seq_col] = (
                df_emerge[seq_col].str.replace('T','U', regex=False)
            )

        # configure alphabet
        alphabet = set(''.join(df_emerge[seq_col].astype(str)))
        unexpected = alphabet - set(['A','G','C','U'])
        if unexpected:
            warnings.warn(
                'sequences contain chars outside canonical RNA nucleotides: '
                f'{unexpected}',
                UserWarning
            )
        alphabet = [
            f'{letter}{i}' for letter in alphabet for i in range(0, seq_len)
        ]

        # check kmax
        if not isinstance(kmax, int):
            raise TypeError('arg `kmax` must be an integer greater than 1`')
        if kmax <= 2:
            raise ValueError('arg `kmax` must be an integer greater than 1`')
        if kmax > seq_len:
            raise ValueError('arg `kmax` must not exceed RNA length')

        cols = [seq_col, n_col, k_col]
        self.df = df_emerge[cols]
        self.df = self.df.rename(
            columns={seq_col: 'seq', n_col: 'n', k_col: 'k'}
        )
        self.df['mle'] = self.df['k'] / self.df['n']

        self.kmax = kmax
        self.seq_len = seq_len
        self.vocab = {i: char for i, char in enumerate(alphabet)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        self.merges = {}
        self.global_ranks = {}
        self.kmer_ranks = {}
        self.kmer_bins = defaultdict(list)
        self.counts = Counter()
        self.forest = None

    def encode(
        self,
        vocab_size: int = 1000,
        special: set[str] | None = None
    ):
        if special is None:
            special = {'E'}
        if not isinstance(special, set):
            raise TypeError('arg `special` must be set of single-char strings')
        for x in special:
            if not isinstance(x, str) or len(x) != 1:
                raise ValueError('special token {x!r} must be a single char')
        special.add('E')

        if not isinstance(vocab_size, int):
            raise TypeError('arg `vocab_size` must be a positive integer')
        if vocab_size < 1:
            raise TypeError('arg `vocab_size` must be a positive integer')

        corpus = []
        for seq in self.df['seq']:
            for i, char in enumerate(seq):
                corpus.append(f'{char}{i}')
            corpus.append('E')

        if special:
            for token in special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        token_ids = [self.inverse_vocab[token] for token in corpus]

        global_rank = 0
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = find_freq_pair(token_ids, self.vocab, 'most')
            if pair_id is None:
                break
            p0, p1 = pair_id
            merged_token = self.vocab[p0] + self.vocab[p1]
            if len(merged_token) > 2 * self.kmax:
                continue

            token_ids = replace_pair(token_ids, pair_id, new_id)
            global_rank += 1
            self.merges[pair_id] = new_id
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
            self.global_ranks[(p0, p1)] = global_rank

            k = len(merged_token)
            self.kmer_bins[k].append(new_id)

        for k, ids in self.kmer_bins.items():
            for i, tid in enumerate(ids, start=1):
                self.kmer_ranks[tid] = i

        self.counts = count_kmers(self.df['seq'], self.kmax)
        if not self.merges:
            warnings.warn('no merges made in encode()')

    def build_forest(self):
        nodes: dict[int, MotifNode] = {}

        def get_node(node_id: int) -> MotifNode:
            if node_id not in nodes:
                nodes[node_id] = MotifNode(
                    node_id = node_id,
                    motif_seq = self.vocab[node_id]
                )
            return nodes[node_id]

        base_pat = re.compile(r'[ACGU]\d+$')
        for tid, tok in self.vocab.items():
            if base_pat.fullmatch(tok):
                get_node(tid)

        for (left_id, right_id), parent_id in self.merges.items():
            parent = get_node(parent_id)
            left = get_node(left_id)
            right = get_node(right_id)

            left.add_parent(parent)
            right.add_parent(parent)

            left.motif_seq = self.vocab[left_id]
            right.motif_seq = self.vocab[right_id]
            parent.motif_seq = self.vocab[parent_id]

            left.seqs = find_matching_seqs(left, self.df, seq_col = 'seq')
            right.seqs = find_matching_seqs(right, self.df, seq_col = 'seq')
            parent.seqs = find_matching_seqs(parent, self.df, seq_col = 'seq')

            parent.rank = self.kmer_ranks.get(parent_id)
            parent.prevalence = len(parent.seqs)
            left.prevalence = len(left.seqs)
            right.prevalence = len(right.seqs)

        all_children = {c for (l, r) in self.merges for c in (l, r)}
        roots = [
            get_node(pid)
            for (l, r), pid in self.merges.items()
            if pid not in all_children
        ]
        self.forest = roots

    def traverse(
        self,
        visited: set | None = None,
        function: Callable[[MotifNode], Any] | None = None
    ) -> list:
        if self.forest is None:
            raise ValueError(
                'arg `forest` cannot be None. try using build_forest method'
            )
        if visited is None:
            visited = set()

        results: list[Any] = []
        def _visit(node: MotifNode) -> None:
            if node in visited:
                return
            visited.add(node)
            if function is not None:
                result = function(node)
            for child in node.children:
                _visit(child)

        for node in self.forest:
            _visit(node)

        return results

    def flatten(self) -> set[MotifNode]:
        visited: set[MotifNode] = set()
        self.traverse(visited=visited, function=None)
        return visited

    def prune(self, node: MotifNode) -> None:
        if not self.forest:
            raise ValueError(
                'arg `forest` cannot be None. try using build_forest method'
            )
        if not node.parents and node in self.forest:
            idx = self.forest.index(node)
            self.forest[idx:idx+1] = node.children

        for parent in node.parents:
            new_children = []
            for child in parent.children:
                if child is node:
                    new_children.extend(node.children)
                else:
                    new_children.append(child)
            parent.children = new_children

        for child in node.children:
            child.parents = [p for p in node.parents]

        node.children = []
        node.parents = []

    # false discovery rate correction
    def bh_fdr(
        self,
        q: float = 0.05
    ) -> None:
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

    def prune_forest(self, q: float = 0.05) -> None:
        if not self.forest:
            raise ValueError(
                'arg `forest` cannot be None. try using build_forest method'
            )
        def _compute_p(node: MotifNode, alpha: float = 0.05) -> None:
            pval = non_enriched(n=self.df.shape[0], node=node) # helper
            node.p = pval

        self.traverse(function=_compute_p)
        self.bh_fdr(q=q)

    def summarize_editing(self, node: MotifNode) -> None:
        J = node.seqs['mle'].shape[0]
        node.avg = node.seqs['mle'].sum() / J
        node.var = (
            (1 / (J - 1))
            * ((node.seqs['mle'] - node.avg)**2).sum()
        )
        """
        df_k = node.seqs['k']
        df_n = node.seqs['n']
        a, b, res = fit_beta_binom(n=df_n, k=df_k)
        if not res.success:
            warnings.warn(
                f'Beta-binomial fit did not converge for node ID {node.node_id}'
                f' / motif {node.motif_seq}: {res.message}'
            )
            print(node.seqs)
        avg = a / (a + b)
        var = (a * b) / (((a + b)**2) * (a + b + 1))
        node.avg = avg
        node.var = var
        """

    def append_edits(self) -> None:
        if not self.forest:
            raise ValueError(
                f'arg `forest` cannot be None. try using build_forest method'
            )
        self.traverse(function=self.summarize_editing)

    def to_html(
        self,
        outfile: str = 'forest.html',
        title: str | None = None,
        subtitle: str | None = None,
        min_length: int = 0,
    ) -> None:
        if not self.forest:
            raise ValueError(
                f'arg `forest` cannot be None. try using build_forest method'
            )

        nodes = list(self.flatten())
        min_length *= 2
        nodes = [
            nd for nd in nodes
            if len(nd.motif_seq) > min_length
        ]

        if not nodes:
            raise ValueError(
                'No motifs to render (all were length-2 or pruned).'
            )

        all_nodes: dict[str, MotifNode] = {
            nd.motif_seq: nd for nd in nodes
        }

        G = nx.DiGraph()
        for nd in all_nodes.values():
            for child in nd.children:
                if len(child.motif_seq) == 2:
                    continue
                if child.motif_seq in all_nodes:
                    G.add_edge(child.motif_seq, nd.motif_seq)

        if len(G.edges()) == 0:
            raise ValueError('Empty graph after filtering motifs.')

        active_seqs = set(G.nodes())
        all_nodes = {
            seq: nd for seq, nd in all_nodes.items()
            if seq in active_seqs
        }
        if not all_nodes:
            raise ValueError('No connected motifs left after filtering.')

        edit_vals = [
            nd.avg for nd in all_nodes.values()
            if getattr(nd, "avg", None) is not None
        ]
        if edit_vals:
            norm = mcolors.Normalize(vmin=min(edit_vals), vmax=max(edit_vals))
            cmap = cm.get_cmap("Reds")
        else:
            norm = None
            cmap = None

        pos = graphviz_layout(G, prog="dot")

        net = Network(
            height="750px",
            width="100%",
            directed=True,
            notebook=False
        )
        net.toggle_physics(False)
        net.set_edge_smooth("straight")

        prevs = np.array([nd.prevalence for nd in all_nodes.values()], float)
        lo, hi = np.percentile(prevs, [5, 95])
        sizes_clipped = np.clip(prevs, lo, hi)
        sizes_scaled = np.sqrt(sizes_clipped)
        node_sizes = 10 + 40 * (sizes_scaled - sizes_scaled.min()) / (
            sizes_scaled.max() - sizes_scaled.min() + 1e-9
        )
        size_map = dict(zip(all_nodes.keys(), node_sizes))

        for seq, nd in all_nodes.items():
            p_raw = getattr(nd, "p", None)
            p_str = f"{p_raw:.4f}" if p_raw is not None else "–"

            avg = getattr(nd, "avg", None)
            var = getattr(nd, "var", None)
            prev = getattr(nd, "prevalence", None)
            if prev is None or prev <= 0:
                prev = 1

            rank = getattr(nd, "rank", None)
            rank_str = rank if rank is not None else "–"

            color = "#cccccc"
            if cmap is not None and avg is not None:
                rgba = cmap(norm(avg))
                color = mcolors.to_hex(rgba)

            edit_str = f"{avg:.3f}" if avg is not None else "–"
            stdev_str = f"{np.sqrt(var):.3f}" if var is not None else "–"

            node_title = (
                f"Seq: {seq}, "
                f"Rank: {rank_str}, "
                f"Count: {prev}, "
                f"Edit avg: {edit_str}, "
                f"Edit stdev: {stdev_str}, "
                f"p-value: {p_str}"
            )
            net.add_node(
                seq,
                label=seq,
                title=node_title,
                color=color,
                size=size_map[seq]
            )

        for u, v in G.edges():
            net.add_edge(u, v)

        node_index = {node["id"]: idx for idx, node in enumerate(net.nodes)}
        for seq, (x, y) in pos.items():
            if seq in node_index:
                idx = node_index[seq]
                net.nodes[idx]["x"] = x
                net.nodes[idx]["y"] = -y

        net.write_html(outfile, open_browser=True)

        if title or subtitle:
            with open(outfile, "r", encoding="utf-8") as f:
                html = f.read()

            section = "<div style='text-align:center; margin-bottom:20px;'>"
            if title:
                section += f"<h2 style='margin:0;'>{title}</h2>"
            if subtitle:
                section += (
                    "<p style='margin:0; font-size:14px; color:#444;'>{}</p>"
                    .format(subtitle)
                )
            section += "</div>"

            html = html.replace("<body>", "<body>" + section, 1)

            with open(outfile, "w", encoding="utf-8") as f:
                f.write(html)

