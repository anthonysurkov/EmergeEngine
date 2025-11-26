#emerge_data.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any
import functools
import operator
import warnings

import pandas as pd
from pandas.api.types import is_bool_dtype
import numpy as np
import numbers
import re

import networkx as nx
from pyvis.network import Network
from matplotlib import cm
from matplotlib import colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout

class EmergePredicate:
    def __init__(
        self,
        func: Callable[[pd.DataFrame], pd.Series],
        name: Optional[str] = None
    ) -> None:
        self._func = func
        self.name = name or getattr(func, "__name__", "predicate")

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        mask = self._func(df)

        if not isinstance(mask, pd.Series):
            raise TypeError(
                f'predicate {self.name!r} must return a pandas Series, '
                f'got {type(mask)}'
            )
        if mask is None:
            raise ValueError(
                f'boolean mask is empty; cannot be'
            )
        if not is_bool_dtype(mask.dtype):
            raise TypeError(
                f'predicate {self.name!r} must return a boolean Series, '
                f'got dtype {mask.dtype}'
            )
        if not mask.index.equals(df.index):
            raise ValueError(
                f'predicate {self.name!r} returned a Series with a different '
                f'index than the input DataFrame'
            )

        return mask

    def __and__(self, other: EmergePredicate) -> EmergePredicate:
        if not isinstance(other, EmergePredicate):
            return NotImplemented

        def _and(df: pd.DataFrame) -> pd.Series:
            return self(df) & other(df)

        return EmergePredicate(_and, name=f'({self} & {other})')

    def __or__(self, other: EmergePredicate) -> EmergePredicate:
        if not isinstance(other, EmergePredicate):
            return NotImplemented

        def _or(df: pd.DataFrame) -> pd.Series:
            return self(df) | other(df)

        return EmergePredicate(_or, name=f'({self} | {other})')

    def __invert__(self) -> EmergePredicate:
        def _not(df: pd.DataFrame) -> pd.Series:
            return ~self(df)

        return EmergePredicate(_not, name=f'(~{self})')

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f'EmergePredicate(name={self.name!r})'

    @staticmethod
    def col_geq(col: str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] >= value
        return EmergePredicate(_pred, name=f'{col} >= {value}')

    @staticmethod
    def col_leq(col: str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] <= value
        return EmergePredicate(_pred, name=f'{col} <= {value}')

    @staticmethod
    def col_gt(col: str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] > value
        return EmergePredicate(_pred, name=f'{col} > {value}')

    @staticmethod
    def col_lt(col: str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] < value
        return EmergePredicate(_pred, name=f'{col} < {value}')

    @staticmethod
    def col_eq(col:str, value: float) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col] == value
        return EmergePredicate(_pred, name=f'{col} == {value}')

    @staticmethod
    def str_contains(
        col: str,
        pattern: str,
        regex: bool = True
    ) -> EmergePredicate:
        def _pred(df: pd.DataFrame) -> pd.Series:
            return df[col].astype('string').str.contains(pattern, regex=regex)
        return EmergePredicate(_pred, name=f'{col}.contains({pattern!r})')

class EmergeHandler:
    def __init__(
        self,
        df_emerge: pd.DataFrame,
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
                '`df_emerge` cannot contain NaN values` in neither `seq_col`'
                ' nor `n_col` nor `k_col`.'
            )
        if not all(isinstance(x, str) for x in df_emerge[seq_col]):
            raise TypeError('Every element of `seq_col` must be a string')
        if not all(isinstance(x, numbers.Integral) for x in df_emerge[k_col]):
            raise TypeError('Every element of `k_col` must be an int')
        if not all(isinstance(x, numbers.Integral) for x in df_emerge[n_col]):
            raise TypeError('Every element of `n_col` must be an int')

        seq_len = len(df_emerge[seq_col].iloc[0])
        if df_emerge[seq_col].str.len().nunique() != 1:
            raise ValueError(
                'Every element of `seq_col` must have the same length'
            )
        if seq_len == 0:
            raise ValueError('sequences cannot be empty (seq_len == 0)')

        df_emerge[seq_col] = df_emerge[seq_col].str.upper()
        if to_rna:
            df_emerge[seq_col] = (
                df_emerge[seq_col].str.replace('T','U', regex=False)
            )

        cols = [seq_col, n_col, k_col]
        self.df = df_emerge[cols]
        self.df = self.df.rename(
            columns={seq_col: '5to3', n_col: 'n', k_col: 'k'}
        )
        self.df['mle'] = self.df['k'] / self.df['n']
        self.seq_len = seq_len
        self.df_len = self.df.shape[0]

    # TODO: provide motif spec option that isn't A0C1G2 etc.
    # want something like ACTG0-3. also support for gappy motifs, e.g.
    # ACTGXXXGT0-8. serves as a nice internal identifier for motifs too
    # specify with bpe_syntax=False in args
    def contains_motif(
        self,
        motif: str
    ) -> EmergePredicate:
        mapping = self._parse_token(motif)
        mask = self._token_to_mask(mapping)

        def _match(seq) -> bool:
            #if seq is None or pd.isna(seq):
            #    return False
            #seq = str(seq)
            return self._matches_mask(seq, mask)

        def _pred(df: pd.DataFrame) -> pd.Series:
            return df['5to3'].apply(_match)

        return EmergePredicate(_pred, name=f'5to3.contains({motif!r})')

    def _token_to_mask(self, mapping: dict[int, str]) -> str:
        mask = ["X"] * self.seq_len
        for pos, base in mapping.items():
            mask[pos] = base
        return "".join(mask)

    @staticmethod
    def _parse_token(seq: str) -> dict[int, str]:
        token_re = re.compile(r'([ACGU])(\d+)')
        return {int(pos): base for base, pos in token_re.findall(seq)}

    @staticmethod
    def _matches_mask(seq: str, mask: str) -> bool:
        return all(m == "X" or s == m for s, m in zip(seq, mask))

    def get_motif_seqs(
        self,
        motif: str
    ) -> pd.DataFrame:
        request = self.contains_motif(motif)
        mask = request(self.df)
        return self.query(request)

    def query(
        self,
        *predicates: EmergePredicate,
        columns: Optional[Iterable[str]] = None,
        copy: bool = True
    ) -> pd.DataFrame:
        df = self.df

        combined = self._combine_predicates(predicates)
        if combined is not None:
            mask = combined(df)
            df = df[mask]

        if columns is None:
            columns = ['5to3','n','k','mle']
        df = df.loc[:, list(columns)]

        return df.copy(deep=True) if copy else df

    def _combine_predicates(
        self,
        predicates: Sequence[EmergePredicate]
    ) -> Optional[EmergePredicate]:
        if not predicates:
            return None
        return functools.reduce(operator.and_, predicates)


@dataclass(eq=False)
class MotifNode:
    motif_seq: Optional[str] = None
    seqs: Optional[pd.DataFrame] = None

    node_id: Optional[int] = None
    rank: Optional[int] = None
    prevalence: Optional[int] = None
    avg_edit: Optional[float] = None
    var_edit: Optional[float] = None

    parents: list[MotifNode] = field(default_factory=list)
    children: list[MotifNode] = field(default_factory=list)

    def __post_init__(self):
        if self.motif_seq is None:
            self.base_chars = []
            self.motif_len = 0
            return
        self.base_chars = sorted(
            {ch for ch in self.motif_seq if not ch.isdigit()}
        )
        self.motif_len = len(
            [ch for ch in self.motif_seq if ch in self.base_chars]
        )

    def add_child(self, child: MotifNode) -> None:
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    def add_parent(self, parent: MotifNode) -> None:
        if parent not in self.parents:
            self.parents.append(parent)
        if self not in parent.children:
            parent.children.append(self)

class MotifForest(EmergeHandler):
    def __init__(
        self,
        forest: list[MotifNode],
        df_emerge: Optional[pd.DataFrame],
        seq_col: str = '5to3',
        n_col: str = 'n',
        k_col: str = 'k',
        to_rna: bool = True
    ) -> None:
        if df_emerge is not None:
            super().__init__(
                df_emerge = df_emerge,
                seq_col = seq_col,
                n_col = n_col,
                k_col = k_col,
                to_rna = to_rna
            )
        if not forest:
            raise TypeError(
                'arg `forest` cannot neither be an empty list nor None.'
            )
        for node in forest:
            if not isinstance(node, MotifNode):
                raise TypeError(
                    'arg `forest` must be composed of MotifNode instances. '
                    f'found: {type(node)}'
                )
        self.forest = forest

    def traverse(
        self,
        visited: Optional[set] = None,
        func: Optional[Callable[..., Any]] = None,
        params: Optional[list[Any]] = None
    ) -> list:
        if self.forest is None:
            raise ValueError(
                'arg `forest` cannot be None. try using build_forest method'
            )
        if visited is None:
            visited = set()
        if params is None:
            params = []

        results: list[Any] = []

        def _visit(node: MotifNode) -> None:
            if node in visited:
                return
            visited.add(node)
            if func is not None:
                result = func(node, *params)
                results.append(result)
            for child in node.children:
                _visit(child)

        for node in self.forest:
            _visit(node)

        return results

    def flatten(self) -> set[MotifNode]:
        visited: set[MotifNode] = set()
        self.traverse(visited=visited, func=None)
        return visited

    def prune(self, node: MotifNode) -> None:
        if not self.forest:
            raise ValueError(
                'arg `forest` cannot be None. Try giving me a forest'
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
            nd.avg_edit for nd in all_nodes.values()
            if getattr(nd, "avg_edit", None) is not None
        ]

        if edit_vals:
            vals = np.asarray(edit_vals, dtype=float)

            vmin = np.percentile(vals, 10)
            vmax = np.percentile(vals, 95)

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
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

            avg = getattr(nd, "avg_edit", None)
            var = getattr(nd, "var_edit", None)
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

