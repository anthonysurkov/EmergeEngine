from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
import re

import networkx as nx
from pyvis.network import Network
from matplotlib import cm
from matplotlib import colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout

from emerge_data import EmergeHandler, MotifNode

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
        self.canopy = None

    def __iter__(self):
        return iter(self.forest.flatten())

    def with_canopy(self):
        self.canopy = ForestCanopy(self)
        return self

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

    def flatten(self, with_canopy: bool = True) -> set[MotifNode]:
        visited: set[MotifNode] = set()
        self.traverse(visited=visited, func=None)
        if with_canopy and self.canopy is not None:
            return visited | set(self.canopy)
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

class ForestCanopy:
    def __init__(
        self,
        forest: MotifForest
    ):
        if not isinstance(forest, MotifForest):
            raise TypeError(
                'arg `forest` must be a MotifForest. '
                f'got: {type(forest)}'
            )
        self.forest = forest
        self.canopy = []
        self.generate()

    def __iter__(self):
        return iter(self.canopy)

    def generate(self) -> None:
        nodes = self.forest.flatten()
        node_ids = [node.node_id for node in nodes]

        while nodes:
            this_node = nodes.pop()
            for that_node in nodes:
                if not self._compatible(
                    this_node.motif_seq,
                    that_node.motif_seq
                ):
                    continue

                candidate_seq = this_node.motif_seq + that_node.motif_seq
                if not self._token_useful(candidate_seq):
                    continue

                new_node = MotifNode(
                    motif_seq = candidate_seq,
                    node_id = max(node_ids) + 1
                )
                new_node.seqs = self.forest.get_motif_seqs(new_node.motif_seq)

                new_node.add_parent(this_node)
                new_node.add_parent(that_node)

                self.canopy.append(new_node)

    def _compatible(self, token1: str, token2: str) -> bool:
        pos1 = self._token_to_pos(token=token1)
        pos2 = self._token_to_pos(token=token2)
        return self._compatible_by_pos(arr1=pos1, arr2=pos2)

    def _token_to_pos(self, token: str) -> np.ndarray:
        assert isinstance(token, str), 'token must be str'

        pos_nums = [int(m) for m in re.findall(r'\d+', token)]
        if not pos_nums:
            return np.array([], dtype=int)

        pos_set = set(pos_nums)
        return np.array(
            [1 if i in pos_set else 0 for i in range(0, self.forest.seq_len+1)],
             dtype=int
        )

    @staticmethod
    def _compatible_by_pos(arr1: np.ndarray, arr2: np.ndarray) -> bool:
        assert np.isin(arr1, [0, 1]).all(), 'arr1 must be 0/1'
        assert np.isin(arr2, [0, 1]).all(), 'arr2 must be 0/1'
        assert arr1.size == arr2.size, 'arrays must be same size'

        product = arr1 * arr2
        return product.sum() == 0

    # TODO: unfuck this
    def _token_useful(self, token: str, offset: int = 2) -> bool:
        if len(token) / 2 > self.forest.seq_len - offset:
            return False
        return True

