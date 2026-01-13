from __future__ import annotations

import pandas as pd
import numpy as np
import re
from typing import Optional, Literal, Callable, Any
from collections.abc import Iterable
from scipy.stats import binomtest
from tqdm.auto import tqdm

import networkx as nx
from pyvis.network import Network
from matplotlib import cm
from matplotlib import colors as mcolors
from networkx.drawing.nx_agraph import graphviz_layout

from emerge_data import EmergeHandler, MotifNode, MotifEdge

class MotifForest(EmergeHandler):
    def __init__(
        self,
        roots: list[MotifNode],
        df_emerge: Optional[pd.DataFrame],
        *,
        seq_col: str = '5to3',
        n_col: str = 'n',
        k_col: str = 'k',
        to_rna: bool = True,
        lazy_canopy: bool = True
    ) -> None:
        if df_emerge is not None:
            super().__init__(
                df_emerge = df_emerge,
                seq_col = seq_col,
                n_col = n_col,
                k_col = k_col,
                to_rna = to_rna
            )
        if not roots:
            raise ValueError('arg `roots` cannot be None')

        self.roots = roots
        self.canopy = None
        if not lazy_canopy:
            self.canopy = ForestCanopy(self)

        for node in roots:
            if not isinstance(node, MotifNode):
                raise TypeError(
                    'arg `forest` must be composed of MotifNode instances. '
                    f'found: {type(node)}'
                )

    def __iter__(self) -> Iterator[MotifNode]:
        return iter(self.flatten())

    def iter_edges(self, with_canopy=True) -> Iterator[MotifEdge]:
        for child in self.flatten(with_canopy=with_canopy):
            for parent in getattr(child, 'parents', []) or []:
                yield MotifEdge(parent=parent, child=child)

    def with_canopy(self) -> MotifForest:
        self.canopy = ForestCanopy(self)
        return self

    def traverse(
        self,
        visited: Optional[set] = None,
        func: Optional[Callable[..., Any]] = None,
        params: Optional[list[Any]] = None,
        with_canopy: bool = True
    ) -> list:
        if self.roots is None:
            raise ValueError('arg `roots` cannot be None')
        if visited is None:
            visited = set()
        if params is None:
            params = []

        results: list[Any] = []
        nodes = self.flatten(with_canopy=with_canopy)

        desc = f'Traversing with {func.__name__}' if func else f'Traversing'
        for node in tqdm(nodes, desc=desc, unit='node'):
            if func is not None:
                results.append(func(node, *params))

        return results

    def flatten(self, with_canopy: bool = True) -> list[MotifNode]:
        if self.roots is None:
            raise ValueError('arg `roots` cannot be None')

        out: list[MotifNode] = []
        seen: set[int] = set()
        stack = list(self.roots)

        while stack:
            node = stack.pop()
            nid = id(node)
            if nid in seen:
                continue
            seen.add(nid)
            out.append(node)
            stack.extend(getattr(node, 'children', []))

        if with_canopy:
            if self.canopy is None:
                self.canopy = ForestCanopy(self)
            for node in self.canopy:
                nid = id(node)
                if nid not in seen:
                    seen.add(nid)
                    out.append(node)

        return out

    def prune(self, node: MotifNode, with_canopy: bool = True) -> None:
        if not self.roots:
            raise ValueError('arg `roots` cannot be None')

        nodes = self.flatten(with_canopy=with_canopy)
        parents = list(node.parents)
        children = list(node.children)

        if not parents and node in nodes:
            i = self.roots.index(node)
            self.roots[i:i+1] = node.children
            return

        for parent in parents:
            new_children = []
            for child in parent.children:
                if child is node:
                    new_children.extend(children)
                else:
                    new_children.append(child)
            parent.children = new_children

        for child in node.children:
            new_parents = [p for p in child.parents if p is not node]
            for parent in parents:
                if parent not in new_parents:
                    new_parents.append(parent)
            child.parents = new_parents

        node.children = []
        node.parents = []

    def to_html(
        self,
        outfile: str = 'forest.html',
        title: str | None = None,
        subtitle: str | None = None,
        *,
        color_by: Literal['editing', 'motif_status'] = 'editing',
        status_attr: str = 'motif_state',
        with_canopy: bool = True,
        open_browser: bool = True,
        prog: str = 'dot',
    ) -> None:

        nodes = self.flatten(with_canopy=with_canopy)
        print("nodes:", len(nodes))
        print("max parents:", max(len(n.parents) for n in nodes))
        print("max children:", max(len(n.children) for n in nodes))
        print("parents>=5:", sum(len(n.parents) >= 5 for n in nodes))
        print("children>=5:", sum(len(n.children) >= 5 for n in nodes))

        seqs = [n.motif_seq for n in nodes if n.motif_seq is not None]
        print("motif_seq collisions:", len(seqs) - len(set(seqs)))


        if not self.roots:
            raise ValueError('arg `forest` cannot be None/empty.')

        nodes = self.flatten(with_canopy=with_canopy)
        nodes = [nd for nd in nodes if nd.motif_seq is not None]
        if not nodes:
            raise ValueError('no motifs to render after filtering for seq')

        # de-duplicate nodes
        all_nodes: dict[str, MotifNode] = {}
        for nd in nodes:
            key = nd.motif_seq
            if key in all_nodes and all_nodes[key] is not nd:
                continue
            all_nodes[key] = nd

        G = nx.DiGraph()
        G.add_nodes_from(all_nodes.keys())

        for nd in all_nodes.values():
            for child in getattr(nd, 'children', []):
                cseq = getattr(child, 'motif_seq', None)
                if cseq in all_nodes:
                    G.add_edge(nd.motif_seq, cseq)

        # layout: graphviz when possible; otherwise fallback
        try:
            if G.number_of_edges() > 0:
                pos = graphviz_layout(G, prog=prog)
            else:
                pos = nx.spring_layout(G, seed=0)
        except Exception:
            pos = nx.spring_layout(G, seed=0)


        xs = np.array([p[0] for p in pos.values()], dtype=float)
        ys = np.array([p[1] for p in pos.values()], dtype=float)

        xs -= xs.mean() if xs.size else 0.0
        ys -= ys.mean() if ys.size else 0.0

        scale = max(
            xs.std()
            if xs.size else 1.0,
            ys.std()
            if ys.size else 1.0, 1e-9
        )
        xs = xs / scale * 800
        ys = ys / scale * 800

        pos = {k: (float(x), float(y)) for k, x, y in zip(pos.keys(), xs, ys)}

        # set up coloring
        norm = None
        cmap = None
        status_palette = {
            0: '#bdbdbd',  # pruned/none (if it slips in)
            1: '#6baed6',  # context-ish
            2: '#31a354',  # true-ish
        }

        if color_by == 'editing':
            edit_vals = [
                getattr(nd, 'edit', None)
                for nd in all_nodes.values()
                if getattr(nd, 'edit', None) is not None
                and np.isfinite(getattr(nd, 'edit', None))
            ]
            if edit_vals:
                vals = np.asarray(edit_vals, dtype=float)
                vmin = np.percentile(vals, 10)
                vmax = np.percentile(vals, 95)
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                    cmap = cm.get_cmap('Reds')

        # sizes
        prevs = np.array([
            float(getattr(nd, 'prevalence', 1) or 1)
            for nd in all_nodes.values()
            ], dtype=float
        )
        prevs[~np.isfinite(prevs)] = 1.0
        prevs[prevs <= 0] = 1.0

        lo, hi = np.percentile(prevs, [5, 95]) if prevs.size else (1.0, 1.0)
        sizes_clipped = np.clip(prevs, lo, hi)
        sizes_scaled = np.sqrt(sizes_clipped)
        denom = (sizes_scaled.max() - sizes_scaled.min()) + 1e-9
        node_sizes = 10 + 40 * (sizes_scaled - sizes_scaled.min()) / denom
        size_map = dict(zip(all_nodes.keys(), node_sizes))

        # pyvis render
        net = Network(
            height='750px', width='100%', directed=True, notebook=False
        )
        net.toggle_physics(False)
        net.set_edge_smooth('straight')

        # add nodes with color
        for seq, nd in all_nodes.items():
            p_raw = getattr(nd, 'pval', None)
            p_str = (
                f'{p_raw:.4f}'
                if isinstance(p_raw, (int, float, np.integer, np.floating))
                and np.isfinite(p_raw)
                else '–'
            )

            avg = getattr(nd, 'edit', None)
            prev = getattr(nd, 'prevalence', None)
            prev = (
                int(prev)
                if isinstance(prev, (int, np.integer))
                and prev > 0
                else 1
            )

            status_val = getattr(nd, status_attr, None)

            if color_by == 'motif_status':
                color = (status_palette.get(
                    int(status_val)
                    if status_val is not None
                    else 0, '#bdbdbd'
                ))
            else:
                color = '#cccccc'
                if (
                    cmap is not None
                    and norm is not None
                    and avg is not None
                    and np.isfinite(avg)
                ):
                    color = mcolors.to_hex(cmap(norm(float(avg))))

            edit_str = (
                f'{avg:.3f}'
                if avg is not None
                and np.isfinite(avg)
                else '–'
            )
            status_str = status_val if status_val is not None else "–"

            node_title = (
                f'Seq: {seq}, '
                f'Count: {prev}, '
                f'Edit avg: {edit_str}, '
                f'p-val: {p_str}, '
                f'{status_attr}: {status_str}'
            )

            net.add_node(
                seq,
                label=seq,
                title=node_title,
                color=color,
                size=float(size_map.get(seq, 10.0)),
            )

        # add edges (may be none; that's fine)
        for u, v in G.edges():
            net.add_edge(u, v)

        # pin positions
        node_index = {node['id']: idx for idx, node in enumerate(net.nodes)}
        for seq, (x, y) in pos.items():
            if seq in node_index:
                idx = node_index[seq]
                net.nodes[idx]['x'] = float(x)
                net.nodes[idx]['y'] = float(-y)

        net.write_html(outfile, open_browser=open_browser)

        # optional header injection
        if title or subtitle:
            with open(outfile, 'r', encoding='utf-8') as f:
                html = f.read()

            section = "<div style='text-align:center; margin-bottom:20px;'>"
            if title:
                section += f"<h2 style='margin:0;'>{title}</h2>"
            if subtitle:
                section += (
                    f"<p style='margin:0; font-size:14px; "
                    f"color:#444;'>{subtitle}</p>"
                )
            section += "</div>"

            html = html.replace('<body>', '<body>' + section, 1)

            with open(outfile, 'w', encoding='utf-8') as f:
                f.write(html)

class ForestCanopy:
    def __init__(
        self,
        forest: MotifForest,
        offset: int = 2
    ):
        if not isinstance(forest, MotifForest):
            raise TypeError(
                'arg `forest` must be a MotifForest. '
                f'got: {type(forest)}'
            )
        self.forest = forest
        self.offset = offset # rename. this is just how much shorter than
                             # seq_len candidate_seq must be in self.generate()
        self.canopy = None
        self.generate()

    def __iter__(self):
        if not self.canopy:
            self.generate()
        return iter(self.canopy)

    def generate(self) -> None:
        if not self.canopy:
            self.canopy = []

        nodes = self.forest.flatten(with_canopy=False)
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
                if len(candidate_seq) / 2 > self.forest.seq_len - self.offset:
                    continue

                new_node = MotifNode(
                    motif_seq = candidate_seq,
                    node_id = max(node_ids) + 1
                )
                new_node.seqs = self.forest.get_motif_seqs(new_node.motif_seq)
                if new_node.seqs is None or new_node.seqs.empty:
                    continue

                new_node.p = self._canopy_bino_test(
                    node1=this_node,
                    node2=that_node,
                    node12=new_node
                )

                new_node.add_parent(this_node)
                new_node.add_parent(that_node)

                self.canopy.append(new_node)

    def _compatible(self, token1: str, token2: str) -> bool:
        pos1 = self._token_to_pos(token=token1)
        pos2 = self._token_to_pos(token=token2)
        return self._compatible_by_pos(arr1=pos1, arr2=pos2)

    @staticmethod
    def _compatible_by_pos(arr1: np.ndarray, arr2: np.ndarray) -> bool:
        assert np.isin(arr1, [0, 1]).all(), 'arr1 must be 0/1'
        assert np.isin(arr2, [0, 1]).all(), 'arr2 must be 0/1'
        assert arr1.size == arr2.size, 'arrays must be same size'

        product = arr1 * arr2
        return product.sum() == 0

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

    #uh figure out what this does lol. before pushing, preferably
    @staticmethod
    def _canopy_bino_test(
        node1: MotifNode,
        node2: MotifNode,
        node12: MotifNode
    ) -> float:
        if getattr(node1, 'avg_edit', None) is None:
            node1.avg_edit = node1.seqs['mle'].mean()
        if getattr(node2, 'avg_edit', None) is None:
            node2.avg_edit = node2.seqs['mle'].mean()

        p0 = max([node1.avg_edit, node2.avg_edit])
        k = int(node12.seqs['k'].sum())
        n = int(node12.seqs['n'].sum())

        res = binomtest(k=k, n=n, p=p0, alternative='greater')

        return res.pvalue

