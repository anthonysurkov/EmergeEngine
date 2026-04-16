from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal, Callable, Any
from collections.abc import Iterable
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
        df_emerge: pd.DataFrame | None,
        *,
        seq_col: str = '5to3',
        n_col: str = 'n',
        k_col: str = 'k',
        to_rna: bool = True,
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

        for node in roots:
            if not isinstance(node, MotifNode):
                raise TypeError(
                    'arg `forest` must be composed of MotifNode instances. '
                    f'found: {type(node)}'
                )

        self._edges: dict[tuple[int, int], MotifEdge] = {}
        self._edges_valid: bool = False

    def __iter__(self) -> Iterator[MotifNode]:
        return iter(self.flatten())

    def invalidate_edges(self) -> None:
        self._edges_valid = False

    def build_edges(self) -> None:
        old = self._edges
        new: dict[tuple[int, int], MotifEdge] = {}

        for child in self.flatten():
            for parent in getattr(child, 'parents', None) or []:
                k = (id(parent), id(child))
                e = old.get(k)
                if e is None:
                    e = MotifEdge(parent=parent, child=child)
                else:
                    e.parent = parent
                    e.child = child
                new[k] = e

        self._edges = new
        self._edges_valid = True

    def iter_edges(self) -> Iterator[MotifEdge]:
        if not self._edges_valid:
            self.build_edges()
        yield from self._edges.values()

    def iter_frontier_edges(self) -> Iterator[MotifEdge]:
        if not self._edges_valid:
            self.build_edges()

        memo: dict[int, list[MotifNode]] = {}
        seen_edges: set[tuple[int, int]] = set()

        for child in self.flatten():
            if not getattr(child, 'alive', True):
                continue

            for parent in self.frontier_parents(child, memo=memo):
                if not getattr(parent, 'alive', True):
                    continue

                k = (id(parent), id(child))
                if k in seen_edges:
                    continue
                seen_edges.add(k)

                e = self._edges.get(k)
                if e is None:
                    e = MotifEdge(parent=parent, child=child)
                    self._edges[k] = e
                yield e

    def traverse(
        self,
        visited: set | None = None,
        func: Callable[..., Any] | None = None,
        params: list[Any] | None = None,
    ) -> list:
        if self.roots is None:
            raise ValueError('arg `roots` cannot be None')
        if visited is None:
            visited = set()
        if params is None:
            params = []

        results: list[Any] = []
        nodes = self.flatten()

        desc = f'Traversing with {func.__name__}' if func else f'Traversing'
        for node in tqdm(nodes, desc=desc, unit='node'):
            if func is not None:
                results.append(func(node, *params))

        return results

    def flatten(self) -> list[MotifNode]:
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

        return out

    # potentially deprecated:
    def prune(self, node: MotifNode) -> None:
        if not self.roots:
            raise ValueError('arg `roots` cannot be None')

        nodes = self.flatten()
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

        self.invalidate_edges()

        node.children = []
        node.parents = []

    def frontier_parents(
        self,
        node: MotifNode,
        *,
        memo: dict[int, list[MotifNode]] | None = None,
    ) -> list[MotifNode]:
        if memo is None:
            memo = {}

        key = id(node)
        if key in memo:
            return memo[key]

        out: list[MotifNode] = []
        seen: set[int] = set()

        stack = list(getattr(node, 'parents', []))
        while stack:
            p = stack.pop()
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)

            if getattr(p, 'alive', True):
                out.append(p)
                continue

            stack.extend(getattr(p, 'parents', []))

        uniq: list[MotifNode] = []
        used: set[int] = set()
        for p in out:
            if id(p) not in used:
                uniq.append(p)
                used.add(id(p))

        memo[key] = uniq
        return uniq

    def to_pd(self) -> pd.DataFrame:
        entries = []
        for node in self.flatten():
            p = getattr(node, 'parents', None)
            entries.append({
                'motif_seq': getattr(node, 'motif_seq', None),
                'alive': getattr(node, 'alive', None),
                'node_id': getattr(node, 'node_id', None),
                'edit': getattr(node, 'edit', None),
                'motif_state': getattr(node, 'motif_state', None),
                'parent1': getattr(p[0], 'motif_seq', None) if p else None,
                'parent2': getattr(p[1], 'motif_seq', None) if p else None
            })
        return pd.DataFrame.from_records(entries)

    def to_html(
        self,
        outfile: str = 'forest.html',
        title: str | None = None,
        subtitle: str | None = None,
        *,
        color_by: Literal['editing', 'motif_status'] = 'editing',
        status_attr: str = 'motif_state',
        open_browser: bool = True,
        prog: str = 'dot',
    ) -> None:
        if not self.roots:
            raise ValueError('arg `forest` cannot be None/empty.')

        # 1) Collect SP edges (assumes you've already computed edge.sp somewhere)
        edges_all = list(self.iter_edges())
        sp_edges = [e for e in edges_all if bool(getattr(e, 'sp', False))]
        if not sp_edges:
            raise ValueError('no SP edges found; nothing to render')

        # 2) Seed node set with endpoints of SP edges
        nodes_by_id: dict[int, MotifNode] = {}
        for e in sp_edges:
            nodes_by_id[id(e.parent)] = e.parent
            nodes_by_id[id(e.child)] = e.child

        # 3) Ancestor closure (pull in roots / dead scaffolding)
        stack = list(nodes_by_id.values())
        seen: set[int] = set(nodes_by_id.keys())
        while stack:
            cur = stack.pop()
            for p in (getattr(cur, 'parents', None) or []):
                pid = id(p)
                if pid in seen:
                    continue
                seen.add(pid)
                nodes_by_id[pid] = p
                stack.append(p)

        # 4) Build a canonical seq->node map (dedupe by motif_seq)
        all_nodes: dict[str, MotifNode] = {}
        for nd in nodes_by_id.values():
            seq = getattr(nd, 'motif_seq', None)
            if seq is None:
                continue
            if seq not in all_nodes:
                all_nodes[seq] = nd

        if not all_nodes:
            raise ValueError('no motifs to render after filtering by motif_seq')

        # 5) Edges to draw: keep any structural edge whose endpoints are included
        included_ids: set[int] = set(nodes_by_id.keys())
        edges_draw: list[MotifEdge] = [
            e for e in edges_all
            if id(e.parent) in included_ids and id(e.child) in included_ids
        ]

        # 6) Build NetworkX graph using motif_seq keys
        G = nx.DiGraph()
        G.add_nodes_from(all_nodes.keys())

        seen_e: set[tuple[str, str]] = set()
        for e in edges_draw:
            u = getattr(e.parent, 'motif_seq', None)
            v = getattr(e.child, 'motif_seq', None)
            if u is None or v is None:
                continue
            if u not in all_nodes or v not in all_nodes:
                continue
            k = (u, v)
            if k in seen_e:
                continue
            seen_e.add(k)
            G.add_edge(u, v)

        # 7) Layout
        try:
            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                pos = graphviz_layout(G, prog=prog)
            else:
                pos = nx.spring_layout(G, seed=0)
        except Exception:
            pos = nx.spring_layout(G, seed=0)

        if pos:
            xs = np.array([p[0] for p in pos.values()], dtype=float)
            ys = np.array([p[1] for p in pos.values()], dtype=float)

            xs -= xs.mean() if xs.size else 0.0
            ys -= ys.mean() if ys.size else 0.0

            scale = max(xs.std() if xs.size else 1.0, ys.std() if ys.size else 1.0, 1e-9)
            xs = xs / scale * 800
            ys = ys / scale * 800

            pos = {k: (float(x), float(y)) for k, x, y in zip(pos.keys(), xs, ys)}
        else:
            pos = {}

        # 8) Coloring setup
        norm = None
        cmap = None
        status_palette = {
            0: '#bdbdbd',  # none/pruned
            1: '#6baed6',  # context-ish
            2: '#31a354',  # true-ish
        }

        if color_by == 'editing':
            edit_vals = [
                float(getattr(nd, 'edit', np.nan))
                for nd in all_nodes.values()
                if getattr(nd, 'edit', None) is not None and np.isfinite(getattr(nd, 'edit'))
            ]
            if edit_vals:
                vals = np.asarray(edit_vals, dtype=float)
                vmin = np.percentile(vals, 10)
                vmax = np.percentile(vals, 95)
                if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                    cmap = cm.get_cmap('Reds')

        # 9) Sizes (by prevalence, default 1)
        prevs = np.array(
            [float(getattr(nd, 'prevalence', 1) or 1) for nd in all_nodes.values()],
            dtype=float
        )
        prevs[~np.isfinite(prevs)] = 1.0
        prevs[prevs <= 0] = 1.0

        lo, hi = np.percentile(prevs, [5, 95]) if prevs.size else (1.0, 1.0)
        sizes_clipped = np.clip(prevs, lo, hi)
        sizes_scaled = np.sqrt(sizes_clipped)
        denom = (sizes_scaled.max() - sizes_scaled.min()) + 1e-9
        node_sizes = 10 + 40 * (sizes_scaled - sizes_scaled.min()) / denom
        size_map = dict(zip(all_nodes.keys(), node_sizes))

        # 10) PyVis render
        net = Network(height='750px', width='100%', directed=True, notebook=False)
        net.toggle_physics(False)
        net.set_edge_smooth('straight')

        for seq, nd in all_nodes.items():
            p_raw = getattr(nd, 'pval', None)
            p_str = (
                f'{float(p_raw):.4f}'
                if isinstance(p_raw, (int, float, np.integer, np.floating)) and np.isfinite(p_raw)
                else '–'
            )

            avg = getattr(nd, 'edit', None)
            edit_str = f'{float(avg):.3f}' if avg is not None and np.isfinite(avg) else '–'

            prev = getattr(nd, 'prevalence', None)
            prev_i = int(prev) if isinstance(prev, (int, np.integer)) and prev > 0 else 1

            status_val = getattr(nd, status_attr, None)

            if color_by == 'motif_status':
                try:
                    color = status_palette.get(int(status_val), '#bdbdbd') if status_val is not None else '#bdbdbd'
                except Exception:
                    color = '#bdbdbd'
            else:
                color = '#cccccc'
                if cmap is not None and norm is not None and avg is not None and np.isfinite(avg):
                    color = mcolors.to_hex(cmap(norm(float(avg))))

            node_title = (
                f'Seq: {seq}, '
                f'Count: {prev_i}, '
                f'Edit avg: {edit_str}, '
                f'p-val: {p_str}, '
                f'{status_attr}: {status_val if status_val is not None else "–"}'
            )

            net.add_node(
                seq,
                label=seq,
                title=node_title,
                color=color,
                size=float(size_map.get(seq, 10.0)),
            )

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
