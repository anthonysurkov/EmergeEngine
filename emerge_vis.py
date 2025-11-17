from __future__ import annotations
from pathlib import Path

from pyvis.network import Network
from typing import List
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from emerge_tokenizer import MotifNode

ROOT = Path(__file__).resolve().parent

def forest_to_html(
    emerge_forest: list[MotifNode],
    outfile = 'emerge_forest.html',
    title: str | None = None,
    subtitle: str | None = None
):
    if not isinstance(emerge_forest, list):
        raise TypeError('emerge_forest must be a list of MotifNode objects')
    for nd in emerge_forest:
        if not isinstance(nd, MotifNode):
            raise TypeError(
                'Elements of emerge_forest must be MotifNode instances'
            )
    if not isinstance(outfile, str):
        raise TypeError('outfile must be a string')

    all_nodes = {}
    stack = emerge_forest.copy()

    while stack:
        nd = stack.pop()
        all_nodes[nd.motif_seq] = nd
        stack.extend(nd.children)

    G = nx.DiGraph()
    for nd in all_nodes.values():
        for child in nd.children:
            G.add_edge(child.motif_seq, nd.motif_seq)

    try:
        pos = graphviz_layout(G, prog='dot')
    except Exception:
        raise RuntimeError(
            'Graphviz `dot` layout failed. Ensure graphviz is installed.'
        )

    edit_vals = [
        nd.editing for nd in all_nodes.values() if nd.editing is not None
    ]

    if edit_vals:
        norm = mcolors.Normalize(vmin=min(edit_vals), vmax=max(edit_vals))
        cmap = cm.get_cmap('Reds')
    else:
        norm = None
        cmap = None

    net = Network(height='800px', width='100%', directed=True, notebook=False)
    net.toggle_physics(False)
    net.set_edge_smooth('straight')

    for seq, nd in all_nodes.items():
        if cmap:
            rgba = cmap(norm(nd.editing if nd.editing is not None else 0))
            color = mcolors.to_hex(rgba)
        else:
            color = '#cccccc'

        prevalence = nd.prevalence if nd.prevalence is not None else 1
        rank = nd.rank if nd.rank is not None else '–'
        edit_str = f'{nd.editing:.3f}' if nd.editing is not None else '–'

        node_title = (
            f'{seq} \ '
            f'Rank: {rank} \ '
            f'Count: {prevalence} \ '
            f'Avg editing: {edit_str}'
        )
        net.add_node(
            seq,
            label=seq,
            title=node_title,
            color=color,
            size=10 + prevalence * 0.25,
            html=True,
        )

    for u, v in G.edges():
        net.add_edge(u, v)

    node_index = {node['id']: idx for idx, node in enumerate(net.nodes)}

    for seq, (x, y) in pos.items():
        idx = node_index[seq]
        net.nodes[idx]['x'] = x
        net.nodes[idx]['y'] = -y

    net.write_html(outfile, open_browser=True)

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
        section += '</div>'

        html = html.replace('<body>', f'<body>{section}', 1)

        with open(outfile, 'w', encoding='utf-8') as f:
            f.write(html)
