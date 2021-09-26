import shutil
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path


def vis(g: nx.Graph, path: Path):
    pos = nx.spring_layout(g, iterations=100, seed=0)
    nx.draw(g, pos)
    plt.savefig(path, format="PNG")
    plt.clf()


def graph_to_graphing(graph: nx.Graph, resolution: int = 1000):
    out_dir = Path('graphing/vis')

    try:
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    except OSError:
        pass

    graph = nx.convert_node_labels_to_integers(graph, first_label=1)
    vis(graph, out_dir / 'orig.png')

    n_verts = len(graph)
    n_bracket = list(range(1, n_verts + 1))  # [n] = 1, 2, ..., n

    j_intervals = [
        list(map(lambda x: round(x, 8), np.linspace((i - 1) / n_verts, i / n_verts, resolution)[:-1]))
        for i in n_bracket
    ]

    X, Y = [], []

    for i, j in graph.edges():
        i, j = sorted([i, j])

        from_interval = j_intervals[i - 1]
        to_interval = j_intervals[j - 1]

        for x in from_interval:
            y = round(x + (j - i) / n_verts, 8)

            if y in to_interval:
                X.append(x)
                Y.append(y)

            else:
                print('here')

    plt.plot(X, Y)
    plt.savefig(out_dir / 'graphing.png')
    plt.clf()

    new_graph = nx.Graph()
    new_graph.add_edges_from(zip(X, Y))

    CGs = [new_graph.subgraph(c) for c in nx.connected_components(new_graph)]
    isos = 0

    for i, g in enumerate(CGs):
        # vis(g, out_dir / f'{i}.png')
        if nx.algorithms.isomorphism.is_isomorphic(graph, g):
            isos += 1

    print(f'Iso {isos} / {len(CGs)}')

g = nx.generators.lattice.grid_graph((5, 5))
# g = nx.generators.classic.cycle_graph(5)

graph_to_graphing(g)
