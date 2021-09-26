import gc
import json
import typer
import typing as t
import numpy as np
import networkx as nx
import itertools as it
import torch_geometric as tg
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from tqdm import tqdm
from collections import defaultdict
from scipy.stats import spearmanr
from common import tools, tests, graphons
from common.graphons import Graphon
from common.evaluators import pertubations
from common.evaluators.eval import eval_tdas, TDA_EVALUATORS
from common.data import DatasetGraphon


def pertub(
    graph: nx.Graph, pertubation: str, probability: float, n_pertubs: int, seed: int
):
    pertubator = getattr(pertubations, pertubation)(
        probability, random_state=np.random.RandomState(seed)
    )
    pertubed_graphs = [pertubator(graph) for _ in range(n_pertubs)]

    return pertubed_graphs


def main(
    graph_file: t.Optional[str] = None,
    graphon_name: t.Optional[Graphon] = None,
    tudataset_name: t.Optional[str] = None,
    n_pertubs: int = 10,
    seed: int = 0,
    networkx: bool = False,
):
    pl.seed_everything(seed)

    if graph_file:
        graphs = [
            nx.convert_node_labels_to_integers(
                tools.nx_from_file(graph_file), first_label=0
            )
        ]
        base_dir = "/".join(graph_file.split("/")[:-1])

    elif graphon_name:
        n_vertices = 250
        graphon_func = getattr(graphons, graphon_name.value)
        graphs = [
            DatasetGraphon.sample(graphon_func, n_vertices=n_vertices)
            for _ in range(10)
        ]
        base_dir = DatasetGraphon(graphon_func).base_path

    elif tudataset_name:
        dataset = tg.datasets.TUDataset(
            name=tudataset_name, root="data/tudataset", cleaned=False
        )
        dataset.shuffle()
        base_dir = dataset.root + f"/{tudataset_name}"

        graphs = []
        for data in tqdm(dataset[:100], "Parsing TUDataset"):
            graphs.append(tg.utils.to_networkx(data, to_undirected=True))

    else:
        raise RuntimeError

    outcomes = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    probabilities = list(np.linspace(0.1, 0.9, 10))
    pertubations = [
        "AddEdges",
        "RemoveEdges",
        "RewireEdges",
    ]

    for pertubation in pertubations:
        if pertubation == "RewireEdges" and networkx:
            all_pertubed_graphs = []

        for p in tqdm(probabilities):
            pertubed_graphs = [
                pg
                for graph in graphs
                for pg in pertub(graph, pertubation, p, n_pertubs, seed)
            ]

            if pertubation == "RewireEdges" and networkx:
                all_pertubed_graphs.append(pertubed_graphs)

            for name, distance in eval_tdas(graphs, pertubed_graphs).items():
                outcomes[pertubation][name]["distances"].append(distance)

        for name in outcomes[pertubation].keys():
            coef = spearmanr(
                np.array(probabilities),
                np.array(outcomes[pertubation][name]["distances"]),
            )
            outcomes[pertubation][name]["spearmanr"] = {
                "correlation": coef[0] if not np.isnan(coef[0]) else None,
                "pvalue": coef[1] if not np.isnan(coef[1]) else None,
            }

        if pertubation == "RewireEdges" and networkx:
            tests.main(
                [
                    ("original", graphs, "red"),
                    *[
                        ("pertubed", pertubed_graphs, [0.5, 0.5, 0.5, p])
                        for p, pertubed_graphs in zip(
                            probabilities, all_pertubed_graphs
                        )
                    ],
                ],
                f"{base_dir}/pertubations.{pertubation}.tests.jpg",
                legend=False,
            )

        gc.collect()

    with open(f"{base_dir}/pertubations.tda.json", "w") as f:
        json.dump(outcomes, f, indent=2)

    subplots = (len(outcomes), len(TDA_EVALUATORS))
    fig, axes = plt.subplots(*subplots, figsize=(subplots[1] * 10, subplots[0] * 10))

    for irow, (pertubation, methods) in enumerate(outcomes.items()):
        for icol, (name, values) in enumerate(methods.items()):
            axes[irow][icol].plot(probabilities, values["distances"], linewidth=1.0)

    pad = 8
    for ax, col in zip(axes[0], map(str, TDA_EVALUATORS)):
        ax.annotate(
            col,
            xy=(0.5, 1.05),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size=30.0,
            ha="center",
            va="baseline",
        )
    for ax, row in zip(axes[:, 0], pertubations):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size=30.0,
            ha="right",
            va="center",
        )
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, top=0.95)

    fig.savefig(f"{base_dir}/pertubations.tda.jpg")
    plt.close(fig)


if __name__ == "__main__":
    typer.run(main)
