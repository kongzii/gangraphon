import math
import pickle
import torch
import typer
import glob
import inspect
import typing as t
import numpy as np
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
import multiprocessing as mp

from tqdm import tqdm
from pebble import concurrent
from concurrent.futures import TimeoutError
from NEMtropy import UndirectedGraph

from common import tools


@tools.persist
def sample_ergms(graph_file: str, n_samples: int, seed: int = 0) -> t.List[nx.Graph]:
    if graph_file.endswith('.graphsplit'):
        with open(graph_file, 'rb') as fr:
            graph, _ = pickle.load(fr)
    else:
        graph = tools.nx_from_file(graph_file)

    nem_graph = UndirectedGraph(edgelist=list(graph.edges))
    nem_graph.solve_tool(
        model="cm",
        method="quasinewton",
        initial_guess="random",
    )

    nem_graph.ensemble_sampler(
        n=n_samples,
        cpu_n=mp.cpu_count(),
        output_dir="/tmp/ergm/",
        seed=seed,
    )

    graphs = [
        nx.convert_node_labels_to_integers(tools.nx_from_file(p), first_label=0)
        for p in glob.glob("/tmp/ergm/*")
    ]

    return graphs


@concurrent.process(timeout=3 * 60)
def do_concurrently(function, *args, **kwargs):
    out = function(*args, **kwargs)
    if inspect.isgenerator(out):
        out = list(out)
    return out


def cumulative_degree_distribution_plot(g: nx.Graph):
    degrees = np.array([d for _, d in g.degree()])
    cdd = np.array([(degrees <= d).sum() / len(g) for d in range(max(degrees))])

    y = cdd
    x = np.array(list(range(len(cdd))))

    return x, y


def sorted_norm_degrees(graph: nx.Graph):
    return np.array(sorted([d / len(graph) for _, d in graph.degree()]))


def sorted_norm_degrees_cumsum(graph: nx.Graph):
    return np.cumsum(sorted([d / len(graph) for _, d in graph.degree()]))


def sorted_degrees(graph: nx.Graph):
    return np.array(sorted([d for _, d in graph.degree()]))


def sorted_degrees_cumsum(graph: nx.Graph):
    return np.cumsum(sorted([d for _, d in graph.degree()]))


def vis_plot(axe, outputs, x=None, **kwargs):
    outputs = [o for o in outputs if o is not None]

    if not outputs:
        return

    y_mean, y_median, y_min, y_max = [], [], [], []

    for i in range(max(len(o) for o in outputs)):
        values = [float(o[i]) for o in outputs if len(o) > i and o[i] is not None]

        if not values:
            continue

        y_median.append(np.median(values))
        y_mean.append(np.mean(values))
        y_min.append(np.min(values))
        y_max.append(np.max(values))

    if not y_mean:
        return

    x = np.array(list(range(len(y_mean)))) if x is None else x

    label = kwargs.get("label", "")
    del kwargs["label"]

    axe.plot(
        x, y_mean, linewidth=1.0, linestyle="--", label=f"{label} (mean)", **kwargs
    )
    axe.plot(
        x, y_median, linewidth=1.0, linestyle=":", label=f"{label} (median)", **kwargs
    )
    axe.fill_between(x, y_min, y_max, alpha=0.1, color=kwargs.get("color"))


def vis_bar(x, axe, outputs, **kwargs):
    outputs = [o for o in outputs if o is not None]

    if not outputs:
        return

    y_median, y_mean, y_min, y_max = (
        np.median(outputs),
        np.mean(outputs),
        np.min(outputs),
        np.max(outputs),
    )

    label = kwargs.get("label", "")
    del kwargs["label"]

    axe.bar(
        x - 0.2,
        y_mean,
        xerr=x,
        yerr=[[y_min], [y_max]],
        linewidth=1.0,
        ecolor=kwargs.get("color"),
        width=0.3,
        label=f"{label} (mean)",
        edgecolor="white",
        **kwargs,
    )
    axe.bar(
        x + 0.2,
        y_median,
        xerr=x,
        yerr=[[y_min], [y_max]],
        linewidth=1.0,
        ecolor=kwargs.get("color"),
        width=0.3,
        label=f"{label} (median)",
        edgecolor="white",
        **kwargs,
    )


def s_metric(g):
    return nx.s_metric(g, normalized=False)


def edge_density(g):
    return len(g.edges()) / math.comb(len(g), 2)


def mean_degree(g):
    deg_hist = nx.degree_histogram(g)
    degrees = list(it.chain(*[[degree] * num for degree, num in enumerate(deg_hist)]))
    return np.mean(degrees)


def median_degree(g):
    deg_hist = nx.degree_histogram(g)
    degrees = list(it.chain(*[[degree] * num for degree, num in enumerate(deg_hist)]))
    return np.median(degrees)


def degree_histogram_cumsum(g):
    y = np.cumsum(nx.degree_histogram(g))
    return y


def degree_histogram_histogram_cumsum(g):
    hist = nx.degree_histogram(g)

    values, bins = np.histogram(hist, bins=len(hist))
    x = bins[:-1]
    y = np.cumsum(values)

    return x, y


TESTS = [
    edge_density,
    s_metric,  # float
    nx.degree_assortativity_coefficient,  # float
    nx.average_clustering,  # float
    nx.global_efficiency,  # float
    nx.laplacian_spectrum,  # list[float]
    cumulative_degree_distribution_plot,
    # degree_histogram_cumsum,  # list[float]
    # degree_histogram_histogram_cumsum,
    # sorted_norm_degrees_cumsum,
    # sorted_norm_degrees,
    # sorted_degrees_cumsum,
    # sorted_degrees,
    mean_degree,
    median_degree,
    nx.pagerank,  # dict[int, float]
    nx.k_nearest_neighbors,  # dict[int, float]
    nx.eigenvector_centrality,  # dict[int, float]
    nx.closeness_centrality,  # dict[int, float]
    nx.average_degree_connectivity,  # dict[int, int]
    nx.hits,  # tuple[dict[int, float], dict[int, float]], taking too long
]


def main(
    graphs_struct,
    fig_path: str,
    tests: list = TESTS,
    cpu_batch: int = 8,
    legend: bool = True,
):
    n_cols = 3
    extra_plots = 0  # 1 for shorter paths
    n_plots = extra_plots + len(tests)
    subplots = (math.ceil(n_plots / n_cols), n_cols)
    fig, axs = plt.subplots(*subplots, figsize=(subplots[1] * 7, subplots[0] * 7))

    multi_graphs = [g for _, g, _ in graphs_struct]
    names = [n for n, _, _ in graphs_struct]
    colors = [c for _, _, c in graphs_struct]

    all_outputs = []
    batch = []

    for test in tqdm(tests):
        outputs = []

        for graphs in multi_graphs:
            graphs_tests = []

            for graph in graphs:
                batch.append(do_concurrently(test, graph))

                if len(batch) == cpu_batch:
                    for task in batch:
                        try:
                            graphs_tests.append(task.result())
                        except TimeoutError:
                            graphs_tests.append(None)
                    batch = []

            if batch:
                for task in batch:
                    try:
                        graphs_tests.append(task.result())
                    except TimeoutError:
                        graphs_tests.append(None)
                batch = []

            outputs.append(graphs_tests)

        all_outputs.append(outputs)

    for index, f, outputs in tqdm(zip(range(len(tests)), tests, all_outputs)):
        axe = axs[math.floor(index / n_cols)][index % n_cols]

        axe.title.set_text(f.__name__)
        legend_loc = "upper right"

        for i, (output, name, color) in enumerate(zip(outputs, names, colors)):
            if f.__name__ in (
                "laplacian_spectrum",
                "degree_histogram_cumsum",
                "sorted_norm_degrees_cumsum",
                "sorted_norm_degrees",
                "sorted_degrees_cumsum",
                "sorted_degrees",
            ):
                legend_loc = "lower right"
                vis_plot(axe, output, label=name, color=color)

            elif f.__name__ in (
                "edge_density",
                "mean_degree",
                "median_degree",
                "s_metric",
                "degree_assortativity_coefficient",
                "average_clustering",
                "global_efficiency",
                "node_connectivity",
            ):
                legend_loc = "lower right"
                vis_bar(i, axe, output, label=name, color=color)

            elif f.__name__ in (
                "pagerank",
                "k_nearest_neighbors",
                "eigenvector_centrality",
                "closeness_centrality",
                "average_degree_connectivity",
            ):
                legend_loc = "lower right"
                vis_plot(
                    axe,
                    [sorted(list(o.values())) for o in output if o is not None],
                    label=name,
                    color=color,
                )

            elif f.__name__ in (
                "cumulative_degree_distribution_plot",
                "cumulative_degree_distribution_plot_2",
                "degree_histogram_histogram_cumsum",
            ):
                legend_loc = "lower right"
                vis_plot(
                    axe,
                    [o[1] for o in output if o is not None],
                    x=max([o[0] for o in output], key=len),
                    label=name,
                    color=color,
                )

            elif f.__name__ in ("hits",):
                legend_loc = "lower right"
                vis_plot(
                    axe,
                    [sorted(list(o[0].values())) for o in output if o is not None],
                    label=name,
                    color=color,
                )

            else:
                print(f"Skipping {f.__name__}.")
                continue

            if legend:
                axe.legend(loc=legend_loc)

            elif axe.get_legend():
                axe.get_legend().remove()

            fig.savefig(fig_path)

    plt.close(fig)


def run(
    graph_file: str,
    checkpoint: str,
    ergm: bool = True,
    n_samples: int = 100,
):
    graph = nx.convert_node_labels_to_integers(
        tools.nx_from_file(graph_file), first_label=0
    )

    struct = [("original", [graph, graph], "red")]

    if checkpoint:
        model = models.GAN.load_from_checkpoint(checkpoint)
        if torch.cuda.is_available():
            model.cuda()
        graphons = [model.sample() for _ in tqdm(range(n_samples), desc="Graphon")]
        struct.append(("graphon", graphons, "green"))

    if ergm:
        ergms = sample_ergms(graph_file, n_samples)
        struct.append(("ergm", ergms, "blue"))

    output_dir = "/".join(checkpoint.split("/")[:-1])
    main(struct, f"{output_dir}/tests.n_samples={n_samples}.jpg")


if __name__ == "__main__":
    from gangraphon import models

    typer.run(run)
