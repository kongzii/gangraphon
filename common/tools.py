import os
import math
import torch
import pickle
import random
import inspect
import functools
import typing as t
import numpy as np
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt

from tqdm import tqdm
from operator import mul
from functools import reduce
from pathlib import Path


class Cache:
    def __init__(self):
        self.cache = {}

    def __call__(self, f: t.Callable[..., t.Any], *args, **kwargs) -> t.Any:
        name = inspect.getsource(f) + str(args) + str(kwargs)

        if name not in self.cache:
            self.cache[name] = f(*args, **kwargs)

        return self.cache[name]


class NPGraph:
    def __init__(self, V: np.ndarray, E: np.ndarray):
        self.V = V.copy()
        self.E = E.copy()

    def __iter__(self):
        yield self.V
        yield self.E

    @classmethod
    def from_adj(cls, adj: np.ndarray):
        return cls.from_nx(nx.from_numpy_matrix(adj))

    @classmethod
    def from_nx(cls, nx_g):
        v = np.array(nx_g.nodes).astype(int)
        e = nx.to_pandas_edgelist(nx_g).values.astype(int)

        return NPGraph(v, e)

    def to_nx(self, graph_class=nx.Graph):
        nx_g = graph_class()

        nx_g.add_nodes_from(self.V)
        nx_g.add_edges_from(self.E)

        return nx_g


def batchit(iterable: t.Iterable[t.Any], n: int):
    batched = []

    for e in iterable:
        batched.append(e)

        if len(batched) >= n:
            yield batched
            batched = []

    if batched:
        yield batched


def adj_matrix_to_edge_index(adj_matrix, adj_feats=None) -> torch.LongTensor:
    assert len(adj_matrix.shape) == 2
    assert adj_matrix.shape[0] == adj_matrix.shape[1]

    nodes = list(range(len(adj_matrix)))

    possible_edges = list(it.permutations(list(range(len(nodes))), 2)) + [(i, i) for i in range(len(nodes))]

    from_ = []
    to_ = []
    feats = []

    for a, b in possible_edges:
        if adj_matrix[a][b]:
            from_.append(a)
            to_.append(b)

            if adj_feats:
                feats.append(adj_feats[a][b])

    edge_index = torch.LongTensor([from_, to_])

    if adj_feats:
        edge_index, torch.stack(adj_feats)

    return edge_index


def nx_from_file(path: str) -> nx.Graph:
    """
    Loads undirected graph from file.
    Multiple edges between same nodes will be removed.
    Line is expected to be in format `node_from node_to`, if more values on line are present (like edge weights), they are ignored.
    If `*.mtx` file, first line is skipped.
    """

    graph = nx.Graph()

    with open(path) as f:
        for i, line in tqdm(enumerate(f), desc=f"Loading {path}"):
            if not i and path.endswith(".mtx"):
                continue

            from_, to_, *_ = line.split()
            graph.add_edge(from_, to_)

    return graph


def complement_graph(vertices, edges):
    # Assume everything is a non-edge
    non_edges = np.ones((len(vertices), len(vertices)), dtype=int)

    # Remove self-edges
    np.fill_diagonal(non_edges, 0)

    # We need only one triangle if graph is undirected
    non_edges = np.triu(non_edges)

    # Remove existing edges in both directions
    non_edges[edges[:, 0], edges[:, 1]] = 0
    non_edges[edges[:, 1], edges[:, 0]] = 0

    # Get (from, to) ordered indices
    non_edges = np.argwhere(non_edges == 1)

    return vertices, non_edges


def complete_graph(n: int):
    v = np.arange(n)
    e = np.transpose(np.vstack(np.tril_indices(n)))
    e = e[e[:, 0] != e[:, 1]]

    return NPGraph(v, e)


def create_all_graphs(up_to_n: int, as_np: bool = True):
    """Creates ALL possible isomorphism-unique graphs of size 2 to n"""

    graphs: t.List[nx.Graph] = []

    # Iterate over vertex counts v
    for v in range(2, up_to_n + 1):
        graph_comp = complete_graph(v)

        # Iterate over all possible graphs of size v
        edge_selectors = np.array(
            [i for i in it.product(range(2), repeat=int(v * (v - 1) / 2))], dtype=bool
        )

        for es in edge_selectors:
            # Discard if independent set
            if es.sum() == 0:
                continue

            graph_nx = NPGraph(graph_comp.V, graph_comp.E[es, :]).to_nx()

            # Check if new graph is isomorphic with any of the existing ones
            isomorphic = False
            for i in range(len(graphs)):
                if nx.algorithms.isomorphism.is_isomorphic(graphs[i], graph_nx):
                    isomorphic = True
                    break

            if not isomorphic:
                graphs.append(graph_nx)

    if as_np:
        graphs = [NPGraph.from_nx(g) for g in graphs]

    return graphs


def product(numbers: t.Iterable[t.SupportsFloat]):
    return reduce(mul, numbers, 1)


def escape(s: t.Any) -> str:
    s = str(s)
    s = s.replace("(", "-")
    s = s.replace(")", "-")
    s = s.replace("/", "-")
    s = s.replace("*", "-")
    return s


def persist(func: t.Callable):
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        name = (
            func.__name__
            + "."
            + "_".join(map(escape, args))
            + "."
            + "_".join(map(escape, kwargs.items()))
        )
        path = tmp_dir / f"{name}.pickle"

        if os.path.isfile(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        else:
            return_value = func(*args, **kwargs)
            with open(path, "wb") as f:
                pickle.dump(return_value, f)
            return return_value

    return wrapper


def create_image_mat(
    graphon: t.Callable,
    resolution: int,
    normalize: bool = False,
):
    """Plots the graphon W as a unit square function."""
    uniform_args = np.linspace(start=0, stop=1, num=resolution)

    cartesian_product = np.transpose(
        [
            np.tile(uniform_args, len(uniform_args)),
            np.repeat(uniform_args, len(uniform_args)),
        ]
    )

    if torch and isinstance(graphon, torch.nn.Module):
        cartesian_product = (
            torch.from_numpy(cartesian_product).float().to(graphon.device)
        )

    img_mat = graphon(cartesian_product[:, 0], cartesian_product[:, 1])

    if torch and isinstance(graphon, torch.nn.Module):
        img_mat = img_mat.cpu().numpy()

    if len(img_mat.shape) != 2 or img_mat.shape[0] != img_mat.shape[1]:
        img_mat = img_mat.reshape(resolution, resolution)

    if normalize:
        img_mat = img_mat / img_mat.max()

    return img_mat


def visualize(
    graphon,
    save_path: str,
    resolution: int,
    normalize: bool = False,
    sample: t.Optional[int] = None,
):
    img_mat = create_image_mat(graphon, resolution, normalize)

    plt.figure()

    plt.imshow(
        X=img_mat, origin="lower", extent=[0, 1, 0, 1], cmap="plasma", vmin=0, vmax=1
    )

    if sample:
        edges_count = 0
        nodes = np.random.rand(sample)

        for a, b in tqdm(it.combinations(nodes, 2), total=math.comb(len(nodes), 2)):
            a, b = sorted([a, b])

            if random.random() <= graphon(np.array([a]), np.array([b])):
                edges_count += 1

            plt.scatter(a, b, s=10, color="red")

        save_path = save_path.replace("nedges", f"nedges={edges_count}")

    plt.savefig(save_path)
    plt.close()


def torch_eval(func):
    @functools.wraps(func)
    def wrapper(model, *args, **kwargs):
        if not isinstance(model, torch.nn.Module):
            raise ValueError("First argument must be an instance of torch.nn.Module.")

        was_training = model.training
        model.eval()
        return_value = func(model, *args, **kwargs)
        if was_training:
            model.train()

        return return_value

    return wrapper


def preprocess_vertices(x, y):
    if isinstance(x, np.ndarray):
        x = torch.Tensor(x)

    if isinstance(y, np.ndarray):
        y = torch.Tensor(y)

    if isinstance(x, float) and isinstance(y, float):
        edges = torch.Tensor([[x, y]])

    elif x.ndim == y.ndim == 1:
        edges = torch.hstack((x.float().unsqueeze(1), y.float().unsqueeze(1)))

    elif x.ndim == y.ndim == 2:
        edges = torch.hstack(
            (torch.reshape(x.float(), (-1, 1)), torch.reshape(y.float(), (-1, 1)))
        )

    else:
        raise ValueError(f"Invalid input of shape {x.shape=}, {y.shape=}.")

    return edges


def pad_graphs(graphs, pad_to=None):
    # find number of nodes in graph with most nodes:
    max_nodes = pad_to or max([len(g) for g in graphs])

    # networkx numbers nodes from 0 to the number of nodes -1 (=length of the graph -1)
    # so len(g) gives the smallest positive integer that can be used as a node name.
    for g in graphs:
        while len(g) < max_nodes:
            g.add_node(len(g))
