import os
import math
import pickle
import torch
import random
import networkx as nx
import numpy as np
import typing as t
import itertools as it
import torch_geometric as tg

from tqdm import tqdm
from pathlib import Path
from collections import namedtuple

from common import tools

# Match graphlet index from `DatasetBase.graphlets` to graphlet name from PGD.
INDEX_TO_PGD_GRAPHLET_NAME = {
    i: n
    for i, n in enumerate(
        [
            "total_2_1edge",
            "total_3_1edge",
            "total_2_star",
            "total_3_tris",
            "total_4_1edge",
            "total_4_2star",
            "total_3_star",
            "total_4_tri",
            "total_4_2edge",
            "total_4_path",
            "total_4_tailed_tris",
            "total_4_cycle",
            "total_4_chordcycle",
            "total_4_clique",
        ]
    )
}
# Calculated using Sage's `len(list(graph.automorphism_group()))` of corresponding graphlets.
INDEX_TO_GRAPHLET_N_AUTOMORPHISMS = {
    0: 2,
    1: 2,
    2: 2,
    3: 6,
    4: 4,
    5: 2,
    6: 6,
    7: 6,
    8: 8,
    9: 2,
    10: 2,
    11: 8,
    12: 4,
    13: 24,
}

Graphlet = namedtuple("Graphlet", "name graph complement_graph")


def isomorphic_adj_matrix(original: np.ndarray) -> np.ndarray:
    """
    Theoretically can return the same matrix.
    """
    assert len(original.shape) == 2, original.shape
    assert original.shape[0] == original.shape[1], original.shape

    c0 = random.randrange(0, original.shape[0])
    c1 = random.randrange(0, original.shape[0])

    isomorphic = original.copy()
    isomorphic[:, [c0, c1]] = isomorphic[:, [c1, c0]]
    isomorphic[[c0, c1], :] = isomorphic[[c1, c0], :]

    return isomorphic


class DatasetBase(torch.utils.data.IterableDataset):
    def __init__(self, graph_file: str, n_graphlets: int):
        super().__init__()

        self.n_graphlets = n_graphlets
        self.graph_file = graph_file
        self.base_path = "/".join(graph_file.split("/")[:-1])
        self.graph = nx.convert_node_labels_to_integers(
            tools.nx_from_file(graph_file), first_label=0
        )
        self.graphs = [self.graph]

        self.graph_degrees = np.array([d for _, d in self.graph.degree()])
        self.graph_degree_distribution = torch.Tensor(
            [
                (self.graph_degrees == d).sum() / len(self.graph)
                for d in range(len(self.graph))
            ]
        )
        self.graph_scaled_degrees = torch.Tensor(
            [degree / len(self.graph) for degree in self.graph_degrees]
        )

        self.graphlets = [
            Graphlet(
                INDEX_TO_PGD_GRAPHLET_NAME.get(i, f"graphlet_{i}"),
                g,
                tools.complement_graph(*g),
            )
            for i, g in enumerate(
                tqdm(tools.create_all_graphs(self.n_graphlets), desc="Graphlets")
            )
        ]

        self.cache = tools.Cache()

    @staticmethod
    def mapping_indices(vertices, edges, n_samples):
        return np.tile(edges.T, n_samples).T + len(vertices) * np.tile(
            np.repeat(np.arange(n_samples).reshape(-1, 1), len(edges), axis=0), 2
        )

    def get_dataloader(self, num_workers: int = 5):
        return torch.utils.data.DataLoader(
            self,
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )


class DatasetPGD(DatasetBase):
    def __init__(self, macro_file_path: str, *args, **kwargs):
        kwargs["n_graphlets"] = kwargs.get("n_graphlets", 4)
        super().__init__(*args, **kwargs)

        self.graphlet_name_to_count = {}

        with open(macro_file_path) as f:
            for line in f:
                name, val = line.strip().split(" = ")
                self.graphlet_name_to_count[name.strip()] = int(val)

    @staticmethod
    def tind(Hc: int, Vg: int, Vh: int):
        # DL Graphons Vu-Huy Hoang.pdf: Definition 2.2.4

        s, t = Vg - Vh + 1, Vg
        prod = tools.product(range(s, t + 1))

        if not prod:
            return None

        return Hc / prod

    def __iter__(self):
        indices = list(range(len(self.graphlets)))
        random.shuffle(indices)

        for i in indices:
            graphlet = self.graphlets[i]

            (vertices, edges) = graphlet.graph
            (_, non_edges) = graphlet.complement_graph

            n_samples = 1_000

            # Map vertices to real numbers from <0,1> n times
            edges_indices = self.mapping_indices(vertices, edges, n_samples)
            non_edges_indices = self.mapping_indices(vertices, non_edges, n_samples)

            # Uniformly sample vertices n times
            sampled_vertices = np.random.uniform(size=n_samples * len(vertices))

            sampled_edges = sampled_vertices[edges_indices]
            sampled_non_edges = sampled_vertices[non_edges_indices]

            sampled_edges_xy = (
                torch.from_numpy(sampled_edges[:, 0]).float(),
                torch.from_numpy(sampled_edges[:, 1]).float(),
            )
            sampled_non_edges_xy = (
                torch.from_numpy(sampled_non_edges[:, 0]).float(),
                torch.from_numpy(sampled_non_edges[:, 1]).float(),
            )

            hd = self.cache(
                lambda v, len_graph, len_vert, n_auts: self.tind(
                    v,
                    len_graph,
                    len_vert,
                )
                * n_auts,
                self.graphlet_name_to_count[graphlet.name],
                len(self.graph),
                len(vertices),
                INDEX_TO_GRAPHLET_N_AUTOMORPHISMS[i],
            )

            yield (
                (graphlet.name,),
                (
                    sampled_edges_xy,
                    len(edges),
                    sampled_non_edges_xy,
                    len(non_edges),
                    hd,
                    n_samples,
                ),
                (self.graph_degree_distribution, self.graph_scaled_degrees),
            )


class DatasetMC(DatasetBase):
    def __init__(self, *args, **kwargs):
        kwargs["n_graphlets"] = kwargs.get("n_graphlets", 4)
        super().__init__(*args, **kwargs)

        self.graph_homomorphism_densities = [
            self.estimate_naive_monte_carlo_homomorphism_density(
                g.graph.to_nx(), self.graph
            )
            for g in self.graphlets
        ]

    @staticmethod
    def estimate_naive_monte_carlo_homomorphism_density(
        g: nx.Graph, G: nx.Graph, epsilon: float = 0.01, gamma: float = 0.95
    ) -> float:
        """Homomorphism density for finite graphs estimation with naive Monte-Carlo"""

        g = nx.convert_node_labels_to_integers(g, first_label=0)
        G = nx.convert_node_labels_to_integers(G, first_label=0)

        # Unpack graph structs
        V_g, E_g = tools.NPGraph.from_nx(g)
        V_G, E_G = tools.NPGraph.from_nx(G)

        # Create adjacency matrix
        A_G = np.zeros((len(V_G), len(V_G)), dtype=int)
        A_G[E_G[:, 0], E_G[:, 1]] = 1
        A_G[E_G[:, 1], E_G[:, 0]] = 1

        # Sample size
        N = math.ceil((math.log(2) - math.log(1 - gamma)) / (2 * epsilon ** 2))

        # Sample vertex mappings V_g -> V_G
        mappings = np.random.randint(low=0, high=len(V_G) - 1, size=N * len(V_g))

        # Create mapped vertex edges to later check in adjacency matrix
        mapping_indices = np.tile(E_g.T, N).T + len(V_g) * np.tile(
            np.repeat(np.arange(N).reshape(-1, 1), len(E_g), axis=0), 2
        )
        mapped_edges = mappings[mapping_indices]

        # Get adjacency for each edge of the random mappings
        adjacency_indicators = A_G[mapped_edges[:, 0], mapped_edges[:, 1]]

        # Multiply adjacencies to check if the mappings represent homomorphisms then average
        hom_density = adjacency_indicators.reshape(N, len(E_g)).prod(axis=1).sum() / N

        return hom_density

    def __iter__(self):
        indices = list(range(len(self.graphlets)))
        random.shuffle(indices)

        for i in indices:
            graphlet = self.graphlets[i]

            (vertices, edges) = graphlet.graph

            n_samples = 1_000

            # Map vertices to real numbers from <0,1> n times
            edges_indices = self.mapping_indices(vertices, edges, n_samples)

            # Uniformly sample vertices n times
            sampled_vertices = np.random.uniform(size=n_samples * len(vertices))

            sampled_edges = sampled_vertices[edges_indices]

            sampled_edges_xy = (
                torch.from_numpy(sampled_edges[:, 0]).float(),
                torch.from_numpy(sampled_edges[:, 1]).float(),
            )

            yield (
                (graphlet.name,),
                (
                    sampled_edges_xy,
                    len(edges),
                    -1,
                    -1,
                    self.graph_homomorphism_densities[i],
                    n_samples,
                ),
                (self.graph_degree_distribution, self.graph_scaled_degrees),
            )


class DatasetGraphon(torch.utils.data.IterableDataset):
    def __init__(
        self,
        graphon_func: t.Callable,
        n_graphlets: int = 4,
        n_sampled_graphs: int = 1,
        sample_size: int = 1000,
        _graphs: t.List[nx.Graph] = None,
    ):
        super().__init__()

        self.datas = None
        self.n_graphlets = n_graphlets
        self.n_sampled_graphs = n_sampled_graphs
        self.sample_size = sample_size
        self.graphon_func = graphon_func
        self.base_path = f"data/artificial/{self.graphon_func.__name__}"
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

        self.graphlets = [
            Graphlet(
                INDEX_TO_PGD_GRAPHLET_NAME.get(i, f"graphlet_{i}"),
                g,
                tools.complement_graph(*g),
            )
            for i, g in enumerate(
                tqdm(tools.create_all_graphs(self.n_graphlets), desc="Graphlets")
            )
        ]

        self.graphs = (
            [
                self.sample(
                    self.graphon_func,
                    n_vertices=self.sample_size,
                )
                for _ in range(self.n_sampled_graphs)
            ]
            if not _graphs
            else _graphs
        )

        self.cache = tools.Cache()

    @staticmethod
    def sample(graphon, n_vertices: int):
        """Returns a random graph sampled from graphon W."""

        # Randomly select nodes in graphon
        V = np.random.uniform(size=n_vertices)

        # All possible edges
        edge_indices = np.transpose(np.vstack(np.tril_indices(len(V))))
        edges = V[edge_indices]

        # Edge probabilites (graphon values)
        edge_probs = graphon(edges[:, 0], edges[:, 1])

        # Convert float values of vertices to integer indexes
        V = np.arange(len(V))

        # Threshold probabilites to decide which pairs become edges
        E = np.random.uniform(size=len(edge_probs)) <= edge_probs
        E = edge_indices[E]

        return tools.NPGraph(V, E).to_nx()

    def get_dataloader(self, num_workers: int = 5):
        return torch.utils.data.DataLoader(
            self,
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    def calc_datas(self):
        datas = []

        for sampled_graph in tqdm(self.graphs, desc="Calculate stats"):
            graph_degrees = np.array([d for _, d in sampled_graph.degree()])
            graph_degree_distribution = torch.Tensor(
                [
                    (graph_degrees == d).sum() / len(sampled_graph)
                    for d in range(len(sampled_graph))
                ]
            )
            graph_scaled_degrees = torch.Tensor(
                [degree / len(sampled_graph) for degree in graph_degrees]
            )

            graph_homomorphism_densities = [
                DatasetMC.estimate_naive_monte_carlo_homomorphism_density(
                    graphlet.graph.to_nx(),
                    sampled_graph,
                    epsilon=0.001,
                    gamma=0.95,
                )
                for graphlet in self.graphlets
            ]

            datas.append(
                (
                    sampled_graph,
                    graph_homomorphism_densities,
                    graph_degree_distribution,
                    graph_scaled_degrees,
                )
            )

        return datas

    def __iter__(self):
        if not self.datas:
            self.datas = self.calc_datas()

        for data in self.datas:
            (
                _,
                graph_homomorphism_densities,
                graph_degree_distribution,
                graph_scaled_degrees,
            ) = data

            indices = list(range(len(self.graphlets)))
            random.shuffle(indices)

            for i in indices:
                n_samples = 1000
                vertices, edges = self.graphlets[i].graph

                # Uniformly sample unit hyper-cube n times
                sampled_vertices = np.random.uniform(size=n_samples * len(vertices))

                # Map vertices to real numbers from <0,1> n times
                mapping_indices = DatasetBase.mapping_indices(
                    vertices, edges, n_samples
                )
                mapped_edges = sampled_vertices[mapping_indices]

                x = torch.from_numpy(mapped_edges[:, 0]).float()
                y = torch.from_numpy(mapped_edges[:, 1]).float()

                yield (
                    (self.graphlets[i].name,),
                    (
                        (x, y),
                        len(edges),
                        -1,
                        -1,
                        graph_homomorphism_densities[i],
                        n_samples,
                    ),
                    (graph_degree_distribution, graph_scaled_degrees),
                )


class DatasetBase2(torch.utils.data.Dataset):
    def __init__(
        self,
        graphs: t.List[t.Union[nx.Graph, nx.DiGraph]],
        stats: bool,
        n_graphlets: int = 4,
        resolution: int = 100,
        base_path: t.Optional[str] = None,
        name: t.Optional[str] = None,
    ):
        assert len(set(g.__class__.__name__ for g in graphs)) == 1

        self.graphs = graphs
        self.directed = any(isinstance(g, nx.DiGraph) for g in self.graphs)
        self.stats = stats
        self.n_graphlets = n_graphlets
        self.resolution = resolution
        self.name = name

        self.graphlets = [
            Graphlet(
                INDEX_TO_PGD_GRAPHLET_NAME.get(i, f"graphlet_{i}"),
                g,
                tools.complement_graph(*g),
            )
            for i, g in enumerate(
                tqdm(tools.create_all_graphs(self.n_graphlets), desc="Graphlets")
            )
        ]

        if base_path is not None:
            self.base_path = base_path
            Path(self.base_path).mkdir(parents=True, exist_ok=True)

        if self.stats:
            self.graphs_homomorphism_densities = [
                torch.Tensor(
                    [
                        DatasetMC.estimate_naive_monte_carlo_homomorphism_density(
                            graphlet.graph.to_nx(),
                            graph,
                            epsilon=0.001,
                            gamma=0.95,
                        )
                        for graphlet in self.graphlets
                    ]
                )
                for graph in tqdm(self.graphs, "HDs")
            ]

            self.graphs_degrees = [
                np.array([d for _, d in graph.degree()]) for graph in self.graphs
            ]

            self.graphs_degree_distributions = [
                torch.Tensor(
                    [(graph_degrees == d).sum() / len(graph) for d in range(len(graph))]
                )
                for graph, graph_degrees in zip(self.graphs, self.graphs_degrees)
            ]

            self.graphs_scaled_degrees = [
                torch.Tensor([degree / len(graph_degrees) for degree in graph_degrees])
                for graph_degrees in self.graphs_degrees
            ]

            assert len(self.graphs) == len(self.graphs_homomorphism_densities)
            assert len(self.graphs) == len(self.graphs_degrees)
            assert len(self.graphs) == len(self.graphs_degree_distributions)
            assert len(self.graphs) == len(self.graphs_scaled_degrees)
            assert all(
                len(g) == len(gd) for g, gd in zip(self.graphs, self.graphs_degrees)
            )
            assert all(
                len(self.graphlets) == len(x)
                for x in self.graphs_homomorphism_densities
            )

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx: int):
        hds = []

        if self.stats:
            for graphlet, homomorphism_density in zip(
                self.graphlets, self.graphs_homomorphism_densities[idx]
            ):
                n_samples = self.resolution
                vertices, edges = graphlet.graph

                # Uniformly sample unit hyper-cube n times
                sampled_vertices = np.random.uniform(size=n_samples * len(vertices))

                # Map vertices to real numbers from <0,1> n times
                mapping_indices = DatasetBase.mapping_indices(
                    vertices, edges, n_samples
                )
                mapped_edges = torch.from_numpy(
                    sampled_vertices[mapping_indices]
                ).float()

                hds.append(
                    (
                        mapped_edges,
                        len(edges),
                        homomorphism_density,
                        n_samples,
                    )
                )

        adj_matrix = np.array(
            nx.convert_matrix.to_numpy_matrix(self.graphs[idx]), dtype=np.float32
        )

        return (
            tuple(hds) if self.stats else -1,
            isomorphic_adj_matrix(adj_matrix),
            self.graphs_degree_distributions[idx] if self.stats else -1,
            self.graphs_scaled_degrees[idx] if self.stats else -1,
        )

    def get_dataloader(
        self,
        num_workers: int = 5,
        batch_size: int = 5,
        persistent_workers: t.Optional[bool] = None,
    ):
        batch_size = min(len(self), batch_size)
        assert (
            len(self) % batch_size == 0
        ), f"`batch_size` {batch_size} does not equally divide this dataset"

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0
            if persistent_workers is None
            else persistent_workers,
        )


class DatasetGraphon2(DatasetBase2):
    def __init__(
        self,
        graphon_func: t.Callable,
        n_sampled_graphs: int,
        sample_size: int,
        **kwargs,
    ):
        self.graphon_func = graphon_func

        graphs = [
            self.sample(self.graphon_func, sample_size)
            for _ in tqdm(range(n_sampled_graphs), desc="Sampling")
        ]

        super().__init__(
            graphs=graphs,
            name=graphon_func.__name__,
            **kwargs,
        )

        self.graphon_func = graphon_func
        self.graphon_grid = tools.create_image_mat(self.graphon_func, 1000)
        self.base_path = f"data/artificial/{self.graphon_func.__name__}"
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

        tools.visualize(self.graphon_func, f'{self.base_path}/graphon.jpg', resolution=150)

    @staticmethod
    def sample(graphon, n_vertices: int):
        """Returns a random graph sampled from graphon W."""

        # Randomly select nodes in graphon
        V = np.random.uniform(size=n_vertices)

        # All possible edges
        edge_indices = np.array(list(it.combinations(list(range(len(V))), 2)))
        edges = V[edge_indices]

        # Edge probabilites (graphon values)
        edge_probs = graphon(edges[:, 0], edges[:, 1])

        # Convert float values of vertices to integer indexes
        V = np.arange(len(V))

        # Threshold probabilites to decide which pairs become edges
        E = np.random.uniform(size=len(edge_probs)) <= edge_probs
        E = edge_indices[E]

        return tools.NPGraph(V, E).to_nx(
            graph_class=nx.Graph
        )


class DatasetTUDataset2(DatasetBase2):
    def __init__(
        self,
        name: str,
        allow_directed: bool,
        class_idx: t.Optional[int] = None,
        limit_graphs: t.Optional[int] = None,
        max_graph_size: t.Optional[int] = None,
        _graphs: t.Optional[list] = None,
        **kwargs,
    ):
        self.limit_graphs = limit_graphs
        self.max_graph_size = max_graph_size

        if _graphs is None:
            dataset = tg.datasets.TUDataset(
                name=name, root="data/tudataset", cleaned=False
            )
            dataset.shuffle()

            graphs = []

            for data in tqdm(dataset, "Parsing TUDataset"):
                # label = data.y.item()
                graph = tg.utils.to_networkx(
                    data,
                    to_undirected=not allow_directed or data.is_undirected(),
                )

                if class_idx is not None and data.y.item() != class_idx:
                    continue

                if max_graph_size is None or len(graph) <= max_graph_size:
                    graphs.append(graph)

                if limit_graphs is not None and len(graphs) == limit_graphs:
                    break

            if not graphs:
                raise RuntimeError("No graphs found.")

        else:
            graphs = _graphs

        super().__init__(
            graphs=graphs,
            name=name,
            **kwargs,
        )

        self.base_path = f"data/tudataset/{self.name}"
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def split(
        name: str,
        class_idx: t.Optional[int],
        train_size: float,
        stats: bool,
        limit_graphs: int,
        max_graph_size: t.Optional[int],
        allow_directed: bool,
    ):
        base = DatasetTUDataset2(
            name=name,
            class_idx=class_idx,
            allow_directed=allow_directed,
            stats=False,
            limit_graphs=limit_graphs,
            max_graph_size=max_graph_size,
        )

        index = int(len(base) * train_size)
        train, val = base.graphs[:index], base.graphs[index:]

        return (
            DatasetTUDataset2(name=name, allow_directed=allow_directed, _graphs=train, stats=stats),
            DatasetTUDataset2(name=name, allow_directed=allow_directed, _graphs=val, stats=False),
        )


class DatasetTUDataset(DatasetGraphon):
    def __init__(self, _graphs, base_path):
        def dummy():
            ...

        super().__init__(
            graphon_func=dummy,
            _graphs=_graphs,
        )

        self.base_path = base_path

    @staticmethod
    def split(
        name: str,
        train_size: float,
        limit_graphs: int,
    ):
        base = DatasetTUDataset2(
            name=name,
            stats=False,
            limit_graphs=limit_graphs,
        )

        index = int(len(base) * train_size)
        train, val = base.graphs[:index], base.graphs[index:]

        return (
            DatasetTUDataset(_graphs=train, base_path=base.base_path),
            DatasetTUDataset(_graphs=val, base_path=base.base_path),
        )


class InducedGraphsDataset:
    @staticmethod
    def split(
        graph_file: str,
        n_induced: int,
        induced_size_range: t.Tuple[int, int],
        stats: bool,
    ):
        assert len(induced_size_range) == 2
        assert 0 < induced_size_range[0] <= induced_size_range[1]

        split = graph_file.split("/")
        base_path = "/".join(split[:-1])
        name = split[-1]

        graph = nx.convert_node_labels_to_integers(
            tools.nx_from_file(graph_file), first_label=0
        )
        assert induced_size_range[0] < len(graph)
        assert induced_size_range[1] < len(graph)

        # Split graph to two.
        graph_split = graph_file + '.graphsplit'

        if not os.path.isfile(graph_split):
            first_half_nodes = random.sample(graph.nodes(), k=int(len(graph) * 0.50))
            second_half_nodes = [
                node for node in graph.nodes()
                if node not in first_half_nodes
            ]

            train_g, test_g = graph.subgraph(first_half_nodes), graph.subgraph(second_half_nodes)

            with open(graph_split, 'wb') as fw:
                pickle.dump((train_g, test_g), fw)

        with open(graph_split, 'rb') as fr:
            train_g, test_g = pickle.load(fr)

        induced_graphs = []
        for _ in range(n_induced):
            nodes = random.sample(train_g.nodes(), k=random.randint(*induced_size_range))
            ind_graph = nx.Graph(train_g.subgraph(nodes))

            if not ind_graph.number_of_edges():
                continue

            nodes_with_edge = set()

            for a, b in ind_graph.edges():
                nodes_with_edge.add(a)
                nodes_with_edge.add(b)

            ind_graph.remove_nodes_from(
                [n for n in ind_graph.nodes() if n not in nodes_with_edge]
            )

            induced_graphs.append(ind_graph)

        print(f"Induced {len(induced_graphs)} graphs.")

        return DatasetBase2(
            graphs=induced_graphs,
            stats=stats,
            base_path=base_path,
            resolution=200,
            name=name,
        ), DatasetBase2(
            graphs=[test_g, test_g],
            stats=False,
            base_path=base_path,
            resolution=200,
            name=name,
        )
