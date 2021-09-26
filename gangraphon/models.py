import ot
import time
import torch
import json
import pickle
import random
import torchmetrics
import numpy as np
import typing as t
import networkx as nx
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch_geometric as tg

from pathlib import Path
from functools import lru_cache, partial
from collections import defaultdict
from common import tools, functions

try:
    from apex import amp

except ImportError:
    print("Warning: APEX not installed")

try:
    from common.evaluators.eval import eval_tdas

    EVALUATE_TDA = True
except ImportError:
    EVALUATE_TDA = False
    print("Warning: TDA code not found.")


def rescale(
    value: float,
    /,
    *,
    original_from: float,
    original_to: float,
    new_from: float,
    new_to: float,
) -> float:
    return (value - original_from) * (new_to - new_from) / (
        original_to - original_from
    ) + new_from


rescale_tanh_to_sigmoid = partial(
    rescale, original_from=-1.0, original_to=1.0, new_from=0.0, new_to=1.0
)


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)


def torch_isomorphic_adj_matrix(batch: torch.Tensor) -> torch.Tensor:
    assert batch.dim() == 3 and batch.size(1) == batch.size(
        2
    ), "[batch_size, n_nodes, n_nodes]"

    indexes = list(range(batch.size(1)))
    random.shuffle(indexes)
    indexes_long = torch.LongTensor(indexes).to(batch.device)

    return torch.index_select(
        torch.index_select(batch, 2, indexes_long), 1, indexes_long
    )


class GNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gnn_layer: str,
        allow_floats_adj: bool,
        depth: int,
    ):
        super().__init__()
        self.allow_floats_adj = allow_floats_adj

        self.input_layer = getattr(tg.nn, gnn_layer)(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.hidden_layers = torch.nn.ModuleList(
            [
                getattr(tg.nn, gnn_layer)(
                    in_channels=out_channels,
                    out_channels=out_channels,
                )
                for _ in range(depth - 1)
            ]
        )

    def forward(self, node_features, adj_matrix, edge_features=None):
        assert node_features.dim() == 3 and node_features.size(1) == adj_matrix.size(
            1
        ), "[batch_size, n_nodes, n_feat]"
        assert adj_matrix.dim() == 3 and adj_matrix.size(1) == adj_matrix.size(
            2
        ), "[batch_size, n_nodes, n_nodes]"
        assert self.allow_floats_adj or bool(
            ((adj_matrix == 0) | (adj_matrix == 1)).all()
        ), "Adj matrix should be zeros and ones"

        # # TODO: Fix batching
        # assert adj_matrix.shape[0] == 1 == node_features.shape[0]
        # node_features = node_features[0]
        # edge_index = tools.adj_matrix_to_edge_index(adj_matrix[0]).to(node_features.device)

        # if edge_features:
        #     assert edge_features.shape[0] == edge_index.shape[1]

        x = self.input_layer(node_features, adj_matrix)  # , edge_attr=edge_features)

        for layer in self.hidden_layers:
            x = layer(x, adj_matrix)  # , edge_attr=edge_features)

        # # TODO: Fix batching
        # x = x.unsqueeze(0)

        return x


class Discriminator(torch.nn.Module):
    def __init__(self, use_gnn: bool, graph_features: int, allow_floats_adj: bool, gnn_layer: str):
        super().__init__()

        self.use_gnn = use_gnn
        self.graph_features = graph_features
        self.allow_floats_adj = allow_floats_adj
        self.head_input_dim = 0

        if self.use_gnn:
            self.gnn = GNN(
                in_channels=1,
                out_channels=32,
                gnn_layer=gnn_layer,
                allow_floats_adj=self.allow_floats_adj,
                depth=2,
            )
            self.head_input_dim += 32

        if self.graph_features:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(graph_features, 64),
                torch.nn.LeakyReLU(negative_slope=0.01),
                torch.nn.Linear(64, 32),
                torch.nn.LeakyReLU(negative_slope=0.01),
            )
            self.head_input_dim += 32

        assert self.head_input_dim, "Discriminator needs at least some inputs."

        self.head = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.head_input_dim, 1),
            # torch.nn.Sigmoid(),  # Using `binary_cross_entropy_with_logits`
        )

    def forward(
        self,
        adj_matrix: t.Optional[torch.Tensor],
        graph_features: t.Optional[torch.Tensor],
    ):
        x = []

        if adj_matrix is not None:
            assert adj_matrix.dim() == 3 and adj_matrix.size(1) == adj_matrix.size(
                2
            ), "[batch_size, n_nodes, n_nodes]"
            node_feats = (
                (adj_matrix.sum(1) + adj_matrix.sum(2)) / adj_matrix.size(1)
            ).unsqueeze(-1)
            gnn_x = self.gnn(
                node_feats, adj_matrix
            )
            gnn_x = torch.nn.functional.leaky_relu(
                torch.mean(gnn_x, dim=1), negative_slope=0.01
            )
            x.append(gnn_x)

        if graph_features is not None:
            mlp_x = self.mlp(graph_features)
            x.append(mlp_x)

        x = torch.cat(x, dim=1)
        x = self.head(x)

        return x


class Generator(torch.nn.Module):
    def __init__(
        self,
        directed: bool,
        output_dim: int,
        hidden_size: int = 256,
        depth: int = 1,
    ):
        super().__init__()
        assert output_dim in (1, 2)

        self.directed = directed
        self.output_dim = output_dim
        self.model = torch.nn.Sequential(
            functions.Block(input_dim=2, hidden_size=hidden_size),
            *[functions.Block(hidden_size, hidden_size) for _ in range(depth)],
            torch.nn.Linear(hidden_size, output_dim),
            torch.nn.Softmax(dim=-1) if output_dim == 2 else torch.nn.Sigmoid(),
        )

    def forward(self, edges):
        assert (
            edges.dim() == 3 and edges.size(2) == 2
        ), "[batch_size, n_edges, <from, to>]"

        x, _ = torch.sort(edges) if not self.directed else (edges, None)

        x = x.view(-1, 2)
        # Inter-Batch so we can generate adj. matrix of huge graphs
        x = torch.vstack(
            [self.model(x[i : i + (150 ** 2)]) for i in range(0, x.size(0), (150 ** 2))]
        )
        x = (
            x.view(edges.size(0), edges.size(1), 2)
            if self.output_dim == 2
            else x.view(edges.size(0), edges.size(1))
        )

        return x


class Gumbel(torch.nn.Module):
    """
    Modified from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    """

    def __init__(self, tau: float, trainable_tau: bool = True):
        super().__init__()
        self.tau = (
            torch.nn.Parameter(torch.Tensor([tau]), requires_grad=True)
            if trainable_tau
            else tau
        )

    def forward(self, logits: torch.Tensor, hard: bool, dim: int = -1):
        log_logits = torch.log(logits)

        gumbels = (
            -torch.empty_like(log_logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels = (log_logits + gumbels) / self.tau

        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(
                logits, memory_format=torch.legacy_contiguous_format
            ).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft

        else:
            # Reparametrization trick.
            ret = y_soft

        return ret


def sigmoid_trick(batch_probs_adj_matrix: torch.Tensor):
    assert ((batch_probs_adj_matrix >= 0) | (batch_probs_adj_matrix <= 1)).all()

    discrete = (
        torch.rand(*batch_probs_adj_matrix.shape).to(batch_probs_adj_matrix.device)
        <= batch_probs_adj_matrix
    ).float()
    return discrete - batch_probs_adj_matrix.detach() + batch_probs_adj_matrix


class GAN(pl.LightningModule):
    def __init__(
        self,
        name: str,
        gv_probs: dict,
        graph_features: int,
        use_gnn: bool,
        use_hds_gnn: bool,
        use_hds_sqr: bool,
        use_dd_sqr: bool,
        directed: bool,
        gumbel: bool,
        gumbel_hard: bool,
        gnn_layer: str,
        tau_value: float,
        trainable_tau: bool,
        amp: bool,
        self_loops: bool,
        sigmoid_trick: bool = False,
        generator_kwargs: t.Optional[dict] = None,
        g_optimizer: str = "Adam",
        g_optimizer_kwargs: t.Optional[dict] = dict(lr=1e-5),
        d_optimizer: str = "Adam",
        d_optimizer_kwargs: t.Optional[dict] = dict(lr=1e-5),
        discriminator_every_nth_batch: int = 1,
        dataset: t.Optional[object] = None,
        visualize_every_nth_epoch: int = 1,
        tda_every_nth_epoch: int = 1,
        tda_metrics: t.Optional[t.List[str]] = None,
        early_stop: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])
        assert sum([gumbel, sigmoid_trick]) != 2

        self.dataset = dataset

        if self.hparams.gumbel:
            self.gumbel = Gumbel(tau=tau_value, trainable_tau=trainable_tau)

        self.generator = Generator(
            directed=directed,
            output_dim=2 if not sigmoid_trick else 1,
            **(generator_kwargs or {}),
        )
        self.discriminator = Discriminator(
            use_gnn=self.hparams.use_gnn,
            graph_features=graph_features if any([self.hparams.use_hds_gnn, ]) else 0,
            allow_floats_adj=not self.hparams.gumbel,
            gnn_layer=gnn_layer,
        )

        self.hds_sqr_multiplier = None
        self.best_tda_metrics = defaultdict(
            lambda: {"epoch": None, "value": float("inf"), "took_min": None}
        )
        self.best_mse_metric = float('inf')
        self.best_mse_err_metric = float('inf')
        self.best_gw_metric = float('inf')
        self.init_time = time.time()
        self.early = 0

    @lru_cache(maxsize=None)
    def possible_edges(self, n_vertices: int) -> torch.LongTensor:
        '''
        This will return edge list in such a way,
        that edge_list.reshape(n, n) will be correct adj. matrix.
        '''

        edges = []

        for i in range(n_vertices):
            for j in range(n_vertices):
                edges.append([i, j])

        return torch.LongTensor(edges)

    @lru_cache(maxsize=None)
    def list_range(self, n_vertices: int) -> t.List[int]:
        return list(range(n_vertices))

    def forward(self, zbatch):
        n_vertices = zbatch.size(1)

        idx = self.possible_edges(n_vertices=n_vertices).to(self.device)

        x = torch.stack(
            [
                torch.hstack([z[idx[:, 0]].unsqueeze(1), z[idx[:, 1]].unsqueeze(1)])
                for z in zbatch
            ]
        )

        x = self.generator(x)
        if self.hparams.gumbel:
            x = self.gumbel(x, hard=self.hparams.gumbel_hard)
        if self.hparams.sigmoid_trick:
            x = sigmoid_trick(x)
        if not self.hparams.sigmoid_trick:
            x = x[:, :, 1]
        x = x.view(zbatch.size(0), zbatch.size(1), zbatch.size(1))
        if not self.hparams.self_loops:
            x[:, self.list_range(n_vertices), self.list_range(n_vertices)] = 0

        return x

    def adversarial_loss(self, y_hat, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)

    def degree_distribution(self, batch):
        pred_dists = []

        for true_degree_dist in batch:
            pred_dist = functions.degree_distribution(
                self.generator, len(true_degree_dist), resolution=100
            )
            pred_dists.append(pred_dist.unsqueeze(0))

        pred_dists = torch.cat(pred_dists)

        return pred_dists

    def degree_distribution_loss(self, batch):
        true_dists = torch.cat([td.unsqueeze(0) for td in batch])
        pred_dists = self.degree_distribution(batch)

        loss = ((pred_dists - true_dists) ** 2).sum(dim=1).mean()

        self.log("train/g_degdist_loss", loss, prog_bar=True)

        return loss

    def homomorphism_density(self, batch):
        pred_hds = []

        for sampled_edges, n_true_edges, true_hd, n_samples in batch:
            gen = self.generator(sampled_edges)
            if not self.hparams.sigmoid_trick:
                gen = gen[:, :, 1]
            probs = gen.reshape(sampled_edges.size(0), n_samples[0], n_true_edges[0])
            pred_hd = probs.prod(dim=2).mean(dim=1).unsqueeze(0)
            pred_hds.append(pred_hd)

        pred_hds = torch.cat(pred_hds)

        return pred_hds

    def homomorphism_density_loss(self, batch):
        true_hds = torch.cat([true_hd.unsqueeze(0) for _, _, true_hd, _ in batch])
        pred_hds = self.homomorphism_density(batch)

        loss = ((pred_hds - true_hds) ** 2).sum(dim=1).mean()

        self.log("train/g_hd_loss", loss, prog_bar=True)

        return loss

    def training_step_generator(self, z, graph_batch, hds_batch):
        valid_labels = torch.ones(graph_batch.size(0), 1).to(self.device)
        valid_adjs = (
            torch_isomorphic_adj_matrix(self(z)) if self.hparams.use_gnn else None
        )
        valid_hds = (
            self.homomorphism_density(hds_batch).transpose(0, 1)
            if self.hparams.use_hds_gnn
            else None
        )

        disc_output = self.discriminator(valid_adjs, valid_hds)
        g_loss = self.adversarial_loss(disc_output, valid_labels)

        if self.hparams.gumbel:
            self.log("train/tau", self.gumbel.tau, prog_bar=True)
        self.log("train/g_gendis_loss", g_loss, prog_bar=True)

        return g_loss

    def training_step_discriminator(self, z, graph_batch, hds_batch):
        # How well can it label as real?
        real_adjs = graph_batch if self.hparams.use_gnn else None
        real_hds = (
            torch.cat(
                [true_hd.unsqueeze(0) for _, _, true_hd, _ in hds_batch]
            ).transpose(0, 1)
            if self.hparams.use_hds_gnn
            else None
        )
        true_real_labels = torch.ones(graph_batch.size(0), 1).to(self.device)
        pred_real_labels = self.discriminator(real_adjs, real_hds)
        real_loss = self.adversarial_loss(pred_real_labels, true_real_labels)

        # How well can it label as fake?
        fake_adjs = self(z).detach() if self.hparams.use_gnn else None
        fake_hds = (
            self.homomorphism_density(hds_batch).transpose(0, 1).detach()
            if self.hparams.use_hds_gnn
            else None
        )
        true_fake_labels = torch.zeros(graph_batch.size(0), 1).to(self.device)
        pred_fake_labels = self.discriminator(fake_adjs, fake_hds)
        fake_loss = self.adversarial_loss(pred_fake_labels, true_fake_labels)

        # Discriminator loss is the average of real and fake
        d_loss = (real_loss + fake_loss) / 2

        # Metrics
        self.log("train/d_loss", d_loss, prog_bar=True)
        self.log(
            "train/d_accuracy/all",
            torchmetrics.functional.classification.accuracy(
                torch.cat([pred_real_labels, pred_fake_labels]),
                torch.cat([true_real_labels, true_fake_labels]).long(),
            ),
        )
        self.log(
            "train/d_accuracy/real",
            torchmetrics.functional.classification.accuracy(
                pred_real_labels, true_real_labels.long()
            ),
        )
        self.log(
            "train/d_accuracy/fake",
            torchmetrics.functional.classification.accuracy(
                pred_fake_labels, true_fake_labels.long()
            ),
        )

        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        hds_batch, graph_batch, degree_dist_batch, scaled_degrees_batch = batch
        assert len(graph_batch.shape) == 3 and graph_batch.size(1) == graph_batch.size(
            2
        )

        z = torch.rand(graph_batch.size(0), graph_batch.size(1)).to(self.device)

        if optimizer_idx == 0:
            loss = self.training_step_generator(
                z=z, graph_batch=graph_batch, hds_batch=hds_batch
            )

            if self.hparams.use_hds_sqr:
                if self.hds_sqr_multiplier is not None:
                    loss += (
                        self.homomorphism_density_loss(hds_batch)
                        * self.hds_sqr_multiplier
                    )

                else:
                    self.hds_sqr_multiplier = max(
                        1.0,
                        (loss / self.homomorphism_density_loss(hds_batch))
                        .detach()
                        .cpu()
                        .item(),
                    )
                    print(
                        f"`self.hds_sqr_multiplier` set to {self.hds_sqr_multiplier}."
                    )

                if self.hparams.use_dd_sqr:
                    loss += self.degree_distribution_loss(degree_dist_batch)

            return loss

        elif optimizer_idx == 1:
            if not batch_idx % self.hparams.discriminator_every_nth_batch:
                return self.training_step_discriminator(
                    z=z, graph_batch=graph_batch, hds_batch=hds_batch
                )

            else:
                return None

        else:
            raise RuntimeError(f"Invalid optimizer_idx {optimizer_idx}.")

    def backward(
        self,
        loss: torch.Tensor,
        optimizer: t.Optional[torch.optim.Optimizer],
        optimizer_idx: t.Optional[int],
        *args,
        **kwargs,
    ):
        if optimizer_idx not in (0, 1):
            raise RuntimeError(f"Invalid optimizer_idx {optimizer_idx}.")

        if self.hparams.amp:
            with amp.scale_loss(loss, optimizer, loss_id=optimizer_idx) as loss_scaled:
                loss_scaled.backward(*args, **kwargs)
        else:
            super().backward(loss=loss, optimizer=optimizer, optimizer_idx=optimizer_idx, *args, **kwargs)

    def configure_optimizers(self):
        optim_generator = getattr(torch.optim, self.hparams.g_optimizer)(
            [{"params": self.generator.parameters()}]
            + (
                [{"params": self.gumbel.parameters(), "lr": 1e-1}]
                if self.hparams.gumbel
                else []
            ),
            **(self.hparams.g_optimizer_kwargs or {}),
        )
        optim_discriminator = getattr(torch.optim, self.hparams.d_optimizer)(
            self.discriminator.parameters(),
            **(self.hparams.d_optimizer_kwargs or {}),
        )

        if self.hparams.amp:
            (self.discriminator, self.generator), (
                optim_generator,
                optim_discriminator,
            ) = amp.initialize(
                [self.discriminator, self.generator],
                [optim_generator, optim_discriminator],
                opt_level="O1",
                num_losses=2,
            )

            scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_generator,
                verbose=True,
                factor=0.9,
                patience=5,
                cooldown=1,
            )
            scheduler_discriminator = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim_discriminator,
                verbose=True,
                factor=0.9,
                patience=5,
                cooldown=1,
            )

            return {
                "optimizer": optim_generator,
                "lr_scheduler": {
                    "scheduler": scheduler_generator,
                    "monitor": "train/g_gendis_loss",
                },
            }, {
                "optimizer": optim_discriminator,
                "lr_scheduler": {
                    "scheduler": scheduler_discriminator,
                    "monitor": "train/d_loss",
                },
            }

        else:
            return optim_generator, optim_discriminator

    @torch.no_grad()
    @tools.torch_eval
    def sample(
        self,
        n_vertices: t.Optional[int] = None,
    ) -> nx.Graph:
        vertices = torch.rand(
            n_vertices
            or random.choices(
                list(self.hparams.gv_probs.keys()),
                weights=list(self.hparams.gv_probs.values()),
                k=1,
            )[0],
        )

        adj_matrix = (
            self(vertices.unsqueeze(0).to(self.device)).squeeze(0).cpu().numpy()
        )
        assert len(adj_matrix.shape) == 2 and adj_matrix.shape[0] == adj_matrix.shape[1]

        if not self.hparams.gumbel:
            adj_matrix = np.random.rand(*adj_matrix.shape) <= adj_matrix

        adj_matrix = adj_matrix.astype(int)

        return nx.convert_matrix.from_numpy_matrix(
            adj_matrix, create_using=nx.DiGraph if self.hparams.directed else nx.Graph
        )

    def on_epoch_end(self):
        self.eval_graphon()

    # def on_fit_end(self):
    #     self.eval_graphon(force=True)

    @tools.torch_eval
    @torch.no_grad()
    def eval_graphon(self, batch_idx: t.Optional[int] = None, force: bool = False):
        path = Path(
            f"{self.dataset.base_path}/training/{self.hparams.name}/{self.__class__.__name__}/epoch={self.current_epoch}"
        )

        if self.current_epoch > 1:
            path.mkdir(exist_ok=True, parents=True)
            self.eval_mse(path)

        if (
            force
            or self.current_epoch == 1
            or not self.current_epoch % self.hparams.tda_every_nth_epoch
        ):
            path.mkdir(exist_ok=True, parents=True)
            self.eval_tda(path=path, save_graphs=not self.current_epoch % 10)

        if (
            force
            or self.current_epoch == 1
            or not self.current_epoch % self.hparams.visualize_every_nth_epoch
        ):
            path.mkdir(exist_ok=True, parents=True)
            self.visualize(path=path)

    @tools.torch_eval
    @torch.no_grad()
    def visualize(self, path: t.Union[str, Path], resolution: int = 100):
        def model(x, y):
            edges = tools.preprocess_vertices(x, y).unsqueeze(0).to(self.device)
            out = self.generator(edges)
            if not self.hparams.sigmoid_trick:
                out = out[:, :, 1]
            grid = out.cpu().view(resolution, resolution).numpy()
            return grid

        tools.visualize(model, f"{path}/graphon.png", resolution=resolution)
        tools.visualize(
            model, f"{path}/graphon.norm.png", resolution=resolution, normalize=True
        )

        # G = self.sample(100)
        # pos = nx.spring_layout(G, iterations=100, seed=0)
        # nx.draw(G, pos)
        # plt.savefig(f"{path}/graph.png", format="PNG")
        # plt.clf()

    @tools.torch_eval
    @torch.no_grad()
    def eval_tda(self, path: t.Union[str, Path], save_graphs: bool = False):
        if EVALUATE_TDA:
            graphs = [self.sample() for _ in range(len(self.dataset.graphs))]

            if save_graphs:
                with open(f"{path}/graphs.pickle", "wb") as f:
                    pickle.dump(graphs, f)

            evals = eval_tdas(
                self.dataset.graphs, graphs, only=self.hparams.tda_metrics
            )

            with open(f"{path}/graphon.tda.n_samples={len(graphs)}.json", "w") as f:
                json.dump(evals, f, indent=2)

            any_new_best = False

            for name, value in evals.items():
                # TDA are distances, we need to minimize
                if value < self.best_tda_metrics[name]["value"]:
                    self.best_tda_metrics[name]["value"] = value
                    self.best_tda_metrics[name]["epoch"] = self.current_epoch
                    self.best_tda_metrics[name]["took_min"] = (
                        time.time() - self.init_time
                    ) / 60

                    any_new_best = True

            if any_new_best:
                self.early = 0
                print(f"Best TDA metrics: {dict(self.best_tda_metrics)}")

            else:
                self.early += 1

            if self.early == self.hparams.early_stop:
                self.trainer.should_stop = True

            with open(
                f"{path}/graphon.tda.best.n_samples={len(graphs)}.json", "w"
            ) as f:
                json.dump(dict(self.best_tda_metrics), f, indent=2)

    @tools.torch_eval
    @torch.no_grad()
    def eval_mse(self, path: t.Union[str, Path]):
        if not hasattr(self.dataset, 'graphon_grid'):
            return

        def model(x, y):
            edges = tools.preprocess_vertices(x, y).unsqueeze(0).to(self.device)
            out = self.generator(edges)
            if not self.hparams.sigmoid_trick:
                out = out[:, :, 1]
            grid = out.cpu().view(1000, 1000).numpy()
            return grid

        graphon = tools.create_image_mat(model, 1000)
        orig_graphon = self.dataset.graphon_grid

        msek = {}
        gwk = {}

        graphons = [
            np.rot90(graphon, k) for k in [1, 2, 3, 4]
        ] + [np.flip(graphon, axis=0), np.flip(graphon, axis=1)]

        for i, graphon in enumerate(graphons):
            mse = mean_square_error(orig_graphon, graphon)
            err = relative_error(orig_graphon, graphon)
            gw = gw_distance(orig_graphon, graphon)

            msek[i] = {
                'mse': mse,
                'err': err,
            }
            gwk[i] = {'gw': gw}

            if mse < self.best_mse_metric:
                self.best_mse_metric = mse
                self.best_mse_err_metric = err

            if gw < self.best_gw_metric:
                self.best_gw_metric = gw

        with open(f"{path}/graphon_gw.json", "w") as f:
            json.dump(gwk, f, indent=2)

        with open(f"{path}/graphon_mse.json", "w") as f:
            json.dump(msek, f, indent=2)

        with open(f"{path}/graphon_gw.best.json", "w") as f:
            json.dump({'gw': self.best_gw_metric}, f, indent=2)

        with open(f"{path}/graphon_mse.best.json", "w") as f:
            json.dump({'mse': self.best_mse_metric, 'err': self.best_mse_err_metric}, f, indent=2)


# Credits: https://github.com/HongtengXu/SGWB-Graphon/blob/master/methods/simulator.py#L109
def gw_distance(graphon: np.ndarray, estimation: np.ndarray) -> float:
    p = np.ones((graphon.shape[0],)) / graphon.shape[0]
    q = np.ones((estimation.shape[0],)) / estimation.shape[0]
    loss_fun = 'square_loss'
    dw2 = ot.gromov.gromov_wasserstein2(graphon, estimation, p, q, loss_fun, log=False, armijo=False)
    return float(np.sqrt(dw2))


# Credits: https://github.com/HongtengXu/SGWB-Graphon/blob/master/methods/simulator.py#L117
def mean_square_error(graphon: np.ndarray, estimation: np.ndarray) -> float:
    return float(np.linalg.norm(graphon - estimation))


# Credits: https://github.com/HongtengXu/SGWB-Graphon/blob/master/methods/simulator.py#L121
def relative_error(graphon: np.ndarray, estimation: np.ndarray) -> float:
    return float(np.linalg.norm(graphon - estimation) / np.linalg.norm(graphon))
