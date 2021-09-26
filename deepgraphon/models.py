import math
import torch
import json
import random
import typing as t
import numpy as np
import networkx as nx
import itertools as it
import pytorch_lightning as pl

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from common import tools, tests, functions
from common.data import DatasetBase
from common.functions import scaled_degree, degree_distribution, torch_eval, CACHE

try:
    from common.evaluators.eval import eval_tdas

    EVALUATE_TDA = True
except ImportError:
    EVALUATE_TDA = False
    print("Warning: TDA code not found.")


class DeepGraphonBase(pl.LightningModule):
    def __init__(
        self,
        name: str,
        use_homomorphism_density_loss: bool,
        use_homomorphism_density_non_edges: bool,
        use_degree_distribution_loss: bool,
        use_degree_stats_loss: bool,
        use_entropy_loss: bool,
        gv_probs: t.Optional[None] = None,
        tda_metrics: t.Optional[t.List[str]] = None,
        evaluate_tda: bool = True,
        evaluate_networkx: bool = False,
        optimizer: str = "Adam",
        lr: float = 1e-3,
        dataset: t.Optional[DatasetBase] = None,
        early_stop: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="dataset")

        self.reset_weight_losses()

        self.dataset = dataset
        self.none_batch_count = 0
        self.best_eval_tdas = defaultdict(lambda: float("inf"))
        self.early = 0
        self.skipping_step = None

    def reset_weight_losses(self):
        self.loss_graphlets = set()

        self.degree_stats_losses = []
        self.degree_distribution_losses = []
        self.homomorphism_density_losses = []
        self.entropy_losses = []

        self.degree_stats_loss_weight = None
        self.degree_distribution_loss_weight = None
        self.homomorphism_density_loss_weight = None
        self.entropy_loss_weight = None

    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_eval_tdas"] = dict(self.best_eval_tdas)
        checkpoint["loss_graphlets"] = self.loss_graphlets
        checkpoint["degree_stats_losses"] = self.degree_stats_losses
        checkpoint["degree_distribution_losses"] = self.degree_distribution_losses
        checkpoint["homomorphism_density_losses"] = self.homomorphism_density_losses
        checkpoint["degree_stats_loss_weight"] = self.degree_stats_loss_weight
        checkpoint["entropy_loss_weight"] = self.entropy_loss_weight
        checkpoint[
            "degree_distribution_loss_weight"
        ] = self.degree_distribution_loss_weight
        checkpoint[
            "homomorphism_density_loss_weight"
        ] = self.homomorphism_density_loss_weight

    def on_load_checkpoint(self, checkpoint):
        self.best_eval_tdas = defaultdict(
            lambda: float("inf"), checkpoint["best_eval_tdas"]
        )
        self.loss_graphlets = checkpoint["loss_graphlets"]
        self.degree_stats_losses = checkpoint["degree_stats_losses"]
        self.degree_distribution_losses = checkpoint["degree_distribution_losses"]
        self.homomorphism_density_losses = checkpoint["homomorphism_density_losses"]
        self.degree_stats_loss_weight = checkpoint["degree_stats_loss_weight"]
        self.entropy_loss_weight = checkpoint.get(
            "entropy_loss_weight", 1
        )  # backward compability
        self.degree_distribution_loss_weight = checkpoint[
            "degree_distribution_loss_weight"
        ]
        self.homomorphism_density_loss_weight = checkpoint[
            "homomorphism_density_loss_weight"
        ]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr
        )

    def forward(self, x, y):
        v = tools.preprocess_vertices(x, y)
        v, _ = torch.sort(v)
        v = self.model(v)

        return v

    def homomorphism_density(self, batch):
        (
            _,
            (
                sampled_edges,
                n_true_edges,
                sampled_non_edges,
                n_true_non_edges,
                _,
                n_samples,
            ),
            _,
        ) = batch

        edges_outputs = self(*sampled_edges)
        edges_outputs_reshaped = edges_outputs.reshape(n_samples, n_true_edges)
        edge_probs = torch.prod(edges_outputs_reshaped, dim=1)

        if self.hparams.use_homomorphism_density_non_edges:
            non_edges_outputs = 1 - self(*sampled_non_edges)
            non_edges_outputs_reshaped = non_edges_outputs.reshape(
                n_samples, n_true_non_edges
            )
            non_edge_probs = torch.prod(non_edges_outputs_reshaped, dim=1)

            hom_density = torch.mean(edge_probs * non_edge_probs)

        else:
            hom_density = torch.mean(edge_probs)

        return hom_density

    def step_part_homomorphism_density_loss(self, batch):
        (
            _,
            (
                _,
                _,
                _,
                _,
                hd,
                _,
            ),
            _,
        ) = batch

        hom_density = self.homomorphism_density(batch)

        return ((hom_density - hd) ** 2).mean()

    def degree_distribution(self, n_vertices: int):
        degree_distribtuion_graphon = degree_distribution(
            self, n_vertices, resolution=100
        )

        return degree_distribtuion_graphon

    def step_part_degree_distribution_loss(self, batch):
        *_, (degree_distribution_true, _) = batch

        degree_distribution_true = degree_distribution_true.squeeze(0)
        degree_distribtuion_graphon = self.degree_distribution(
            len(degree_distribution_true)
        )

        difference = degree_distribution_true - degree_distribtuion_graphon
        loss = ((difference ** 2)).sum()

        return loss

    def step_part_degree_stats_loss(self, batch):
        *_, (_, scaled_degrees_true) = batch
        scaled_degrees_true = scaled_degrees_true.squeeze(0)

        n = min(100, len(scaled_degrees_true))

        if random.random() <= n / len(scaled_degrees_true):
            loss = 0
            scaled_degrees_graphon = scaled_degree(self, torch.rand(n).view(-1, 1))

            sd_max_true = CACHE(lambda: scaled_degrees_true.max())
            sd_max_graphon = scaled_degrees_graphon.max()
            loss += (sd_max_true - sd_max_graphon) ** 2

            sd_min_true = CACHE(lambda: scaled_degrees_true.min())
            sd_min_graphon = scaled_degrees_graphon.min()
            loss += (sd_min_true - sd_min_graphon) ** 2

            return loss

        return torch.Tensor([0]).to(self.device)

    def step_part_entropy_loss(self, batch):
        n = 50  # comb(50, 2) = 1225 samples

        vertices = torch.rand(n)
        combinations = torch.combinations(vertices, 2).to(self.device)

        p = self(combinations[:, 0], combinations[:, 1])

        weight = 1 / (self.current_epoch or 1)
        return (
            weight
            * (1 / math.comb(n, 2))
            * (torch.sum((1 - p) * torch.log(1 - p)) + torch.sum(p * torch.log(p)))
        )

    def sum_loss(self, loss: t.Optional[torch.Tensor], add: torch.Tensor):
        if (
            add is None
            or torch.isnan(add)
            or torch.isinf(add)
            or add.detach().cpu().item() == 0
        ):
            return loss

        if loss is None:
            loss = 0

        return loss + add

    def step(self, batch) -> torch.Tensor:
        ((graphlet_name,),), *_ = batch

        calc_weights = (
            not self.hparams.use_homomorphism_density_loss
            or (len(self.loss_graphlets) == len(self.dataset.graphlets))
        ) and (
            not self.hparams.use_degree_stats_loss
            or any(l != 0 for l in self.degree_stats_losses)
        )
        skip_step = (
            (
                self.hparams.use_degree_stats_loss
                and self.degree_stats_loss_weight is None
            )
            or (
                self.hparams.use_degree_distribution_loss
                and self.degree_distribution_loss_weight is None
            )
            or (
                self.hparams.use_homomorphism_density_loss
                and self.homomorphism_density_loss_weight is None
            )
            or (self.hparams.use_entropy_loss and self.entropy_loss_weight is None)
        )

        if calc_weights and skip_step:
            mean_degree_stats_losses = (
                np.mean(self.degree_stats_losses) if self.degree_stats_losses else 0
            )
            mean_degree_distribution_losses = (
                np.mean(self.degree_distribution_losses)
                if self.degree_distribution_losses
                else 0
            )
            mean_homomorphism_density_losses = (
                np.mean(self.homomorphism_density_losses)
                if self.homomorphism_density_losses
                else 0
            )
            # mean_entropy_losses = (
            #     # This loss is negative, we need absolute value for scaling with others.
            #     np.mean([abs(x) for x in self.entropy_losses]) if self.entropy_losses else 0
            # )

            max_loss = np.max(
                [
                    mean_degree_stats_losses,
                    mean_degree_distribution_losses,
                    mean_homomorphism_density_losses,
                    # mean_entropy_losses,
                ]
            )

            # Do not scale degree stats loss when keeping fixed n in calc
            self.degree_stats_loss_weight = 1
            # self.degree_stats_loss_weight = (
            #     (max_loss / mean_degree_stats_losses)
            #     if self.degree_stats_losses
            #     else None
            # )
            self.degree_distribution_loss_weight = (
                (max_loss / mean_degree_distribution_losses)
                if self.degree_distribution_losses
                else None
            )
            self.homomorphism_density_loss_weight = (
                (max_loss / mean_homomorphism_density_losses)
                if self.homomorphism_density_losses
                else None
            )
            # Do not scale entropy loss, its scaled by epoch number
            self.entropy_loss_weight = 1
            # self.entropy_loss_weight = (
            #     (max_loss / mean_entropy_losses) if self.entropy_losses else None
            # )

            skip_step = False

            print(
                f"""
                degree_stats_loss_weight set to {self.degree_stats_loss_weight}.
                degree_distribution_loss_weight set to {self.degree_distribution_loss_weight}.
                homomorphism_density_loss_weight set to {self.homomorphism_density_loss_weight}.
                entropy_loss_weight set to {self.entropy_loss_weight}.
            """
            )

        loss = None

        if self.hparams.use_degree_stats_loss:
            degree_stats_loss = self.step_part_degree_stats_loss(batch)

            if skip_step:
                self.degree_stats_losses.append(degree_stats_loss.cpu().item())

            else:
                degree_stats_loss *= self.degree_stats_loss_weight
                self.log("degree_stats_loss", degree_stats_loss, prog_bar=True)
                loss = self.sum_loss(loss, degree_stats_loss)

        if self.hparams.use_degree_distribution_loss:
            degree_distribution_loss = self.step_part_degree_distribution_loss(batch)

            if skip_step:
                self.degree_distribution_losses.append(
                    degree_distribution_loss.cpu().item()
                )

            else:
                degree_distribution_loss *= self.degree_distribution_loss_weight
                self.log(
                    "degree_distribution_loss", degree_distribution_loss, prog_bar=True
                )
                loss = self.sum_loss(loss, degree_distribution_loss)

        if self.hparams.use_homomorphism_density_loss:
            homomorphism_density_loss = self.step_part_homomorphism_density_loss(batch)

            if skip_step:
                self.homomorphism_density_losses.append(
                    homomorphism_density_loss.cpu().item()
                )

            else:
                homomorphism_density_loss *= self.homomorphism_density_loss_weight
                self.log(
                    "homomorphism_density_loss",
                    homomorphism_density_loss,
                    prog_bar=True,
                )
                self.log(
                    f"homomorphism_density_loss/{graphlet_name}",
                    homomorphism_density_loss,
                )
                loss = self.sum_loss(loss, homomorphism_density_loss)

        if self.hparams.use_entropy_loss:
            entropy_loss = self.step_part_entropy_loss(batch)

            if skip_step:
                self.entropy_losses.append(entropy_loss.cpu().item())

            else:
                entropy_loss *= self.entropy_loss_weight
                self.log("entropy_loss", entropy_loss, prog_bar=True)
                loss = self.sum_loss(loss, entropy_loss)

        self.loss_graphlets.add(graphlet_name)

        if skip_step:
            self.skipping_step = True
            return None

        else:
            if loss is None:
                self.none_batch_count += 1

                if self.none_batch_count == 100:
                    print("Too many None losses, stopping.")
                    self.trainer.should_stop = True

                return None

            self.log("loss", loss)
            self.skipping_step = False
            return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch)

    def on_train_epoch_end(self, *_, **__):
        if self.current_epoch == 0:
            return

        directory = f"{self.dataset.base_path}/training/{self.hparams.name}/{self.__class__.__name__}/epoch={self.current_epoch}"

        if not self.skipping_step and self.current_epoch >= 10 and self.current_epoch % 1 == 0:
            self.save(directory=directory)

        # if (
        #     self.current_epoch
        #     % (5 if hasattr(self.dataset, "graphon_func") else 100)
        #     == 0
        # ):
        #     self.reset_weight_losses()

    def on_fit_end(self):
        self.save(
            directory=f"{self.dataset.base_path}/training/{self.hparams.name}/{self.__class__.__name__}/epoch=fit_end",
            log=False,
            force_save_checkpoint=True,
        )

    @torch.no_grad()
    @torch_eval
    def save(
        self,
        directory: t.Union[str, Path],
        log: bool = True,
        force_save_checkpoint: bool = False,
    ):
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        tools.visualize(self, str(directory / "graphon.jpg"), resolution=100)
        tools.visualize(
            self, str(directory / "graphon.norm.jpg"), resolution=100, normalize=True
        )

        save_checkpoint = bool(self.dataset)

        if self.dataset and hasattr(self.dataset, "graphs"):
            if len(self.dataset.graphs) == 1 and len(self.dataset.graphs[0]) < 300:
                n_samples = 100
            elif len(self.dataset.graphs) == 1:
                n_samples = 2
            else:
                n_samples = len(self.dataset.graphs)

            graphs = [self.sample() for _ in tqdm(range(n_samples), desc="Graphon")]
            orig_graphs = (
                self.dataset.graphs * 2
                if len(self.dataset.graphs) == 1
                else self.dataset.graphs
            )

            if self.hparams.evaluate_networkx:
                struct = [
                    ("original", orig_graphs, "red"),
                    ("graphon", graphs, "green"),
                ]

                try:
                    tests.main(
                        struct, str(directory / f"tests.n_samples={n_samples}.jpg")
                    )
                except Exception as e:
                    print(f"Skipping tests because of {e}.")

            if EVALUATE_TDA and self.hparams.evaluate_tda:
                save_checkpoint = False

                evals = eval_tdas(
                    orig_graphs,
                    graphs,
                    only=self.hparams.tda_metrics,
                )

                self.early += 1

                for name, value in evals.items():
                    if value < self.best_eval_tdas[name]:
                        save_checkpoint = True
                        self.best_eval_tdas[name] = value
                        self.early = 0
                        print(f"Improvement on TDA {name}, resetting early stop.")

                if self.early >= self.hparams.early_stop:
                    print(f"No improvement on TDA over {self.early} epochs, stopping.")
                    self.trainer.should_stop = True

                with open(
                    directory / f"graphon.tda.n_samples={n_samples}.json", "w"
                ) as f:
                    json.dump(evals, f, indent=2)

                with open(
                    directory / f"graphon.tda.best.n_samples={n_samples}.json", "w"
                ) as f:
                    json.dump(dict(self.best_eval_tdas), f, indent=2)

                if log:
                    for evaluator, distance in evals.items():
                        self.log(f"tda/{str(evaluator)}", distance)

        if self.trainer and (save_checkpoint or force_save_checkpoint):
            print("Saving checkpoint.")
            self.trainer.save_checkpoint(str(directory / "checkpoint.ckpt"))

    @torch.no_grad()
    @torch_eval
    def sample(
        self,
        n_vertices: t.Optional[int] = None,
        batch_size: int = 1_000,
        include_all_nodes: bool = True,
    ) -> nx.Graph:
        vertices = torch.rand(
            n_vertices
            or random.choices(
                list(self.hparams.gv_probs.keys()),
                weights=list(self.hparams.gv_probs.values()),
                k=1,
            )[0],
        )
        vertices_int = list(range(len(vertices)))

        graph = nx.Graph()

        if include_all_nodes:
            graph.add_nodes_from(vertices_int)

        for batch in tools.batchit(it.combinations(vertices_int, 2), batch_size):
            batch_torch = torch.LongTensor(batch)

            edges_probs = (
                self(
                    vertices[batch_torch[:, 0]].to(self.device),
                    vertices[batch_torch[:, 1]].to(self.device),
                )
                .squeeze(1)
                .cpu()
            )
            edges = torch.rand(len(edges_probs)) <= edges_probs

            for (x, y), is_edge in zip(batch, edges):
                if is_edge:
                    graph.add_edge(x, y)

        return graph


class DeepGraphon(DeepGraphonBase):
    def __init__(self, *args, depth: int, hidden_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters("depth", "hidden_size")

        self.model = torch.nn.Sequential(
            functions.Block(input_dim=2, hidden_size=hidden_size),
            *[functions.Block(hidden_size, hidden_size) for _ in range(self.hparams.depth)],
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid(),
        )
