import os
import json
import pickle
import typer
import torch
import typing as t
import networkx as nx
import pytorch_lightning as pl

from pathlib import Path
from collections import Counter
from gangraphon.models import GAN
from common import graphons, data

NAME = (
    "GAN_GNN={use_gnn}_gnn={gnn_layer}"
    "_selfloops={self_loops}"
    "_dir={directed}"
    "_amp={amp}"
    "_HDS-GNN={use_hds_gnn}_HDS-SQR={use_hds_sqr}"
    "_dd_sqr={use_dd_sqr}"
    "_gmb={gumbel}_gmb_hrd={gumbel_hard}_tau={tau_value}_train_tau={trainable_tau}"
    "_max_size={max_graph_size}"
    "_{graph_name}_{datasetname}_class_idx={class_idx}"
    "_v{version}"
)


def main(
    tudataset_name: t.Optional[str] = None,
    class_idx: t.Optional[int] = None,
    graphon_name: t.Optional[graphons.Graphon] = None,
    graph_file: t.Optional[str] = None,
    grid_graph: bool = False,
    self_loops: bool = True,
    amp: bool = False,
    use_gnn: bool = True,
    use_hds_gnn: bool = False,
    use_hds_sqr: bool = False,
    use_dd_sqr: bool = False,
    gumbel: bool = True,
    gumbel_hard: bool = True,
    tau_value: float = 1000.0,
    trainable_tau: bool = True,
    allow_directed: bool = False,
    num_workers: int = 0,
    max_epochs: int = 10_000,
    gnn_layer: str = 'DenseSAGEConv',
    g_optimizer: str = "Adam",
    g_optimizer_lr: float = 1e-5,
    d_optimizer: str = "Adam",
    d_optimizer_lr: float = 1e-5,
    seed: int = 0,
    batch_size: int = 1,
    max_graph_size: int = 500,
    early_stop: int = 100,
    gpus: t.List[int] = [0],
):
    assert sum(map(bool, [tudataset_name, graphon_name, graph_file, grid_graph])) == 1

    pl.seed_everything(seed, workers=True)

    if graphon_name:
        ds_name = graphon_name.value
        graphon_func = getattr(graphons, graphon_name.value)

        train_dataset = data.DatasetGraphon2(
            graphon_func=graphon_func,
            n_sampled_graphs=100,
            sample_size=max_graph_size,
            stats=any([use_hds_gnn, use_hds_sqr]),
        )
        val_dataset = data.DatasetGraphon2(
            graphon_func=graphon_func,
            n_sampled_graphs=100,
            sample_size=max_graph_size,
            stats=False,
        )

        assert train_dataset.directed == val_dataset.directed
        directed = train_dataset.directed
        assert not directed

    elif tudataset_name:
        ds_name = tudataset_name
        train_dataset, val_dataset = data.DatasetTUDataset2.split(
            name=tudataset_name,
            class_idx=class_idx,
            train_size=0.5,
            stats=any([use_hds_gnn, use_hds_sqr]),
            limit_graphs=None,
            max_graph_size=max_graph_size,
            allow_directed=allow_directed,
        )
        assert train_dataset.directed == val_dataset.directed
        directed = train_dataset.directed

    elif graph_file:
        ds_name = graph_file.split("/")[-1][:20]
        directed = False
        train_dataset, val_dataset = data.InducedGraphsDataset.split(
            graph_file=graph_file,
            n_induced=100,
            induced_size_range=(100, 200),
            stats=any([use_hds_gnn, use_hds_sqr]),
        )

    elif grid_graph:
        res = 10
        ds_name = f'grid_graph={res}x{res}'
        directed = False
        train_dataset = data.DatasetBase2(
            graphs=[nx.generators.lattice.grid_graph(dim=(res, res)) for _ in range(10)],
            stats=any([use_hds_gnn, use_hds_sqr]),
            base_path=f'data/artificial/{ds_name}',
            name=ds_name,
        )
        val_dataset = train_dataset

    assert train_dataset.__class__.__name__ == val_dataset.__class__.__name__
    assert train_dataset.base_path == val_dataset.base_path

    best_metrics: t.Optional[t.List[str]]

    try:
        with open(f"{train_dataset.base_path}/pertubations.tda.json", "r") as f:
            pertubations_tda = json.load(f)

        best_correlation = max(
            x["spearmanr"]["correlation"] for x in pertubations_tda["RewireEdges"].values()
        )
        best_metrics = [
            name
            for name, v in pertubations_tda["RewireEdges"].items()
            if v["spearmanr"]["correlation"] == best_correlation
        ]

    except FileNotFoundError:
        print("Warning: Execute `docker-compose run base python common/evaluators/pertub.py --help` before running this, otherwise all TDAs will be evaluated.")
        best_metrics = None

    if graphon_name or tudataset_name:
        gv_counts = Counter([len(g) for g in train_dataset.graphs])
        gv_probs = {
            size: count / sum(gv_counts.values())
            for size, count in gv_counts.most_common()
        }

    elif graph_file or grid_graph:
        gv_probs = {len(val_dataset.graphs[0]): 1.0}

    else:
        raise RuntimeError

    datasetname = train_dataset.__class__.__name__

    version = 0
    name = NAME.format(
        class_idx=class_idx,
        amp=amp,
        graph_name=ds_name,
        directed=directed,
        self_loops=self_loops,
        datasetname=datasetname,
        version=version,
        use_gnn=use_gnn,
        use_hds_gnn=use_hds_gnn,
        use_hds_sqr=use_hds_sqr,
        use_dd_sqr=use_dd_sqr,
        gumbel=gumbel,
        gumbel_hard=gumbel_hard,
        tau_value=tau_value,
        trainable_tau=trainable_tau,
        gnn_layer=gnn_layer,
        max_graph_size=max_graph_size,
    )

    while os.path.exists(f"{train_dataset.base_path}/training/{name}"):
        version += 1
        name = NAME.format(
            class_idx=class_idx,
            amp=amp,
            graph_name=ds_name,
            directed=directed,
            self_loops=self_loops,
            datasetname=datasetname,
            version=version,
            use_gnn=use_gnn,
            use_hds_gnn=use_hds_gnn,
            use_hds_sqr=use_hds_sqr,
            use_dd_sqr=use_dd_sqr,
            gumbel=gumbel,
            gumbel_hard=gumbel_hard,
            tau_value=tau_value,
            trainable_tau=trainable_tau,
            gnn_layer=gnn_layer,
            max_graph_size=max_graph_size,
        )

    path = Path(f"{train_dataset.base_path}/training/{name}")
    path.mkdir(parents=True)

    with open(path / "train_val_graphs.pickle", "wb") as f:
        pickle.dump(
            {
                "train": train_dataset.graphs,
                "test": val_dataset.graphs,
            },
            f,
        )

    print(f"Model name {name}.")

    model = GAN(
        name=name,
        gv_probs=gv_probs,
        graph_features=14,
        use_gnn=use_gnn,
        use_hds_gnn=use_hds_gnn,
        use_hds_sqr=use_hds_sqr,
        use_dd_sqr=use_dd_sqr,
        directed=directed,
        self_loops=self_loops,
        gumbel=gumbel,
        gumbel_hard=gumbel_hard,
        tau_value=tau_value,
        trainable_tau=trainable_tau,
        generator_kwargs=dict(),
        amp=amp,
        g_optimizer=g_optimizer,
        g_optimizer_kwargs=dict(lr=g_optimizer_lr),
        d_optimizer=d_optimizer,
        d_optimizer_kwargs=dict(lr=d_optimizer_lr),
        dataset=val_dataset,
        tda_metrics=best_metrics,
        early_stop=early_stop,
        gnn_layer=gnn_layer,
    )

    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="data/tensorboard_logs",
        name=name,
    )

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        logger=logger,
        terminate_on_nan=True,
    )
    trainer.fit(
        model,
        train_dataset.get_dataloader(num_workers=num_workers, batch_size=batch_size),
    )


if __name__ == "__main__":
    typer.run(main)
