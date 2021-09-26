import os
import json
import torch
import typer
import pickle
import typing as t
import pytorch_lightning as pl

from pathlib import Path
from collections import Counter
from deepgraphon import models
from common import data, graphons, tools

NAME = "DG_{graph_name}_v{version}_probStats_float_hd={use_homomorphism_density_loss}_ds={use_degree_stats_loss}_dd={use_degree_distribution_loss}_ent={use_entropy_loss}_hidden_size={hidden_size}_depth={depth}"


def main(
    graph_file: t.Optional[str] = None,
    graphon_name: t.Optional[graphons.Graphon] = None,
    graphon_graph_size: int = 250,
    graphon_n_sampled_graphs: int = 1,
    tudataset_name: t.Optional[str] = None,
    epochs: int = 150,
    initial_seed: int = 0,
    num_workers: int = 0,
    resume_from_checkpoint: t.Optional[str] = None,
    evaluate_tda: bool = True,
    evaluate_networkx: bool = False,
):
    pl.seed_everything(initial_seed, workers=True)

    if graph_file:
        graph_name = graph_file.split("/")[-1]
        pgd_macro_file = "/".join(graph_file.split("/")[:-1]) + "/pgd.macro"

        if os.path.exists(pgd_macro_file):
            print("PGD statistics found, using DatasetPGD.")
            train_dataset = data.DatasetPGD(
                macro_file_path=pgd_macro_file, graph_file=graph_file
            )
            use_homomorphism_density_non_edges = True

        else:
            print("PGD statistics NOT found, using DatasetMC.")
            train_dataset = data.DatasetMC(graph_file=graph_file)
            use_homomorphism_density_non_edges = False

        val_dataset = train_dataset  # TODO: No `val` graphs available in this case

    elif graphon_name:
        graph_name = f"{graphon_name}_{graphon_n_sampled_graphs}x"
        graphon_func = getattr(graphons, graphon_name.value)
        train_dataset = data.DatasetGraphon(
            graphon_func=graphon_func,
            sample_size=graphon_graph_size,
            n_sampled_graphs=graphon_n_sampled_graphs,
        )
        val_dataset = data.DatasetGraphon(
            graphon_func=graphon_func,
            sample_size=graphon_graph_size,
            n_sampled_graphs=graphon_n_sampled_graphs,
        )
        use_homomorphism_density_non_edges = False
        tools.visualize(graphon_func, f"{train_dataset.base_path}/graphon.jpg", 100)
        tools.visualize(
            graphon_func,
            f"{train_dataset.base_path}/graphon.norm.jpg",
            100,
            normalize=True,
        )

    elif tudataset_name:
        graph_name = f"{tudataset_name}"
        use_homomorphism_density_non_edges = False
        train_dataset, val_dataset = data.DatasetTUDataset.split(
            name=tudataset_name,
            train_size=0.5,
            limit_graphs=200,
        )

    else:
        raise RuntimeError("Incomplete input parameters.")

    assert train_dataset.__class__.__name__ == val_dataset.__class__.__name__
    assert train_dataset.base_path == val_dataset.base_path

    best_metrics: t.Optional[t.List[str]]

    try:
        with open(f"{train_dataset.base_path}/pertubations.tda.json", "r") as f:
            pertubations_tda = json.load(f)

        best_correlation = max(
            x["spearmanr"]["correlation"]
            for x in pertubations_tda["RewireEdges"].values()
        )
        best_metrics = [
            name
            for name, v in pertubations_tda["RewireEdges"].items()
            if v["spearmanr"]["correlation"] == best_correlation
        ]

    except FileNotFoundError:
        print(
            """
        Warning:
        `pertubations.tda.json` not found, execute `docker-compose run base python common/evaluators/pertub.py`
        with correpsonding graph source before running this.
        All metrics will be evaluated or none if those files are missing.
        """
        )
        best_metrics = None

    if graphon_name or tudataset_name:
        gv_counts = Counter([len(g) for g in train_dataset.graphs])
        gv_probs = {
            size: count / sum(gv_counts.values())
            for size, count in gv_counts.most_common()
        }

    elif graph_file:
        gv_probs = {len(val_dataset.graphs[0]): 1.0}

    else:
        raise RuntimeError

    model_params = dict(
        depth=1,
        hidden_size=512,
        use_homomorphism_density_loss=True,
        use_degree_distribution_loss=True,
        use_degree_stats_loss=True,
        use_entropy_loss=True,
        evaluate_tda=evaluate_tda,
        evaluate_networkx=evaluate_networkx,
    )

    version = 0
    name = NAME.format(graph_name=graph_name, version=version, **model_params)

    while os.path.exists(os.path.join("data/tensorboard_logs", name)):
        version += 1
        name = NAME.format(graph_name=graph_name, version=version, **model_params)

    if tudataset_name:
        path = Path(f"{train_dataset.base_path}/training/{name}")
        path.mkdir(parents=True)

        with open(path / "deepgraphon_train_val_graphs.pickle", "wb") as f:
            pickle.dump(
                {
                    "train": train_dataset.graphs,
                    "test": val_dataset.graphs,
                },
                f,
            )

    print(f"Model name {name}.")

    model = models.DeepGraphon(
        name=name,
        use_homomorphism_density_non_edges=use_homomorphism_density_non_edges,
        dataset=val_dataset,
        gv_probs=gv_probs,
        tda_metrics=best_metrics,
        **model_params,
    )

    logger = pl.loggers.tensorboard.TensorBoardLogger(
        save_dir="data/tensorboard_logs", name=name
    )

    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else None,
        deterministic=True,
        checkpoint_callback=False,
        max_epochs=epochs,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
        precision=32,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataset.get_dataloader(num_workers=num_workers),
    )


if __name__ == "__main__":
    typer.run(main)
