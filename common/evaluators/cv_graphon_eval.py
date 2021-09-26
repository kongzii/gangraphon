import json
import typer
import numpy as np
import pytorch_lightning as pl

from tqdm import tqdm
from math import comb
from itertools import combinations
from common import graphons
from common.graphons import Graphon
from common.data import DatasetGraphon
from common.evaluators.eval import eval_tdas, TDA_EVALUATORS


def main(
    graphon_name: Graphon,
    n_samples: int = 10,
    n_vertices: int = 250,
):
    pl.seed_everything(0)

    graphon_func = getattr(graphons, graphon_name.value)

    output = f"data/artificial/{graphon_name.value}/cv.tda.json"
    graphs = [
        DatasetGraphon.sample(graphon_func, n_vertices=n_vertices)
        for _ in tqdm(range(n_samples))
    ]
    results = {str(e): {"values": []} for e in TDA_EVALUATORS}

    for a, b in tqdm(combinations(graphs, 2), total=comb(len(graphs), 2)):
        for name, value in eval_tdas([a, a], [b, b]).items():
            results[name]["values"].append(value)

        with open(output, "w") as f:
            json.dump(results, f, indent=2)

    for name, value in results.items():
        results[name]["mean"] = np.mean(value["values"])
        results[name]["std"] = np.std(value["values"])

    with open(output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
