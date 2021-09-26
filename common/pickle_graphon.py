import typer
import pickle

from common import graphons
from common.data import DatasetGraphon


def main(
    graphon: graphons.Graphon,
    n_vertices: int,
    train_count: int = 100,
    test_count: int = 100,
):
    graphon_func = getattr(graphons, graphon.value)

    train = [
        DatasetGraphon.sample(graphon_func, n_vertices) for _ in range(train_count)
    ]
    test = [DatasetGraphon.sample(graphon_func, n_vertices) for _ in range(test_count)]

    with open(
        f"data/artificial/{graphon.value}/train_count={train_count}.test_count={test_count}.pickle",
        "wb",
    ) as f:
        pickle.dump(
            {
                "train": train,
                "test": test,
            },
            f,
        )


if __name__ == "__main__":
    typer.run(main)
