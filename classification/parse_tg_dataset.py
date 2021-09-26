import typer
import torch_geometric as tg

from tqdm import tqdm
from pathlib import Path
from collections import Counter


def parse(dataset):
    counter = Counter()
    path = Path(dataset.root) / dataset.name / "graphs"

    for data in tqdm(dataset):
        label = data.y.item()

        graph = tg.utils.to_networkx(data, to_undirected=True)

        output_dir = path / f"{label}"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / f"{counter[label]}.n={len(graph)}.edges", "w") as f:
            for a, b in graph.edges():
                f.write(f"{a} {b}\n")

        counter[label] += 1


def main(tudataset_name: str):
    dataset = tg.datasets.TUDataset(
        name=tudataset_name, root="data/tudataset", cleaned=True
    )
    parse(dataset)


if __name__ == "__main__":
    typer.run(main)
