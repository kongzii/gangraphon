import glob
import typer
import random
import pytorch_lightning as pl

from tqdm import tqdm
from pathlib import Path
from shutil import copyfile


def main(
    path: Path,
    in_train: int = typer.Option(
        30, help="Total samples in training dataset, in_train / n_classes per class."
    ),
    in_val: int = typer.Option(
        30, help="Total samples in validation dataset, in_val / n_classes per class."
    ),
    seed: int = 0,
):
    pl.seed_everything(seed)

    if not (path / "graphs").exists():
        raise RuntimeError(
            "Input should be directory with output from `parse_tg_dataset.py` script."
        )

    if (path / "splits").exists():
        raise RuntimeError(f"Splits already exists? `rm -d {path}/splits`")

    labels = [p.split("/")[-1] for p in glob.glob(f"{path}/graphs/*")]

    train_per_class = int(in_train / len(labels))
    val_per_class = int(in_val / len(labels))

    for label in tqdm(labels):
        graphs = glob.glob(f"{path}/graphs/{label}/*.edges")
        random.shuffle(graphs)

        (path / "splits" / "train" / str(label)).mkdir(parents=True, exist_ok=False)
        (path / "splits" / "val" / str(label)).mkdir(parents=True, exist_ok=False)
        (path / "splits" / "test" / str(label)).mkdir(parents=True, exist_ok=False)

        for file in graphs[:train_per_class]:
            filename = file.split("/")[-1]
            copyfile(file, f"{path}/splits/train/{label}/{filename}")

        for file in graphs[train_per_class : train_per_class + val_per_class]:
            filename = file.split("/")[-1]
            copyfile(file, f"{path}/splits/val/{label}/{filename}")

        for file in graphs[train_per_class + val_per_class :]:
            filename = file.split("/")[-1]
            copyfile(file, f"{path}/splits/test/{label}/{filename}")


if __name__ == "__main__":
    typer.run(main)
