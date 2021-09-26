import glob
import json
import torch
import typer
import pytorch_lightning as pl
import torch_geometric as tg
import itertools as it

from tqdm import tqdm
from pathlib import Path
from collections import Counter
from pprint import pprint
from sklearn.model_selection import train_test_split

from classification.model import GNN
from gangraphon.models import GAN


DATASETS = [
    "COLLAB",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "deezer_ego_nets",
    "github_stargazers",
    "reddit_threads",
    "twitch_egos",
]


def find_checkpoint(dataset_name: str):
    def sort(name):
        epoch = int(name.split("epoch=")[1].split("-")[0])
        try:
            size = int(name.split("size=")[1].split("_")[0])
        except Exception:
            size = 0
        return size, epoch

    files = sorted(glob.glob(f"data/tensorboard_logs/*{dataset_name}*class_idx=0*/version_0/checkpoints/*.ckpt"), key=sort, reverse=True)

    try:
        print(files[0])
    except IndexError:
        raise RuntimeError(dataset_name)

    return files[0]


def prepare_dataset(
    datasets: list,
    reversed: bool,
):
    max_degree = 0

    for name in tqdm(datasets, "Max Degree"):
        dataset = tg.datasets.TUDataset(
            name=name, root="data/tudataset", cleaned=False
        )
        max_degree = max(max_degree, max(
            d
            for sample in dataset
            for _, d in tg.utils.to_networkx(sample, to_undirected=True).degree()
        ))

    transform = tg.transforms.Compose([
        tg.transforms.ToUndirected(),
        tg.transforms.OneHotDegree(max_degree),
    ])

    all_real_samples = []

    class_to_model = {}

    for label, name in enumerate(tqdm(datasets, "Parsing")):
        dataset = tg.datasets.TUDataset(
            name=name, root="data/tudataset", cleaned=False
        )
        dataset.shuffle()

        checkpoint = find_checkpoint(name)
        model = GAN.load_from_checkpoint(checkpoint)
        model.freeze()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        class_to_model[label] = model

        for sample in dataset:
            if sample.y.item() != 0:
                continue

            transformed_sample = transform(sample)
            transformed_sample.x = (transformed_sample.x.argmax(1) / transformed_sample.num_nodes).unsqueeze(-1)
            transformed_sample.y = torch.LongTensor([label])
            transformed_sample.is_sampled = False

            all_real_samples.append(transformed_sample)

    train_val_samples, test_samples = train_test_split(all_real_samples, test_size=0.2)
    train_samples, val_samples = train_test_split(train_val_samples, test_size=0.1)

    train_aug_samples = []
    val_aug_samples = []
    test_aug_samples = []

    for label, model in class_to_model.items():
        if reversed:
            for _ in tqdm(range(sum(1 if s.y.item() == label else 0 for s in train_samples)), f'Sampling train {label}'):
                graph = model.sample()

                transformed_sample = transform(tg.utils.from_networkx(graph))
                transformed_sample.x = (transformed_sample.x.argmax(1) / transformed_sample.num_nodes).unsqueeze(-1)
                transformed_sample.y = torch.LongTensor([label])
                transformed_sample.is_sampled = True

                train_aug_samples.append(transformed_sample)

            for _ in tqdm(range(sum(1 if s.y.item() == label else 0 for s in val_samples)), f'Sampling val {label}'):
                graph = model.sample()

                transformed_sample = transform(tg.utils.from_networkx(graph))
                transformed_sample.x = (transformed_sample.x.argmax(1) / transformed_sample.num_nodes).unsqueeze(-1)
                transformed_sample.y = torch.LongTensor([label])
                transformed_sample.is_sampled = True

                val_aug_samples.append(transformed_sample)

        for _ in tqdm(range(sum(1 if s.y.item() == label else 0 for s in test_samples)), f'Sampling aug test {label}'):
            graph = model.sample()

            transformed_sample = transform(tg.utils.from_networkx(graph))
            transformed_sample.x = (transformed_sample.x.argmax(1) / transformed_sample.num_nodes).unsqueeze(-1)
            transformed_sample.y = torch.LongTensor([label])
            transformed_sample.is_sampled = True

            test_aug_samples.append(transformed_sample)

    n_features = 1

    if reversed:
        return n_features, list(class_to_model.keys()), train_aug_samples, val_aug_samples, test_samples, test_aug_samples

    return n_features, list(class_to_model.keys()), train_samples, val_samples, test_samples, test_aug_samples


def train(
    datasets: list,
    epochs: int = 100,
    reversed: bool = False,
    sampled_weight: float = 1.0,
    batch_size: int = 1,
) -> dict:
    n_features, labels, train_data, val_data, test_data, test_aug_data = prepare_dataset(
        datasets=datasets,
        reversed=reversed,
    )

    ratios = {}

    for name, data in [
        ("train_data", train_data),
        ("val_data", val_data),
        ("test_data", test_data),
        ("test_aug_data", test_aug_data),
    ]:
        counter = Counter([s.y.item() for s in data])
        ratios[name] = {label: round(counter[label] / sum(counter.values()), 2) for label in labels}

    pprint(ratios)

    train_loader = tg.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = tg.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = tg.data.DataLoader(test_data, batch_size=batch_size)
    test_aug_loader = tg.data.DataLoader(test_aug_data, batch_size=batch_size)

    augmented_class_loss_weight = {c: sampled_weight for c in labels}

    model = GNN(
        n_features=n_features,
        n_classes=len(labels),
        augmented_class_loss_weight=augmented_class_loss_weight,
    )

    checkpoint_val_callback = pl.callbacks.ModelCheckpoint(
        dirpath="/tmp/checkpoints",
        monitor="val/accuracy",
        mode="max",
    )
    es_val_callback = pl.callbacks.EarlyStopping(
        monitor="val/accuracy",
        mode="max",
        patience=3,
    )

    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else None,
        logger=pl.loggers.tensorboard.TensorBoardLogger("/tmp/tensorboard_logs"),
        callbacks=[checkpoint_val_callback, es_val_callback],
        max_epochs=epochs,
    )

    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=val_loader,
    )

    test_metrics = trainer.test(
        test_dataloaders=test_loader,
        ckpt_path=checkpoint_val_callback.best_model_path,
    )

    test_aug_metrics = trainer.test(
        test_dataloaders=test_aug_loader,
        ckpt_path=checkpoint_val_callback.best_model_path,
    )

    return {
        "reversed": reversed,
        "sampled_weight": sampled_weight,
        "val_accuracy": float(checkpoint_val_callback.best_model_score),
        "test_accuracy": float(test_metrics[0]["test/accuracy"]),
        "test_aug_accuracy": float(test_aug_metrics[0]["test/accuracy"]),
        "in_train": len(train_data),
        "in_val": len(val_data),
        "in_test": len(test_data),
        "ratios": ratios,
        "orig_accuracies": {
            f"{k}": float(v) for k, v in test_metrics[0].items()
        },
        "aug_accuracies": {
            f"{k}": float(v) for k, v in test_aug_metrics[0].items()
        },
    }


def main(reversed: bool = False, dir: str = "data/classification2"):
    pl.seed_everything(0)

    out_dir = Path(dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in [2, 3]:
        for data in it.combinations(DATASETS, i):
            try:
                result = train(datasets=list(data), reversed=reversed)

            except Exception:
                continue

            with open(out_dir / f"datasets.{data}.{reversed=}.json".replace(" ", "_"), "w") as f:
                json.dump(result, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
