import glob
import json
import torch
import typer
import pytorch_lightning as pl
import torch_geometric as tg

from tqdm import tqdm
from pathlib import Path

from classification.model import GNN
from gangraphon.models import GAN


def find_checkpoint(dataset_name: str, class_idx: int):
    def sort(name):
        epoch = int(name.split("epoch=")[1].split("-")[0])
        try:
            size = int(name.split("size=")[1].split("_")[0])
        except Exception:
            size = 0
        return size, epoch

    files = sorted(glob.glob(f"data/tensorboard_logs/*{dataset_name}*class_idx={class_idx}*/version_0/checkpoints/*.ckpt"), key=sort, reverse=True)

    try:
        print(files[0])
    except IndexError:
        raise RuntimeError(dataset_name)

    return files[0]


def prepare_dataset(
    name: str,
    n_sample: int,
):
    dataset = tg.datasets.TUDataset(
        name=name, root="data/tudataset", cleaned=False
    )
    dataset.shuffle()

    max_degree = 0
    class_to_model = {}

    gen_samples = []
    train_samples = []
    val_samples = []
    test_samples = []

    max_degree = max(
        d
        for sample in tqdm(dataset, "Max Degree")
        for _, d in tg.utils.to_networkx(sample, to_undirected=True).degree()
    )
    transform = tg.transforms.Compose([
        tg.transforms.ToUndirected(),
        tg.transforms.OneHotDegree(max_degree),
    ])

    for sample in tqdm(dataset, "Parsing TUDataset"):
        label = sample.y.item()

        if label not in class_to_model:
            checkpoint = find_checkpoint(name, label)
            if checkpoint is None:
                raise RuntimeError((name, label))
            model = GAN.load_from_checkpoint(checkpoint)
            model.freeze()
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            class_to_model[label] = model

        transformed_sample = transform(sample)
        transformed_sample.x = (transformed_sample.x.argmax(1) / transformed_sample.num_nodes).unsqueeze(-1)
        transformed_sample.y = sample.y
        transformed_sample.is_sampled = False

        if sum(1 if s.y.item() == label else 0 for s in train_samples) < 10:
            train_samples.append(transformed_sample)

        elif sum(1 if s.y.item() == label else 0 for s in val_samples) < 50:
            val_samples.append(transformed_sample)

        else:
            test_samples.append(transformed_sample)

    for label, model in class_to_model.items():
        for _ in tqdm(range(n_sample), f'Sampling train {label}'):
            graph = model.sample()

            transformed_sample = transform(tg.utils.from_networkx(graph))
            transformed_sample.x = (transformed_sample.x.argmax(1) / transformed_sample.num_nodes).unsqueeze(-1)
            transformed_sample.y = torch.LongTensor([label])
            transformed_sample.is_sampled = True

            gen_samples.append(transformed_sample)

    n_features = 1

    return n_features, list(class_to_model.keys()), gen_samples, train_samples, val_samples, test_samples


def train(
    dataset_name: str,
    epochs: int = 50,
    n_sample: int = 0,
    sampled_weight: float = 1.0,
    batch_size: int = 1,
) -> dict:
    n_features, labels, gen_samples, train_data, val_data, test_data = prepare_dataset(
        name=dataset_name,
        n_sample=n_sample,
    )

    if gen_samples:
        gen_loader = tg.data.DataLoader(gen_samples, batch_size=batch_size, shuffle=True)

    train_loader = tg.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = tg.data.DataLoader(val_data, batch_size=batch_size)
    test_loader = tg.data.DataLoader(test_data, batch_size=batch_size)

    augmented_class_loss_weight = {c: sampled_weight for c in labels}

    model = GNN(
        n_features=n_features,
        n_classes=len(labels),
        augmented_class_loss_weight=augmented_class_loss_weight,
    )

    if gen_samples:
        pre_trainer = pl.Trainer(
            gpus=-1 if torch.cuda.is_available() else None,
            logger=pl.loggers.tensorboard.TensorBoardLogger("/tmp/tensorboard_logs"),
            max_epochs=epochs,
        )

        pre_trainer.fit(
            model,
            train_dataloader=gen_loader,
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

    return {
        "n_sample": n_sample,
        "sampled_weight": sampled_weight,
        "val_accuracy": float(checkpoint_val_callback.best_model_score),
        "test_accuracy": float(test_metrics[0]["test/accuracy"]),
        "in_pre_train": len(gen_samples),
        "in_train": len(train_data),
        "in_val": len(val_data),
        "in_test": len(test_data),
    }


def main(dataset_name: str):
    pl.seed_everything(0)

    out_dir = Path("data/classification")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    results.append(
        train(
            dataset_name=dataset_name,
            n_sample=0,
        )
    )

    for sampled_weight in [1]:  # [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
        for n_sample in [5, 10, 50, 100, 200, 300, 500]:
            results.append(
                train(
                    dataset_name=dataset_name,
                    n_sample=n_sample,
                    sampled_weight=sampled_weight,
                )
            )

            print(results)
            with open(out_dir / f"{dataset_name}.json", "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    typer.run(main)
