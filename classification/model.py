import torch
import typing as t
import torch_geometric as tg
import pytorch_lightning as pl


class GNN(pl.LightningModule):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        augmented_class_loss_weight: dict,
        n_layers: int = 2,
        hidden_size: int = 32,
        optimizer: str = "Adam",
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = tg.nn.SAGEConv(n_features, hidden_size)
        self.convs = torch.nn.ModuleList()
        for _ in range(n_layers - 1):
            self.convs.append(tg.nn.SAGEConv(hidden_size, hidden_size))
        self.lin1 = torch.nn.Linear(hidden_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, n_classes)

        self.watched_metric = 0
        self.watched_metric_epoch = -1
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return getattr(torch.optim, self.hparams.optimizer)(
            self.parameters(), lr=self.hparams.lr
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = torch.nn.functional.relu(conv(x, edge_index))
        x = tg.nn.global_mean_pool(x, batch)
        x = torch.nn.functional.relu(self.lin1(x))
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def step(self, data, name) -> torch.Tensor:
        out = self(data)
        labels = data.y.view(-1)

        raw_loss = self.loss(out, labels)
        loss_weights = torch.Tensor(
            [
                (
                    self.hparams.augmented_class_loss_weight[int(class_.item())]
                    if bool(sampled.item())
                    else 1.0
                )
                for sampled, class_ in zip(data.is_sampled.cpu(), labels.cpu())
            ]
        ).to(self.device)
        loss = (loss_weights * raw_loss).mean()

        eq = out.detach().max(1)[1].eq(labels)

        self.log(f"{name}/accuracy", eq.sum() / data.num_graphs)

        self.log(f"{name}/loss", loss)
        self.log(f"{name}/raw_loss", raw_loss.mean())

        for label in torch.unique(labels):
            mask = labels == label

            self.log(f"{name}/accuracy/class_{label}", eq[mask].sum() / mask.sum())

        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self.step(batch, "train")

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: t.Optional[int] = None
    ) -> torch.Tensor:
        return self.step(batch, "val")

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: t.Optional[int] = None
    ) -> torch.Tensor:
        return self.step(batch, "test")
