from typing import List
import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from skcausal._pytorch.density_ratio_net.loss import LogisticRandomLoss


class LinearInterpolationProbaLogic(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=0.001,
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

        self.loss = _Logloss()

    def training_step(self, batch, batch_idx):

        x, t = batch

        loss = self.loss(model=self.model, x=x, t=t)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        test_loss = self.loss(model=self.model, x=x, t=t)
        self.log(
            "val_loss",
            test_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            on_step=False,
        )
        return test_loss

    def configure_optimizers(self):
        return optim.Adam(
            [{"params": self.parameters()}], lr=self.learning_rate, amsgrad=False
        )


class _Logloss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, model, x, t):
        w = model(x, t)
        prob = 1 / w
        loss = -torch.log(prob).mean()
        return loss
