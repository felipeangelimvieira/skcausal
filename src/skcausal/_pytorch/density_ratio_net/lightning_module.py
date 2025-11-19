from typing import List
from lightning import LightningModule
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from skcausal._pytorch.density_ratio_net.loss import LogisticRandomLoss, LogisticLoss


class _BatchLevelDirectPropensityTrainingLogic(LightningModule):
    def __init__(
        self,
        model,
        learning_rate=0.001,
        l2_penalty=0,
        l2_inv_penalty=0,
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate

        self.loss = LogisticLoss(l2_penalty=l2_penalty, l2_inv_penalty=l2_inv_penalty)

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):

        x, t, x_synth, t_synth = batch

        loss = self.loss(model=self.model, x=x, t=t, x_synth=x_synth, t_synth=t_synth)
        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, t, x_synth, t_synth = batch
        test_loss = self.loss(
            model=self.model, x=x, t=t, x_synth=x_synth, t_synth=t_synth
        )
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
            [{"params": self.model.parameters()}], lr=self.learning_rate, amsgrad=False
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, t = batch
        return self.model(x, t)
