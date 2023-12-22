import lightning as L

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from torch.optim import Adam


class myModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, z):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        lr = self.config['lr']
        b1 = self.config['b1']
        b2 = self.config['b2']

        opt = Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        return opt

    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        pass
