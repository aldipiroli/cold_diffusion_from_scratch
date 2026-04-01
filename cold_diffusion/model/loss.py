import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, config, logger):
        super(BaseLoss, self).__init__()
        self.config = config
        self.logger = logger

    def forward(self, preds, labels):
        pass


class ColdDiffusionLoss(BaseLoss):
    def __init__(self, config, logger):
        super(ColdDiffusionLoss, self).__init__(config, logger)

    def forward(self, x0, x_denoise):
        loss = torch.nn.functional.l1_loss(x0, x_denoise)
        loss_dict = {"cold_diffusion_loss": loss}
        return loss, loss_dict
