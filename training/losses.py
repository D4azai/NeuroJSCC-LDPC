import torch.nn as nn


class ReconstructionLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, reconstructed, original):
        return self.loss(reconstructed, original)