import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    """
    Differentiable AWGN channel.

    y = x + n

    n ~ N(0, sigma²)

    Signal power is assumed to be normalized to 1.
    """

    def __init__(
        self,
        snr_db: float = 10.0,
    ):
        super().__init__()

        self.snr_db = float(snr_db)

    @staticmethod
    def snr_to_noise_std(
        snr_db: float
    ) -> float:

        snr_linear = 10.0 ** (
            snr_db / 10.0
        )

        noise_variance = (
            1.0 / snr_linear
        )

        return noise_variance ** 0.5

    def forward(
        self,
        x,
        snr_db=None
    ):

        if snr_db is None:
            snr_db = self.snr_db

        noise_std = self.snr_to_noise_std(
            snr_db
        )

        noise = torch.randn_like(x)

        return x + noise * noise_std