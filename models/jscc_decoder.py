import torch
import torch.nn as nn


class JSCCDecoder(nn.Module):
    """
    Neural JSCC decoder.

    Converts received channel symbols back
    into the semantic latent representation.
    """

    def __init__(
        self,
        latent_channels: int = 8,
        latent_height: int = 8,
        latent_width: int = 8,
        channel_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.latent_height = latent_height
        self.latent_width = latent_width

        self.latent_dim = (
            latent_channels
            * latent_height
            * latent_width
        )

        self.network = nn.Sequential(
            nn.Linear(
                channel_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.Linear(
                hidden_dim,
                self.latent_dim
            ),
        )

    def forward(self, y):

        if y.ndim != 2:
            raise ValueError(
                "Expected received signal "
                "[B, channel_dim], "
                f"got {y.shape}"
            )

        z_hat = self.network(y)

        z_hat = z_hat.view(
            -1,
            self.latent_channels,
            self.latent_height,
            self.latent_width
        )

        return z_hat