import torch
import torch.nn as nn


class JSCCEncoder(nn.Module):
    """
    Neural Joint Source-Channel Coding encoder.

    Converts the semantic latent representation
    into continuous channel symbols.

    Input:
        [B, C, H, W]

    Output:
        [B, channel_dim]
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

        self.latent_dim = (
            latent_channels
            * latent_height
            * latent_width
        )

        self.channel_dim = channel_dim

        self.network = nn.Sequential(
            nn.Linear(
                self.latent_dim,
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
                channel_dim
            ),
        )

    @staticmethod
    def power_normalize(x):
        """
        Normalize each sample to unit average power.

        E[x²] ≈ 1
        """

        power = torch.mean(
            x ** 2,
            dim=1,
            keepdim=True
        )

        return x / torch.sqrt(
            power + 1e-8
        )

    def forward(self, z):

        if z.ndim != 4:
            raise ValueError(
                "Expected latent tensor "
                "[B, C, H, W], "
                f"got {z.shape}"
            )

        # [B, C, H, W]
        #       ↓
        # [B, latent_dim]

        z = z.flatten(
            start_dim=1
        )

        x = self.network(z)

        # Enforce transmission power constraint
        x = self.power_normalize(x)

        return x