import torch
import torch.nn as nn

from models.encoder import CNNEncoder
from models.decoder import CNNDecoder

from models.jscc_encoder import JSCCEncoder
from models.jscc_decoder import JSCCDecoder

from channel.awgn import AWGNChannel


class NeuroJSCC(nn.Module):
    """
    Complete Step 1 + Step 2 communication architecture.

    Video
        ↓
    Semantic Encoder
        ↓
    Semantic Latent
        ↓
    JSCC Encoder
        ↓
    AWGN Channel
        ↓
    JSCC Decoder
        ↓
    Recovered Semantic Latent
        ↓
    Semantic Decoder
        ↓
    Reconstruction
    """

    def __init__(
        self,
        latent_channels=16,
        channel_dim=512,
        hidden_dim=512,
        snr_db=10.0,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.channel_dim = channel_dim

        # -------------------------------
        # STEP 1 — Semantic Encoder
        # -------------------------------

        self.encoder = CNNEncoder(
            latent_channels=latent_channels
        )

        # -------------------------------
        # STEP 2 — JSCC Encoder
        # -------------------------------

        self.jscc_encoder = JSCCEncoder(
            latent_channels=latent_channels,
            latent_height=8,
            latent_width=8,
            channel_dim=channel_dim,
            hidden_dim=hidden_dim,
        )

        # -------------------------------
        # Communication Channel
        # -------------------------------

        self.channel = AWGNChannel(
            snr_db=snr_db
        )

        # -------------------------------
        # STEP 2 — JSCC Decoder
        # -------------------------------

        self.jscc_decoder = JSCCDecoder(
            latent_channels=latent_channels,
            latent_height=8,
            latent_width=8,
            channel_dim=channel_dim,
            hidden_dim=hidden_dim,
        )

        # -------------------------------
        # STEP 1 — Semantic Decoder
        # -------------------------------

        self.decoder = CNNDecoder(
            latent_channels=latent_channels
        )

    def forward(
        self,
        x,
        snr_db=None,
        return_intermediate=False,
    ):

        # ===============================
        # STEP 1 — Semantic encoding
        # ===============================

        z = self.encoder(x)

        # ===============================
        # STEP 2 — JSCC encoding
        # ===============================

        transmitted = self.jscc_encoder(z)

        # ===============================
        # Communication channel
        # ===============================

        received = self.channel(
            transmitted,
            snr_db=snr_db,
        )

        # ===============================
        # STEP 2 — JSCC decoding
        # ===============================

        z_hat = self.jscc_decoder(
            received
        )

        # ===============================
        # STEP 1 — Semantic decoding
        # ===============================

        reconstruction = self.decoder(
            z_hat
        )

        if return_intermediate:

            return {
                "latent": z,
                "transmitted": transmitted,
                "received": received,
                "recovered_latent": z_hat,
                "reconstruction": reconstruction,
            }

        return reconstruction