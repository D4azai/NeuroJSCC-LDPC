import os

import torch
from torch.utils.data import DataLoader
from training.losses import ReconstructionLoss
from models import CNNEncoder, CNNDecoder
from data.video_dataset import VideoFrameDataset

VIDEO_PATH = "data/videos/test_video.mp4"

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-3

LATENT_CHANNELS = 12


def train():

    # ==========================================
    # Device
    # ==========================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 50)
    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("=" * 50)

    # ==========================================
    # Dataset
    # ==========================================

    dataset = VideoFrameDataset(
        video_path=VIDEO_PATH,
        frame_interval=5,
        image_size=128
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # ==========================================
    # Models
    # ==========================================

    encoder = CNNEncoder(
        latent_channels=LATENT_CHANNELS
    ).to(device)

    decoder = CNNDecoder(
        latent_channels=LATENT_CHANNELS
    ).to(device)

    # ==========================================
    # Loss
    # ==========================================

    criterion = ReconstructionLoss()

    # ==========================================
    # Optimizer
    # ==========================================

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        +
        list(decoder.parameters()),
        lr=LEARNING_RATE
    )

    # ==========================================
    # Training
    # ==========================================

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    for epoch in range(EPOCHS):

        encoder.train()
        decoder.train()

        running_loss = 0.0

        for batch_idx, frames in enumerate(
            dataloader
        ):

            frames = frames.to(
                device,
                non_blocking=True
            )

            # ------------------------------
            # Forward
            # ------------------------------

            latent = encoder(frames)

            reconstructed = decoder(
                latent
            )

            # ------------------------------
            # Loss
            # ------------------------------

            loss = criterion(
                reconstructed,
                frames
            )

            # ------------------------------
            # Backpropagation
            # ------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = (
            running_loss
            /
            len(dataloader)
        )

        print(
            f"Epoch "
            f"[{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.6f}"
        )

        # ======================================
        # Save checkpoint
        # ======================================

        torch.save(
            {
                "epoch": epoch + 1,
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": average_loss
            },
            f"checkpoints/step1_latent{LATENT_CHANNELS}.pth"
        )


if __name__ == "__main__":
    train()