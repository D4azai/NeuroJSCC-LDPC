import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from training.losses import ReconstructionLoss
from models import CNNEncoder, CNNDecoder
from data.video_dataset import VideoFrameDataset


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-3

LATENT_CHANNELS = 16

FRAME_INTERVAL = 5
IMAGE_SIZE = 128

# Save checkpoints/metrics/reconstruction after training.
SAVE_EVERY_EPOCH = True


# ============================================================
# METRICS
# ============================================================

def compute_psnr(reconstructed, original):
    """
    Compute PSNR in dB and always return a Python float.

    Images are assumed to be normalized to [0, 1].
    """

    mse = torch.mean(
        (reconstructed - original) ** 2
    )

    mse_value = float(mse.detach().item())

    if mse_value <= 1e-12:
        return float("inf")

    return float(
        10.0 * np.log10(1.0 / mse_value)
    )


def evaluate_model(
    model_encoder,
    model_decoder,
    data_loader,
    criterion,
    device,
):
    """
    Evaluate a complete dataset.

    Returns:
        avg_loss
        avg_mse
        avg_mae
        avg_psnr
        sample_frames
    """

    model_encoder.eval()
    model_decoder.eval()

    total_loss = 0.0
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_pixels = 0
    total_samples = 0

    sample_frames = None

    with torch.inference_mode():

        for frames in data_loader:

            if device.type == "cuda":
                frames = frames.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                frames = frames.to(device)

            amp_enabled = (
                device.type == "cuda"
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                latent = model_encoder(frames)
                reconstructed = model_decoder(latent)
                loss = criterion(
                    reconstructed,
                    frames,
                )

            diff = (
                reconstructed.float()
                - frames.float()
            )

            batch_squared_error = torch.sum(
                diff ** 2
            ).item()

            batch_absolute_error = torch.sum(
                torch.abs(diff)
            ).item()

            total_loss += (
                float(loss.item())
                * frames.size(0)
            )

            total_squared_error += (
                batch_squared_error
            )

            total_absolute_error += (
                batch_absolute_error
            )

            total_pixels += frames.numel()

            total_samples += frames.size(0)

            if sample_frames is None:

                sample_frames = {
                    "original": (
                        frames[:2]
                        .detach()
                        .float()
                        .cpu()
                    ),
                    "reconstructed": (
                        reconstructed[:2]
                        .detach()
                        .float()
                        .cpu()
                    ),
                }

    if total_samples == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            None,
        )

    avg_loss = (
        total_loss
        / total_samples
    )

    avg_mse = (
        total_squared_error
        / max(total_pixels, 1)
    )

    avg_mae = (
        total_absolute_error
        / max(total_pixels, 1)
    )

    if avg_mse <= 1e-12:
        avg_psnr = float("inf")
    else:
        avg_psnr = float(
            10.0
            * np.log10(
                1.0 / avg_mse
            )
        )

    return (
        float(avg_loss),
        float(avg_mse),
        float(avg_mae),
        float(avg_psnr),
        sample_frames,
    )


# ============================================================
# PLOTTING
# ============================================================

def save_training_graph(
    train_history,
    val_history,
    val_psnr_history,
    save_path,
):
    """
    Save training/validation curves.

    Explicitly converts every history item to a Python float
    so CUDA tensors can never be passed to Matplotlib.
    """

    if plt is None:
        print(
            "Matplotlib is unavailable; "
            "skipping training graph."
        )
        return

    train_values = [
        float(value)
        for value in train_history
    ]

    val_values = [
        float(value)
        for value in val_history
    ]

    psnr_values = [
        float(value)
        for value in val_psnr_history
    ]

    epochs = list(
        range(
            1,
            len(train_values) + 1,
        )
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        constrained_layout=True,
    )

    axes[0].plot(
        epochs,
        train_values,
        label="Train Loss",
        color="tab:blue",
        linewidth=2,
    )

    axes[0].plot(
        epochs,
        val_values,
        label="Val Loss",
        color="tab:orange",
        linewidth=2,
    )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend()
    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[1].plot(
        epochs,
        psnr_values,
        label="Val PSNR (dB)",
        color="tab:green",
        linewidth=2,
    )

    axes[1].set_title(
        "Validation PSNR"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "PSNR (dB)"
    )

    axes[1].legend()
    axes[1].grid(
        True,
        alpha=0.3,
    )

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved training graph to: "
        f"{save_path}"
    )


# ============================================================
# RECONSTRUCTION VIDEO
# ============================================================

def save_reconstruction_video(
    validation_loader,
    model_encoder,
    model_decoder,
    device,
    save_path,
):
    """
    Save a small side-by-side reconstruction video
    from validation samples.

    Left  = original
    Right = reconstruction
    """

    if cv2 is None:
        print(
            "OpenCV is unavailable; "
            "skipping reconstruction video."
        )
        return

    model_encoder.eval()
    model_decoder.eval()

    clips = []

    with torch.inference_mode():

        for frames in validation_loader:

            if device.type == "cuda":

                frames = frames.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )

            else:

                frames = frames.to(device)

            amp_enabled = (
                device.type == "cuda"
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):

                reconstructed = (
                    model_decoder(
                        model_encoder(frames)
                    )
                )

            count = min(
                2,
                frames.size(0),
            )

            for idx in range(count):

                original_np = (
                    frames[idx]
                    .detach()
                    .float()
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                )

                reconstructed_np = (
                    reconstructed[idx]
                    .detach()
                    .float()
                    .cpu()
                    .permute(1, 2, 0)
                    .numpy()
                )

                original_np = (
                    original_np
                    .clip(0.0, 1.0)
                    * 255.0
                ).astype("uint8")

                reconstructed_np = (
                    reconstructed_np
                    .clip(0.0, 1.0)
                    * 255.0
                ).astype("uint8")

                side_by_side = np.concatenate(
                    (
                        original_np,
                        reconstructed_np,
                    ),
                    axis=1,
                )

                clips.append(
                    cv2.cvtColor(
                        side_by_side,
                        cv2.COLOR_RGB2BGR,
                    )
                )

            if len(clips) >= 16:
                break

    if not clips:
        print(
            "No validation frames available "
            "for reconstruction video."
        )
        return

    height, width = (
        clips[0].shape[:2]
    )

    writer = cv2.VideoWriter(
        str(save_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        5,
        (width, height),
    )

    if writer.isOpened():

        for frame in clips:
            writer.write(frame)

        writer.release()

        print(
            f"Saved reconstruction video to: "
            f"{save_path}"
        )

    else:

        montage_path = str(
            Path(save_path).with_suffix(".png")
        )

        montage = np.concatenate(
            clips[:4],
            axis=0,
        )

        cv2.imwrite(
            montage_path,
            montage,
        )

        print(
            f"Could not create MP4. "
            f"Saved montage to: "
            f"{montage_path}"
        )


# ============================================================
# CSV METRICS
# ============================================================

def save_metrics_csv(
    metrics_path,
    rows,
):
    with open(
        metrics_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_psnr_db",
                "val_mse",
                "val_mae",
            ]
        )

        writer.writerows(rows)

    print(
        f"Saved metrics CSV to: "
        f"{metrics_path}"
    )


# ============================================================
# CHECKPOINT HELPER
# ============================================================

def build_checkpoint(
    epoch,
    encoder,
    decoder,
    optimizer,
    train_loss,
    val_loss,
    val_mse,
    val_mae,
    val_psnr,
):
    return {
        "epoch": int(epoch),
        "latent_channels": int(
            LATENT_CHANNELS
        ),
        "image_size": int(
            IMAGE_SIZE
        ),
        "frame_interval": int(
            FRAME_INTERVAL
        ),
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_mse": float(val_mse),
        "val_mae": float(val_mae),
        "val_psnr": float(val_psnr),
    }


# ============================================================
# TRAIN
# ============================================================

def train():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

        # Safe performance settings on modern NVIDIA GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 60)
    print("STEP 1 — CNN SEMANTIC AUTOENCODER TRAINING")
    print("=" * 60)

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Epochs:",
        EPOCHS,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Latent channels:",
        LATENT_CHANNELS,
    )

    print(
        "Frame interval:",
        FRAME_INTERVAL,
    )

    print("=" * 60)

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = VideoFrameDataset(
        root_dir=TRAIN_DIR,
        frame_interval=FRAME_INTERVAL,
        image_size=IMAGE_SIZE,
    )

    val_dataset = VideoFrameDataset(
        root_dir=VAL_DIR,
        frame_interval=FRAME_INTERVAL,
        image_size=IMAGE_SIZE,
    )

    test_dataset = VideoFrameDataset(
        root_dir=TEST_DIR,
        frame_interval=FRAME_INTERVAL,
        image_size=IMAGE_SIZE,
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    print()
    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # ========================================================
    # MODELS
    # ========================================================

    encoder = CNNEncoder(
        latent_channels=LATENT_CHANNELS
    ).to(device)

    decoder = CNNDecoder(
        latent_channels=LATENT_CHANNELS
    ).to(device)

    if device.type == "cuda":

        encoder = encoder.to(
            memory_format=torch.channels_last
        )

        decoder = decoder.to(
            memory_format=torch.channels_last
        )

    # ========================================================
    # LOSS
    # ========================================================

    criterion = ReconstructionLoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        +
        list(decoder.parameters()),
        lr=LEARNING_RATE,
    )

    # ========================================================
    # MIXED PRECISION SCALER
    # ========================================================

    amp_enabled = (
        device.type == "cuda"
    )

    if hasattr(torch, "amp"):

        try:

            scaler = torch.amp.GradScaler(
                device,
                enabled=amp_enabled,
            )

        except (TypeError, AttributeError):

            scaler = torch.amp.GradScaler(
                enabled=amp_enabled,
            )

    else:

        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled
        )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # HISTORY
    # ========================================================

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_psnr": [],
    }

    metric_rows = []

    best_val_loss = float("inf")
    best_val_psnr = float("-inf")
    best_epoch = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        encoder.train()
        decoder.train()

        running_loss = 0.0
        total_train_samples = 0

        if tqdm is not None:

            pbar = tqdm(
                train_loader,
                desc=(
                    f"Epoch "
                    f"[{epoch + 1}/{EPOCHS}]"
                ),
                leave=False,
                dynamic_ncols=True,
            )

        else:

            pbar = train_loader

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        for step, frames in enumerate(
            pbar
        ):

            if device.type == "cuda":

                frames = frames.to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )

            else:

                frames = frames.to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):

                latent = encoder(frames)

                reconstructed = decoder(
                    latent
                )

                loss = criterion(
                    reconstructed,
                    frames,
                )

            optimizer.zero_grad(
                set_to_none=True
            )

            scaler.scale(
                loss
            ).backward()

            # Gradient clipping provides a little
            # extra stability for future expansions.
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters())
                +
                list(decoder.parameters()),
                max_norm=1.0,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            batch_loss = float(
                loss.detach().item()
            )

            running_loss += (
                batch_loss
                * frames.size(0)
            )

            total_train_samples += (
                frames.size(0)
            )

            current_avg_loss = (
                running_loss
                /
                max(
                    total_train_samples,
                    1,
                )
            )

            if tqdm is not None:

                pbar.set_postfix(
                    batch_loss=(
                        f"{batch_loss:.5f}"
                    ),
                    avg_loss=(
                        f"{current_avg_loss:.5f}"
                    ),
                )

            elif (
                (step + 1) % 50 == 0
                or
                (step + 1)
                ==
                len(train_loader)
            ):

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Step [{step + 1}/"
                    f"{len(train_loader)}] "
                    f"| Batch Loss: "
                    f"{batch_loss:.5f} "
                    f"| Running Avg Loss: "
                    f"{current_avg_loss:.5f}"
                )

        train_loss = (
            running_loss
            /
            max(
                total_train_samples,
                1,
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            val_loss,
            val_mse,
            val_mae,
            val_psnr,
            _
        ) = evaluate_model(
            encoder,
            decoder,
            val_loader,
            criterion,
            device,
        )

        history["train_loss"].append(
            float(train_loss)
        )

        history["val_loss"].append(
            float(val_loss)
        )

        history["val_psnr"].append(
            float(val_psnr)
        )

        print()
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| Val MSE: {val_mse:.6f} "
            f"| Val MAE: {val_mae:.6f} "
            f"| Val PSNR: {val_psnr:.4f} dB"
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        checkpoint = build_checkpoint(
            epoch=epoch + 1,
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
            val_mse=val_mse,
            val_mae=val_mae,
            val_psnr=val_psnr,
        )

        if SAVE_EVERY_EPOCH:

            epoch_path = (
                Path(CHECKPOINT_DIR)
                /
                f"step1_epoch_{epoch + 1}.pth"
            )

            torch.save(
                checkpoint,
                epoch_path,
            )

        latest_path = (
            Path(CHECKPOINT_DIR)
            /
            "step1_latest.pth"
        )

        torch.save(
            checkpoint,
            latest_path,
        )

        # ----------------------------------------------------
        # METRICS ROW
        # ----------------------------------------------------

        metric_rows.append(
            [
                epoch + 1,
                float(train_loss),
                float(val_loss),
                float(val_psnr),
                float(val_mse),
                float(val_mae),
            ]
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        is_better = (
            val_loss < best_val_loss
            or
            (
                abs(
                    val_loss
                    -
                    best_val_loss
                )
                < 1e-12
                and
                val_psnr
                >
                best_val_psnr
            )
        )

        if is_better:

            best_val_loss = float(
                val_loss
            )

            best_val_psnr = float(
                val_psnr
            )

            best_epoch = (
                epoch + 1
            )

            torch.save(
                checkpoint,
                Path(CHECKPOINT_DIR)
                /
                "step1_best.pth",
            )

        print(
            f"Best checkpoint: "
            f"{Path(CHECKPOINT_DIR) / 'step1_best.pth'} "
            f"(epoch {best_epoch}, "
            f"val_loss={best_val_loss:.6f}, "
            f"val_psnr={best_val_psnr:.4f} dB)"
        )

    # ========================================================
    # TRAINING ARTIFACTS
    # ========================================================

    metrics_csv = (
        Path(CHECKPOINT_DIR)
        /
        "metrics.csv"
    )

    save_metrics_csv(
        metrics_csv,
        metric_rows,
    )

    graph_path = (
        Path(CHECKPOINT_DIR)
        /
        "training_metrics.png"
    )

    save_training_graph(
        history["train_loss"],
        history["val_loss"],
        history["val_psnr"],
        graph_path,
    )

    # ========================================================
    # LOAD BEST MODEL BEFORE FINAL TEST
    # ========================================================

    best_checkpoint_path = (
        Path(CHECKPOINT_DIR)
        /
        "step1_best.pth"
    )

    print("=" * 60)
    print(
        "Training complete."
    )

    print(
        "Loading BEST checkpoint "
        "for final test evaluation..."
    )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=device,
    )

    encoder.load_state_dict(
        best_checkpoint["encoder"]
    )

    decoder.load_state_dict(
        best_checkpoint["decoder"]
    )

    encoder.eval()
    decoder.eval()

    print(
        f"Best epoch: "
        f"{best_checkpoint['epoch']}"
    )

    print(
        f"Best validation PSNR: "
        f"{best_checkpoint['val_psnr']:.4f} dB"
    )

    # ========================================================
    # BEST-MODEL VALIDATION RECONSTRUCTION
    # ========================================================

    save_reconstruction_video(
        val_loader,
        encoder,
        decoder,
        device,
        Path(CHECKPOINT_DIR)
        /
        "reconstruction_best.mp4",
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "Running final TEST evaluation..."
    )

    (
        test_loss,
        test_mse,
        test_mae,
        test_psnr,
        _
    ) = evaluate_model(
        encoder,
        decoder,
        test_loader,
        criterion,
        device,
    )

    print()
    print("=" * 60)
    print("FINAL STEP 1 TEST RESULTS")
    print("=" * 60)

    print(
        f"Test Loss:  {test_loss:.6f}"
    )

    print(
        f"Test MSE:   {test_mse:.8f}"
    )

    print(
        f"Test MAE:   {test_mae:.8f}"
    )

    print(
        f"Test PSNR:  {test_psnr:.4f} dB"
    )

    print("=" * 60)

    print(
        "Step 1 training and evaluation complete."
    )


if __name__ == "__main__":
    train()
