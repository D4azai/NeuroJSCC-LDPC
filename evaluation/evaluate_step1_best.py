import os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models import CNNEncoder, CNNDecoder
from data.video_dataset import VideoFrameDataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "test"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "step1_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "step1_evaluation"
)

IMAGE_SIZE = 128

BATCH_SIZE = 8

NUM_VISUAL_SAMPLES = 16

NUM_VIDEO_FRAMES = 16

FRAME_INTERVAL = 5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PSNR
# ============================================================

def compute_psnr(
    reconstructed,
    original,
):
    """
    Compute PSNR in dB.

    Returns a Python float.
    """

    mse = torch.mean(
        (reconstructed - original) ** 2
    )

    mse_value = mse.item()

    if mse_value <= 1e-12:
        return float("inf")

    psnr = (
        10.0
        * np.log10(
            1.0 / mse_value
        )
    )

    return float(psnr)


# ============================================================
# FIND BATCH MSE / MAE
# ============================================================

def compute_batch_metrics(
    reconstructed,
    original,
):
    diff = (
        reconstructed
        - original
    )

    mse = torch.mean(
        diff ** 2
    ).item()

    mae = torch.mean(
        torch.abs(diff)
    ).item()

    return mse, mae


# ============================================================
# SAVE VISUAL COMPARISON
# ============================================================

def save_visual_comparison(
    original_frames,
    reconstructed_frames,
    output_path,
):
    num_frames = min(
        len(original_frames),
        NUM_VISUAL_SAMPLES,
    )

    rows = num_frames

    fig, axes = plt.subplots(
        rows,
        2,
        figsize=(8, rows * 3),
    )

    if rows == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    for i in range(rows):

        original = (
            original_frames[i]
            .numpy()
        )

        reconstructed = (
            reconstructed_frames[i]
            .numpy()
        )

        axes[i, 0].imshow(
            original
        )

        axes[i, 0].set_title(
            f"Original {i + 1}"
        )

        axes[i, 0].axis("off")

        axes[i, 1].imshow(
            reconstructed
        )

        axes[i, 1].set_title(
            f"Reconstructed {i + 1}"
        )

        axes[i, 1].axis("off")

    fig.suptitle(
        "Step 1 — Best Checkpoint Reconstruction",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# SAVE RECONSTRUCTION VIDEO
# ============================================================

def save_reconstruction_video(
    model_encoder,
    model_decoder,
    device,
    output_path,
):
    """
    Creates a small side-by-side reconstruction video
    from frames sampled from the test dataset.
    """

    # Find videos
    video_extensions = {
        ".avi",
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
    }

    videos = []

    for path in TEST_DIR.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in video_extensions
        ):
            videos.append(path)

    videos.sort()

    if not videos:
        print(
            "No videos found for reconstruction video."
        )
        return

    video_path = videos[0]

    print(
        f"Reconstruction video source: "
        f"{video_path}"
    )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        print(
            f"Could not open: {video_path}"
        )
        return

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if total_frames <= 0:
        cap.release()
        return

    # Choose evenly spaced frames
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        min(
            NUM_VIDEO_FRAMES,
            total_frames,
        ),
        dtype=int,
    )

    output_frames = []

    model_encoder.eval()
    model_decoder.eval()

    with torch.no_grad():

        for frame_idx in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_idx),
            )

            success, frame = cap.read()

            if not success:
                continue

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # Resize
            frame_rgb = cv2.resize(
                frame_rgb,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ),
            )

            # [0,255] -> [0,1]
            original = (
                frame_rgb.astype(
                    np.float32
                )
                / 255.0
            )

            # HWC -> CHW
            tensor = (
                torch.from_numpy(
                    original
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            # Encoder -> Decoder
            latent = model_encoder(
                tensor
            )

            reconstructed = model_decoder(
                latent
            )

            reconstructed = (
                reconstructed
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            reconstructed = np.clip(
                reconstructed,
                0.0,
                1.0,
            )

            # Convert to uint8
            original_uint8 = (
                original * 255.0
            ).astype(
                np.uint8
            )

            reconstructed_uint8 = (
                reconstructed * 255.0
            ).astype(
                np.uint8
            )

            # RGB -> BGR for OpenCV
            original_bgr = cv2.cvtColor(
                original_uint8,
                cv2.COLOR_RGB2BGR,
            )

            reconstructed_bgr = cv2.cvtColor(
                reconstructed_uint8,
                cv2.COLOR_RGB2BGR,
            )

            # Side-by-side
            side_by_side = np.concatenate(
                [
                    original_bgr,
                    reconstructed_bgr,
                ],
                axis=1,
            )

            output_frames.append(
                side_by_side
            )

    cap.release()

    if not output_frames:
        print(
            "No reconstruction frames created."
        )
        return

    height, width = (
        output_frames[0].shape[:2]
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        5,
        (width, height),
    )

    if not writer.isOpened():
        print(
            "Could not create output video."
        )
        return

    for frame in output_frames:
        writer.write(frame)

    writer.release()

    print(
        f"Saved reconstruction video:\n"
        f"{output_path}"
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("STEP 1 — BEST CHECKPOINT EVALUATION")
    print("=" * 70)

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Device:",
        DEVICE,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print()

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test directory not found:\n"
            f"{TEST_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    latent_channels = int(
        checkpoint[
            "latent_channels"
        ]
    )

    best_epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    best_val_psnr = checkpoint.get(
        "val_psnr",
        None,
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Latent channels:",
        latent_channels,
    )

    if best_val_psnr is not None:
        print(
            f"Best validation PSNR: "
            f"{best_val_psnr:.4f} dB"
        )

    print()

    # --------------------------------------------------------
    # Build models
    # --------------------------------------------------------

    encoder = CNNEncoder(
        latent_channels=latent_channels
    ).to(DEVICE)

    decoder = CNNDecoder(
        latent_channels=latent_channels
    ).to(DEVICE)

    encoder.load_state_dict(
        checkpoint["encoder"]
    )

    decoder.load_state_dict(
        checkpoint["decoder"]
    )

    encoder.eval()
    decoder.eval()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = VideoFrameDataset(
        root_dir=str(TEST_DIR),
        frame_interval=FRAME_INTERVAL,
        image_size=IMAGE_SIZE,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    print(
        "Test samples:",
        len(test_dataset),
    )

    print()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    total_samples = 0

    saved_originals = []
    saved_reconstructions = []

    with torch.no_grad():

        for batch_idx, frames in enumerate(
            test_loader
        ):

            frames = frames.to(
                DEVICE,
                non_blocking=(
                    DEVICE.type == "cuda"
                ),
            )

            latent = encoder(
                frames
            )

            reconstructed = decoder(
                latent
            )

            batch_mse, batch_mae = (
                compute_batch_metrics(
                    reconstructed,
                    frames,
                )
            )

            batch_psnr = compute_psnr(
                reconstructed,
                frames,
            )

            batch_size = frames.size(0)

            total_mse += (
                batch_mse
                * batch_size
            )

            total_mae += (
                batch_mae
                * batch_size
            )

            total_psnr += (
                batch_psnr
                * batch_size
            )

            total_samples += batch_size

            # Save some examples
            if len(saved_originals) < NUM_VISUAL_SAMPLES:

                remaining = (
                    NUM_VISUAL_SAMPLES
                    - len(saved_originals)
                )

                count = min(
                    remaining,
                    batch_size,
                )

                originals_cpu = (
                    frames[:count]
                    .detach()
                    .cpu()
                )

                reconstructed_cpu = (
                    reconstructed[:count]
                    .detach()
                    .cpu()
                )

                saved_originals.extend(
                    list(originals_cpu)
                )

                saved_reconstructions.extend(
                    list(reconstructed_cpu)
                )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    if total_samples == 0:
        raise RuntimeError(
            "Test dataset contains no samples."
        )

    avg_mse = (
        total_mse
        / total_samples
    )

    avg_mae = (
        total_mae
        / total_samples
    )

    avg_psnr = (
        total_psnr
        / total_samples
    )

    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test MSE:  {avg_mse:.6f}"
    )

    print(
        f"Test MAE:  {avg_mae:.6f}"
    )

    print(
        f"Test PSNR: {avg_psnr:.4f} dB"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Save visual comparison
    # --------------------------------------------------------

    comparison_path = (
        OUTPUT_DIR
        / "test_reconstruction_comparison.png"
    )

    save_visual_comparison(
        saved_originals,
        saved_reconstructions,
        comparison_path,
    )

    print(
        f"Saved comparison:\n"
        f"{comparison_path}"
    )

    # --------------------------------------------------------
    # Save reconstruction video
    # --------------------------------------------------------

    video_path = (
        OUTPUT_DIR
        / "test_reconstruction.mp4"
    )

    save_reconstruction_video(
        encoder,
        decoder,
        DEVICE,
        video_path,
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()