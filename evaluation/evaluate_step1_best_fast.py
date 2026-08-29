from pathlib import Path
import time

import cv2
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from models import CNNEncoder, CNNDecoder


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "test"
)

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

FRAME_INTERVAL = 5

BATCH_SIZE = 64

NUM_VISUAL_SAMPLES = 16

NUM_VIDEO_FRAMES = 32


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GPU OPTIMIZATION
# ============================================================

if DEVICE.type == "cuda":

    torch.backends.cudnn.benchmark = True

    # Useful on modern NVIDIA GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_frame(frame):
    """
    OpenCV BGR frame
    ->
    RGB
    ->
    128x128
    ->
    float32 [0,1]
    ->
    CHW numpy array
    """

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    frame = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    frame = (
        frame.astype(np.float32)
        / 255.0
    )

    # HWC -> CHW
    frame = np.transpose(
        frame,
        (2, 0, 1),
    )

    return frame


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the already-trained best Step 1 model.
    """

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    latent_channels = int(
        checkpoint["latent_channels"]
    )

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

    return (
        encoder,
        decoder,
        checkpoint,
    )


# ============================================================
# FIND TEST VIDEOS
# ============================================================

def find_test_videos():

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

    return videos


# ============================================================
# RUN ONE BATCH
# ============================================================

def run_batch(
    frames,
    encoder,
    decoder,
):
    """
    Run one batch through:

        Encoder -> Decoder
    """

    # numpy -> tensor
    batch = torch.from_numpy(
        np.stack(frames, axis=0)
    )

    batch = batch.to(
        DEVICE,
        non_blocking=True,
    )

    # channels-last can improve CNN inference
    if DEVICE.type == "cuda":

        batch = batch.contiguous(
            memory_format=torch.channels_last
        )

    # ----------------------------------------
    # Inference
    # ----------------------------------------

    with torch.inference_mode():

        if DEVICE.type == "cuda":

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                latent = encoder(batch)

                reconstructed = decoder(
                    latent
                )

        else:

            latent = encoder(batch)

            reconstructed = decoder(
                latent
            )

    return batch, reconstructed


# ============================================================
# EVALUATE ENTIRE TEST SET
# ============================================================

def evaluate_test_set(
    videos,
    encoder,
    decoder,
):
    """
    Sequentially reads every test video.

    Frames are sampled every FRAME_INTERVAL frames,
    then processed in GPU batches.
    """

    total_squared_error = 0.0

    total_absolute_error = 0.0

    total_pixels = 0

    total_frames = 0

    valid_frames = 0

    failed_videos = 0

    visual_originals = []

    visual_reconstructions = []

    start_time = time.perf_counter()

    # ========================================================
    # Process videos
    # ========================================================

    for video_number, video_path in enumerate(
        videos,
        start=1,
    ):

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():

            print(
                f"\nWARNING: Could not open:\n"
                f"{video_path}"
            )

            failed_videos += 1

            continue

        frame_index = 0

        batch_frames = []

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Only sample every Nth frame
            if (
                frame_index
                % FRAME_INTERVAL
                != 0
            ):

                frame_index += 1
                continue

            processed = preprocess_frame(
                frame
            )

            batch_frames.append(
                processed
            )

            total_frames += 1

            # ---------------------------------------------
            # Process full batch
            # ---------------------------------------------

            if len(batch_frames) >= BATCH_SIZE:

                original_batch, reconstructed = (
                    run_batch(
                        batch_frames,
                        encoder,
                        decoder,
                    )
                )

                # CPU for metrics
                original_cpu = (
                    original_batch
                    .float()
                    .cpu()
                )

                reconstructed_cpu = (
                    reconstructed
                    .float()
                    .cpu()
                )

                diff = (
                    reconstructed_cpu
                    - original_cpu
                )

                total_squared_error += (
                    torch.sum(
                        diff ** 2
                    ).item()
                )

                total_absolute_error += (
                    torch.sum(
                        torch.abs(diff)
                    ).item()
                )

                batch_pixel_count = (
                    original_cpu.numel()
                )

                total_pixels += (
                    batch_pixel_count
                )

                batch_size_actual = (
                    original_cpu.shape[0]
                )

                valid_frames += (
                    batch_size_actual
                )

                # -----------------------------------------
                # Save visual examples
                # -----------------------------------------

                if (
                    len(visual_originals)
                    <
                    NUM_VISUAL_SAMPLES
                ):

                    remaining = (
                        NUM_VISUAL_SAMPLES
                        -
                        len(
                            visual_originals
                        )
                    )

                    count = min(
                        remaining,
                        batch_size_actual,
                    )

                    for idx in range(count):

                        visual_originals.append(
                            original_cpu[idx]
                        )

                        visual_reconstructions.append(
                            reconstructed_cpu[idx]
                        )

                batch_frames = []

            frame_index += 1

        # ====================================================
        # Process remaining frames
        # ====================================================

        if batch_frames:

            original_batch, reconstructed = (
                run_batch(
                    batch_frames,
                    encoder,
                    decoder,
                )
            )

            original_cpu = (
                original_batch
                .float()
                .cpu()
            )

            reconstructed_cpu = (
                reconstructed
                .float()
                .cpu()
            )

            diff = (
                reconstructed_cpu
                - original_cpu
            )

            total_squared_error += (
                torch.sum(
                    diff ** 2
                ).item()
            )

            total_absolute_error += (
                torch.sum(
                    torch.abs(diff)
                ).item()
            )

            total_pixels += (
                original_cpu.numel()
            )

            batch_size_actual = (
                original_cpu.shape[0]
            )

            valid_frames += (
                batch_size_actual
            )

            if (
                len(visual_originals)
                <
                NUM_VISUAL_SAMPLES
            ):

                remaining = (
                    NUM_VISUAL_SAMPLES
                    -
                    len(
                        visual_originals
                    )
                )

                count = min(
                    remaining,
                    batch_size_actual,
                )

                for idx in range(count):

                    visual_originals.append(
                        original_cpu[idx]
                    )

                    visual_reconstructions.append(
                        reconstructed_cpu[idx]
                    )

        cap.release()

        # ====================================================
        # Progress
        # ====================================================

        elapsed = (
            time.perf_counter()
            - start_time
        )

        videos_done = video_number

        video_percent = (
            videos_done
            /
            len(videos)
            *
            100.0
        )

        fps = (
            valid_frames
            /
            elapsed
            if elapsed > 0
            else 0.0
        )

        print(
            f"\r"
            f"Videos: "
            f"{videos_done}/{len(videos)} "
            f"({video_percent:6.2f}%) | "
            f"Frames: {valid_frames:,} | "
            f"Speed: {fps:7.1f} frames/s | "
            f"Time: {elapsed:7.1f}s",
            end="",
            flush=True,
        )

    print()

    # ========================================================
    # Calculate metrics
    # ========================================================

    if total_pixels == 0:

        raise RuntimeError(
            "No valid test frames were processed."
        )

    global_mse = (
        total_squared_error
        /
        total_pixels
    )

    global_mae = (
        total_absolute_error
        /
        total_pixels
    )

    if global_mse <= 1e-12:

        global_psnr = float("inf")

    else:

        global_psnr = (
            10.0
            * np.log10(
                1.0
                /
                global_mse
            )
        )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {
        "mse": global_mse,
        "mae": global_mae,
        "psnr": global_psnr,
        "frames": valid_frames,
        "failed_videos": failed_videos,
        "elapsed_seconds": elapsed,
        "originals": visual_originals,
        "reconstructions": visual_reconstructions,
    }


# ============================================================
# SAVE VISUAL COMPARISON
# ============================================================

def save_visual_comparison(
    originals,
    reconstructions,
    output_path,
):
    if not originals:
        return

    count = min(
        len(originals),
        NUM_VISUAL_SAMPLES,
    )

    fig, axes = plt.subplots(
        count,
        2,
        figsize=(8, count * 3),
    )

    if count == 1:
        axes = np.expand_dims(
            axes,
            axis=0,
        )

    for i in range(count):

        original = (
            originals[i]
            .permute(1, 2, 0)
            .numpy()
        )

        reconstructed = (
            reconstructions[i]
            .permute(1, 2, 0)
            .numpy()
        )

        original = np.clip(
            original,
            0.0,
            1.0,
        )

        reconstructed = np.clip(
            reconstructed,
            0.0,
            1.0,
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
        "Step 1 — Test Reconstruction",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# CREATE SIDE-BY-SIDE RECONSTRUCTION VIDEO
# ============================================================

def create_reconstruction_video(
    video_path,
    encoder,
    decoder,
    output_path,
):
    """
    Creates a short side-by-side video from the first
    test video.

    Original | Reconstruction
    """

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            f"Could not open video:\n"
            f"{video_path}"
        )

        return

    sampled_frames = []

    frame_index = 0

    while (
        len(sampled_frames)
        <
        NUM_VIDEO_FRAMES
    ):

        success, frame = cap.read()

        if not success:
            break

        if (
            frame_index
            % FRAME_INTERVAL
            == 0
        ):

            sampled_frames.append(
                preprocess_frame(frame)
            )

        frame_index += 1

    cap.release()

    if not sampled_frames:
        return

    # ========================================================
    # Batched inference
    # ========================================================

    batch, reconstructed = run_batch(
        sampled_frames,
        encoder,
        decoder,
    )

    batch = (
        batch
        .float()
        .cpu()
        .numpy()
    )

    reconstructed = (
        reconstructed
        .float()
        .cpu()
        .numpy()
    )

    output_frames = []

    for i in range(
        len(sampled_frames)
    ):

        original = np.transpose(
            batch[i],
            (1, 2, 0),
        )

        recon = np.transpose(
            reconstructed[i],
            (1, 2, 0),
        )

        original = np.clip(
            original * 255.0,
            0,
            255,
        ).astype(
            np.uint8
        )

        recon = np.clip(
            recon * 255.0,
            0,
            255,
        ).astype(
            np.uint8
        )

        original = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR,
        )

        recon = cv2.cvtColor(
            recon,
            cv2.COLOR_RGB2BGR,
        )

        side_by_side = np.concatenate(
            [
                original,
                recon,
            ],
            axis=1,
        )

        output_frames.append(
            side_by_side
        )

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
            "Could not create reconstruction video."
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
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 1 — FAST BEST CHECKPOINT EVALUATION")
    print("=" * 70)

    print(
        "Device:",
        DEVICE,
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Frame interval:",
        FRAME_INTERVAL,
    )

    print()

    # ========================================================
    # Prepare output directory
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load model
    # ========================================================

    encoder, decoder, checkpoint = (
        load_model()
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )

    print(
        "Best epoch:",
        checkpoint["epoch"],
    )

    print(
        "Latent channels:",
        checkpoint["latent_channels"],
    )

    print(
        f"Best validation PSNR: "
        f"{checkpoint['val_psnr']:.4f} dB"
    )

    print()

    # ========================================================
    # Find videos
    # ========================================================

    videos = find_test_videos()

    if not videos:

        raise RuntimeError(
            f"No videos found in:\n"
            f"{TEST_DIR}"
        )

    print(
        f"Test videos found: "
        f"{len(videos)}"
    )

    print(
        "Test directory:",
        TEST_DIR,
    )

    print()

    # ========================================================
    # Evaluate
    # ========================================================

    results = evaluate_test_set(
        videos,
        encoder,
        decoder,
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Test frames: "
        f"{results['frames']:,}"
    )

    print(
        f"Test MSE:    "
        f"{results['mse']:.8f}"
    )

    print(
        f"Test MAE:    "
        f"{results['mae']:.8f}"
    )

    print(
        f"Test PSNR:   "
        f"{results['psnr']:.4f} dB"
    )

    print(
        f"Failed videos: "
        f"{results['failed_videos']}"
    )

    print(
        f"Evaluation time: "
        f"{results['elapsed_seconds']:.2f} s"
    )

    if results["elapsed_seconds"] > 0:

        fps = (
            results["frames"]
            /
            results["elapsed_seconds"]
        )

        print(
            f"Overall throughput: "
            f"{fps:.1f} frames/s"
        )

    print("=" * 70)

    # ========================================================
    # Save image comparison
    # ========================================================

    comparison_path = (
        OUTPUT_DIR
        /
        "test_reconstruction_comparison_fast.png"
    )

    save_visual_comparison(
        results["originals"],
        results["reconstructions"],
        comparison_path,
    )

    print(
        f"Saved comparison:\n"
        f"{comparison_path}"
    )

    # ========================================================
    # Reconstruction video
    # ========================================================

    video_output_path = (
        OUTPUT_DIR
        /
        "test_reconstruction_fast.mp4"
    )

    create_reconstruction_video(
        videos[0],
        encoder,
        decoder,
        video_output_path,
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()