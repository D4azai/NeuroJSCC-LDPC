import cv2
import torch
import numpy as np

from models import CNNEncoder, CNNDecoder


# ==========================================
# Configuration
# ==========================================

VIDEO_PATH = "data/videos/test_video.mp4"
CHECKPOINT_PATH = "checkpoints/step1_latest.pth"

IMAGE_SIZE = 128

# Frames we want to evaluate
NUM_FRAMES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# PSNR calculation
# ==========================================

def calculate_psnr(original, reconstructed):

    mse = np.mean(
        (original - reconstructed) ** 2
    )

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10(
        1.0 / mse
    )

    return psnr


# ==========================================
# Load checkpoint
# ==========================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

latent_channels = checkpoint.get(
    "latent_channels",
    checkpoint["encoder"]["encoder.9.weight"].shape[0]
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


print("=" * 60)
print("STEP 1 EVALUATION")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)
print(
    "Latent channels:",
    latent_channels
)

print("=" * 60)


# ==========================================
# Open video
# ==========================================

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print(
    "Total video frames:",
    total_frames
)


# ==========================================
# Select frames
# ==========================================

frame_indices = np.linspace(
    0,
    total_frames - 1,
    NUM_FRAMES,
    dtype=int
)


# ==========================================
# Evaluate
# ==========================================

psnr_values = []

comparison_rows = []


for row, frame_idx in enumerate(
    frame_indices
):

    # ------------------------------
    # Read frame
    # ------------------------------

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_idx)
    )

    success, frame = cap.read()

    if not success:
        print(
            f"Could not read frame {frame_idx}"
        )
        continue

    # BGR → RGB
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    frame = cv2.resize(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Normalize [0,255] → [0,1]
    original = (
        frame.astype(np.float32)
        / 255.0
    )

    # HWC → CHW
    tensor = torch.from_numpy(
        original
    ).permute(2, 0, 1)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Move to CPU/GPU
    tensor = tensor.to(DEVICE)

    # ------------------------------
    # Encoder → Decoder
    # ------------------------------

    with torch.no_grad():

        latent = encoder(tensor)

        reconstructed = decoder(
            latent
        )

    # ------------------------------
    # Convert reconstruction
    # ------------------------------

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
        1.0
    )

    # ------------------------------
    # PSNR
    # ------------------------------

    psnr = calculate_psnr(
        original,
        reconstructed
    )

    psnr_values.append(psnr)

    print(
        f"Frame {frame_idx:4d} | "
        f"PSNR: {psnr:.2f} dB"
    )

    # ------------------------------
    # Visualization
    # ------------------------------

    original_image = (
        np.clip(original * 255.0, 0, 255)
        .astype(np.uint8, copy=True)
    )
    reconstructed_image = (
        np.clip(reconstructed * 255.0, 0, 255)
        .astype(np.uint8, copy=True)
    )

    original_image = np.ascontiguousarray(original_image)
    reconstructed_image = np.ascontiguousarray(
        reconstructed_image
    )

    original_image = cv2.putText(
        original_image,
        f"Original - Frame {frame_idx}",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )
    reconstructed_image = cv2.putText(
        reconstructed_image,
        f"Reconstructed - {psnr:.2f} dB",
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    comparison_rows.append(
        cv2.hconcat([
            cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(
                reconstructed_image,
                cv2.COLOR_RGB2BGR
            )
        ])
    )


cap.release()

if not comparison_rows:
    raise RuntimeError("No video frames were successfully evaluated")

comparison = cv2.vconcat(comparison_rows)
output_path = "evaluation/reconstruction_comparison.jpg"
if not cv2.imwrite(output_path, comparison):
    raise RuntimeError(
        f"Could not save comparison image: {output_path}"
    )

print(f"Saved comparison image: {output_path}")


# ==========================================
# Average PSNR
# ==========================================

average_psnr = np.mean(
    psnr_values
)

print("=" * 60)

print(
    f"Average PSNR: "
    f"{average_psnr:.2f} dB"
)

print("=" * 60)