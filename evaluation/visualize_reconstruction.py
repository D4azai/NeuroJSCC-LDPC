import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models import CNNEncoder, CNNDecoder


# ==========================================
# Configuration
# ==========================================

VIDEO_PATH = "data/videos/test_video.mp4"
CHECKPOINT_PATH = "checkpoints/step1_latest.pth"

IMAGE_SIZE = 128
LATENT_CHANNELS = 8

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
# Load models
# ==========================================

encoder = CNNEncoder(
    latent_channels=LATENT_CHANNELS
).to(DEVICE)

decoder = CNNDecoder(
    latent_channels=LATENT_CHANNELS
).to(DEVICE)


# ==========================================
# Load checkpoint
# ==========================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

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

fig, axes = plt.subplots(
    NUM_FRAMES,
    2,
    figsize=(8, NUM_FRAMES * 3)
)


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

    axes[row, 0].imshow(original)
    axes[row, 0].set_title(
        f"Original — Frame {frame_idx}"
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(reconstructed)
    axes[row, 1].set_title(
        f"Reconstructed — PSNR {psnr:.2f} dB"
    )
    axes[row, 1].axis("off")


cap.release()

plt.tight_layout()

plt.show()


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