import cv2
import torch
import matplotlib.pyplot as plt

from models import CNNEncoder, CNNDecoder


# ==========================================
# Configuration
# ==========================================

VIDEO_PATH = "data/videos/test_video.mp4"
CHECKPOINT_PATH = "checkpoints/step1_latest.pth"

IMAGE_SIZE = 128
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


# ==========================================
# Load models
# ==========================================

encoder = CNNEncoder(
    latent_channels=latent_channels
).to(DEVICE)

decoder = CNNDecoder(
    latent_channels=latent_channels
).to(DEVICE)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()


# ==========================================
# Open video
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ==========================================
# Read first frame
# ==========================================

success, frame = cap.read()

cap.release()

if not success:
    raise RuntimeError("Could not read first frame")


# ==========================================
# OpenCV BGR → RGB
# ==========================================

frame_rgb = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB
)


# ==========================================
# Resize to 128×128
# ==========================================

frame_resized = cv2.resize(
    frame_rgb,
    (IMAGE_SIZE, IMAGE_SIZE)
)


# ==========================================
# Convert to PyTorch tensor
# ==========================================

frame_tensor = torch.from_numpy(
    frame_resized
).float() / 255.0


# H × W × C → C × H × W
frame_tensor = frame_tensor.permute(
    2, 0, 1
)


# Add batch dimension
frame_tensor = frame_tensor.unsqueeze(0)


# Move to GPU
frame_tensor = frame_tensor.to(DEVICE)


# ==========================================
# Encoder → Latent
# ==========================================

with torch.no_grad():

    latent = encoder(frame_tensor)

    reconstructed = decoder(latent)


# ==========================================
# Print information
# ==========================================

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Original shape:",
    frame_tensor.shape
)

print(
    "Latent shape:",
    latent.shape
)
print("Latent channels:", latent_channels)

print(
    "Reconstructed shape:",
    reconstructed.shape
)


# ==========================================
# Move reconstructed frame to CPU
# ==========================================

reconstructed = (
    reconstructed
    .squeeze(0)
    .permute(1, 2, 0)
    .cpu()
    .numpy()
)


# ==========================================
# Display
# ==========================================

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(frame_resized)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed)
plt.title("Reconstructed")
plt.axis("off")

plt.tight_layout()
plt.show()