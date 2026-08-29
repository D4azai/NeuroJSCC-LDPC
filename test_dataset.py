from data.video_dataset import VideoFrameDataset


print("=" * 60)
print("TESTING UCF101 DATASET")
print("=" * 60)


train_dataset = VideoFrameDataset(
    root_dir="data/train",
    frame_interval=5,
    image_size=128,
)

val_dataset = VideoFrameDataset(
    root_dir="data/val",
    frame_interval=5,
    image_size=128,
)

test_dataset = VideoFrameDataset(
    root_dir="data/test",
    frame_interval=5,
    image_size=128,
)


print()
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ---------------------------------------------------------
# Read a few samples
# ---------------------------------------------------------

print()
print("Testing individual frames...")

if len(train_dataset) == 0:
    raise RuntimeError(
        "No samples available in the training split. "
        "Run: python .\\scripts\\prepare_ucf101.py\n"
        "and ensure OpenCV has FFmpeg support."
    )

for i in range(min(3, len(train_dataset))):

    frame = train_dataset[i]

    print(
        f"Sample {i}:",
        frame.shape,
        f"min={frame.min().item():.3f}",
        f"max={frame.max().item():.3f}"
    )


print()
print("=" * 60)
print("DATASET TEST PASSED")
print("=" * 60)