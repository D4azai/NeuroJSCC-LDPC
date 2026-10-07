import torch

from models.neurojscc import NeuroJSCC


# Use the 16-channel Step 1 checkpoint that matches the
# current model architecture. The older step1_latest.pth
# file was produced with 8 latent channels and would fail
# to load into the 16-channel model.
CHECKPOINT = "checkpoints/step1_latent16.pth"

LATENT_CHANNELS = 16
CHANNEL_DIM = 512
HIDDEN_DIM = 512
SNR_DB = 10.0

BATCH_SIZE = 4


def load_step1_weights(model, checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    encoder_out_channels = (
        checkpoint["encoder"]["encoder.9.weight"].shape[0]
    )
    model_out_channels = (
        model.encoder.encoder[9].weight.shape[0]
    )

    if encoder_out_channels != model_out_channels:
        raise RuntimeError(
            "Checkpoint latent channels do not match the model: "
            f"checkpoint has {encoder_out_channels}, "
            f"model expects {model_out_channels}. "
            "Use a checkpoint trained with the same latent dimension."
        )

    model.encoder.load_state_dict(
        checkpoint["encoder"]
    )

    model.decoder.load_state_dict(
        checkpoint["decoder"]
    )

    print(
        f"Loaded Step 1 checkpoint "
        f"(epoch {checkpoint['epoch']})"
    )

    if "val_psnr" in checkpoint:
        print(
            f"Step 1 validation PSNR: "
            f"{checkpoint['val_psnr']:.4f} dB"
        )


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("STEP 2 — JSCC FORWARD PASS TEST")
    print("=" * 70)

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()
    print("Checkpoint:", CHECKPOINT)
    print("Latent channels:", LATENT_CHANNELS)
    print("Channel symbols:", CHANNEL_DIM)
    print("SNR:", SNR_DB, "dB")

    # ----------------------------------
    # Create complete Step 2 model
    # ----------------------------------

    model = NeuroJSCC(
        latent_channels=LATENT_CHANNELS,
        channel_dim=CHANNEL_DIM,
        hidden_dim=HIDDEN_DIM,
        snr_db=SNR_DB,
    ).to(device)

    # ----------------------------------
    # Load trained Step 1
    # ----------------------------------

    load_step1_weights(
        model,
        CHECKPOINT,
        device,
    )

    model.eval()

    # ----------------------------------
    # Fake input
    # ----------------------------------

    x = torch.rand(
        BATCH_SIZE,
        3,
        128,
        128,
        device=device,
    )

    # ----------------------------------
    # Forward pass
    # ----------------------------------

    with torch.no_grad():

        outputs = model(
            x,
            snr_db=SNR_DB,
            return_intermediate=True,
        )

    # ----------------------------------
    # Print shapes
    # ----------------------------------

    print()
    print("-" * 70)
    print("TENSOR SHAPES")
    print("-" * 70)

    print(
        "Input:",
        tuple(x.shape)
    )

    print(
        "Semantic latent:",
        tuple(outputs["latent"].shape)
    )

    print(
        "Transmitted symbols:",
        tuple(outputs["transmitted"].shape)
    )

    print(
        "Received symbols:",
        tuple(outputs["received"].shape)
    )

    print(
        "Recovered latent:",
        tuple(outputs["recovered_latent"].shape)
    )

    print(
        "Reconstruction:",
        tuple(outputs["reconstruction"].shape)
    )

    # ----------------------------------
    # Check shapes
    # ----------------------------------

    assert outputs["latent"].shape == (
        BATCH_SIZE,
        16,
        8,
        8,
    )

    assert outputs["transmitted"].shape == (
        BATCH_SIZE,
        CHANNEL_DIM,
    )

    assert outputs["received"].shape == (
        BATCH_SIZE,
        CHANNEL_DIM,
    )

    assert outputs["recovered_latent"].shape == (
        BATCH_SIZE,
        16,
        8,
        8,
    )

    assert outputs["reconstruction"].shape == (
        BATCH_SIZE,
        3,
        128,
        128,
    )

    # ----------------------------------
    # Numerical sanity checks
    # ----------------------------------

    for name, tensor in outputs.items():

        if not torch.isfinite(tensor).all():

            raise RuntimeError(
                f"{name} contains NaN or Inf values!"
            )

    print()
    print("=" * 70)
    print("STEP 2 FORWARD PASS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()