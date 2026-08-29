import torch

from models import CNNEncoder, CNNDecoder


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    encoder = CNNEncoder(
        latent_channels=8
    ).to(device)

    decoder = CNNDecoder(
        latent_channels=8
    ).to(device)

    # Fake video frame batch
    x = torch.randn(
        4,
        3,
        128,
        128
    ).to(device)

    # Encoder
    z = encoder(x)

    # Decoder
    reconstructed = decoder(z)

    print("Input:", x.shape)
    print("Latent:", z.shape)
    print("Output:", reconstructed.shape)


if __name__ == "__main__":
    main()