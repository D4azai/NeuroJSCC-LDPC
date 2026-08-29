import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models import CNNDecoder, CNNEncoder


DEFAULT_CHECKPOINTS = [
    "checkpoints/step1_latent8.pth",
    "checkpoints/step1_latent12.pth",
    "checkpoints/step1_latent16.pth",
    "checkpoints/step1_latent32.pth"
]


def metrics(original, reconstructed):
    mse = float(np.mean((original - reconstructed) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)
    mae = float(np.mean(np.abs(original - reconstructed)))
    return mse, psnr, mae


def latent_channels(checkpoint):
    if "latent_channels" in checkpoint:
        return int(checkpoint["latent_channels"])
    return int(checkpoint["encoder"]["encoder.9.weight"].shape[0])


def load_experiment(path, device):
    checkpoint = torch.load(path, map_location=device)
    channels = latent_channels(checkpoint)
    encoder = CNNEncoder(latent_channels=channels).to(device)
    decoder = CNNDecoder(latent_channels=channels).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, channels


def read_frames(video_path, image_size, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"Video contains no frames: {video_path}")

    indices = np.linspace(0, total - 1, min(num_frames, total), dtype=int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()
        if not success:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size))
        original = frame.astype(np.float32) / 255.0
        tensor = torch.from_numpy(original).permute(2, 0, 1).unsqueeze(0)
        frames.append((int(index), original, tensor))
    cap.release()

    if not frames:
        raise RuntimeError("Could not read any selected frames")
    return frames


def image_uint8(image):
    return np.ascontiguousarray(
        np.clip(image * 255.0, 0, 255).astype(np.uint8)
    )


def save_metrics_plot(results, output_path):
    names = [row["experiment"] for row in results]
    channels = [row["latent_channels"] for row in results]
    psnr = np.array([row["mean_psnr_db"] for row in results])
    mse = np.array([row["mean_mse"] for row in results])
    mae = np.array([row["mean_mae"] for row in results])
    colors = [(46, 160, 46)] + [(168, 120, 76)] * (len(results) - 1)

    best = results[0]
    baseline = results[-1]
    psnr_gain = best["mean_psnr_db"] - baseline["mean_psnr_db"]
    mse_reduction = (
        100.0 * (baseline["mean_mse"] - best["mean_mse"])
        / baseline["mean_mse"]
        if baseline["mean_mse"] > 0
        else 0.0
    )

    canvas = np.full((900, 1400, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas, "Reconstruction Experiment Comparison", (40, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv2.LINE_AA,
    )

    def draw_bars(area, title, values, higher_is_better, decimals):
        x0, y0, x1, y1 = area
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (225, 225, 225), 1)
        cv2.putText(
            canvas, title, (x0 + 15, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA,
        )
        low = float(np.min(values))
        high = float(np.max(values))
        span = high - low or 1.0
        base = y1 - 45
        top = y0 + 55
        width = max(1, (x1 - x0 - 40) // len(values))
        for index, value in enumerate(values):
            height = int(35 + ((value - low) / span) * (base - top - 35))
            left = x0 + 20 + index * width
            right = left + width - 12
            bar_top = base - height
            cv2.rectangle(canvas, (left, bar_top), (right, base), colors[index], -1)
            cv2.putText(
                canvas, f"{value:.{decimals}f}", (left, max(top, bar_top - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, names[index], (left, base + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1, cv2.LINE_AA,
            )
        direction = "higher is better" if higher_is_better else "lower is better"
        cv2.putText(
            canvas, direction, (x0 + 15, y1 - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA,
        )

    draw_bars((40, 75, 680, 370), "Mean PSNR (dB)", psnr, True, 2)
    draw_bars((720, 75, 1360, 370), "Mean MSE", mse, False, 6)
    draw_bars((40, 400, 680, 695), "Mean MAE", mae, False, 6)

    x0, y0, x1, y1 = (720, 400, 1360, 695)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (225, 225, 225), 1)
    cv2.putText(
        canvas, "Latent channels vs PSNR", (x0 + 15, y0 + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 40, 40), 2, cv2.LINE_AA,
    )
    min_channel, max_channel = min(channels), max(channels)
    min_psnr, max_psnr = min(psnr), max(psnr)
    channel_span = max_channel - min_channel or 1
    psnr_span = max_psnr - min_psnr or 1
    points = []
    for channel, value, name in zip(channels, psnr, names):
        px = x0 + 45 + int((channel - min_channel) / channel_span * (x1 - x0 - 90))
        py = y1 - 45 - int((value - min_psnr) / psnr_span * (y1 - y0 - 100))
        points.append((px, py))
        cv2.circle(canvas, (px, py), 6, (46, 160, 46), -1)
        cv2.putText(
            canvas, name, (px - 25, py - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (30, 30, 30), 1, cv2.LINE_AA,
        )
    for first, second in zip(points, points[1:]):
        cv2.line(canvas, first, second, (46, 160, 46), 2)
    cv2.putText(
        canvas, f"{min_channel}", (x0 + 40, y1 - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, f"{max_channel}", (x1 - 55, y1 - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA,
    )

    insight_lines = [
        f"BEST: {best['experiment']} ({best['latent_channels']} latent channels)",
        f"Mean PSNR: {best['mean_psnr_db']:.2f} dB",
        f"Against lowest PSNR: +{psnr_gain:.2f} dB, {mse_reduction:.1f}% lower MSE",
        "",
        "How to read this:",
        "- Higher PSNR means better reconstruction quality.",
        "- Lower MSE and MAE mean smaller pixel errors.",
        "- More latent channels are useful only when PSNR improves.",
    ]
    cv2.rectangle(canvas, (40, 730), (1360, 865), (242, 242, 242), -1)
    for index, line in enumerate(insight_lines):
        cv2.putText(
            canvas, line, (60, 760 + index * 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA,
        )

    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not save metrics plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare reconstruction quality across checkpoints."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoint paths to compare.",
    )
    parser.add_argument(
        "--video",
        default="data/videos/test_video.mp4",
        help="Video used for every experiment.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="evaluation/comparison_results",
        help="Folder for metrics_plot.png and comparison.jpg.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    if args.image_size <= 0 or args.num_frames <= 0:
        parser.error("--image-size and --num-frames must be positive")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = [Path(path) for path in args.checkpoints]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    frames = read_frames(args.video, args.image_size, args.num_frames)
    experiments = []
    for path in paths:
        encoder, decoder, channels = load_experiment(path, device)
        experiments.append((path, encoder, decoder, channels))

    results = []
    reconstructions = {index: [] for index, _, _ in frames}
    for path, encoder, decoder, channels in experiments:
        values = []
        for index, original, tensor in frames:
            with torch.no_grad():
                reconstructed = decoder(encoder(tensor.to(device)))
            reconstructed = (
                reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
            )
            reconstructed = np.clip(reconstructed, 0.0, 1.0)
            mse, psnr, mae = metrics(original, reconstructed)
            values.append((mse, psnr, mae))
            reconstructions[index].append(
                (path.stem, psnr, reconstructed)
            )

        results.append({
            "experiment": path.stem,
            "checkpoint": str(path),
            "latent_channels": channels,
            "mean_psnr_db": float(np.mean([v[1] for v in values])),
            "mean_mse": float(np.mean([v[0] for v in values])),
            "mean_mae": float(np.mean([v[2] for v in values])),
        })

    results.sort(key=lambda row: row["mean_psnr_db"], reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_plot_path = output_dir / "metrics_plot.png"
    save_metrics_plot(results, metrics_plot_path)

    montage_rows = []
    for index, original, _ in frames:
        images = [image_uint8(original)]
        for name, psnr, reconstructed in reconstructions[index]:
            image = image_uint8(reconstructed)
            cv2.putText(
                image, f"{name}: {psnr:.2f} dB", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA,
            )
            images.append(image)
        montage_rows.append(cv2.hconcat(images))

    montage_path = output_dir / "comparison.jpg"
    montage = cv2.vconcat(montage_rows)
    if not cv2.imwrite(
        str(montage_path), cv2.cvtColor(montage, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"Could not save {montage_path}")

    print(f"Device: {device}")
    print(f"Frames evaluated: {len(frames)}")
    print("\nRanking (higher PSNR is better):")
    for rank, row in enumerate(results, 1):
        print(
            f"{rank}. {row['experiment']} | "
            f"latent={row['latent_channels']} | "
            f"PSNR={row['mean_psnr_db']:.2f} dB | "
            f"MSE={row['mean_mse']:.6f} | MAE={row['mean_mae']:.6f}"
        )
    print(f"\nSaved metrics plot: {metrics_plot_path}")
    print(f"Saved: {montage_path}")


if __name__ == "__main__":
    main()
