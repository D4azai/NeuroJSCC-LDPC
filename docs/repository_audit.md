# Repository audit

The repository began as a compact two-stage communication prototype. The tracked
code contains a four-layer convolutional semantic encoder and mirrored decoder,
a video-frame dataset backed by OpenCV, MSE reconstruction training, saved model
checkpoints, and a PSNR visualization script. The working tree also contained a
JSCC encoder/decoder, an end-to-end `NeuroJSCC` module, and a differentiable AWGN
channel; these have been retained and are now part of the research branch.

The semantic autoencoder maps 128×128 RGB frames to an 8×8 latent tensor. The
JSCC modules flatten that tensor, map it to normalized real channel symbols, add
AWGN, and map the noisy symbols back to the latent shape. The original training
script trains only the semantic autoencoder. The JSCC training file, Rayleigh
channel, and channel evaluation file were empty at audit time. Configuration was
held in module constants, and no dependency manifest or automated tests existed.

No LDPC matrix, binary encoder, classical decoder, or graph-learning component
was present. The added research pipeline is therefore adjacent to the analog
NeuroJSCC path: it studies valid binary codewords over BPSK/AWGN without changing
the existing image system. Joining the two paths would require an explicit and
scientifically justified latent quantization/channel-coding interface, which is
left as future work rather than implied by the current implementation.

Fragile areas retained for compatibility include hard-coded image dimensions,
per-item reopening of the video file, checkpoint-specific latent dimensions,
and evaluation scripts that execute at import time. `pytest.ini` confines test
discovery to the new `tests/` directory so those legacy executable scripts are
not imported during the research test suite.

## Mainline integration note

The research audit was performed on the `staging` worktree supplied at the start
of the task. The repository's unrelated `main` history also contained a more
substantial UCF101 preparation, training, checkpoint-comparison, and evaluation
workflow. During final integration the research commits were replayed on top of
`origin/main`, preserving those mainline files. Large datasets and checkpoints
remain excluded by the mainline `.gitignore`, so they must be supplied locally
when running the original video experiments.
