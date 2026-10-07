import torch

from ldpc.coding import BinaryLinearCode
from ldpc.belief_propagation import min_sum_decode
from ldpc.channel import transmit_bpsk
from ldpc.parity_check import hamming_7_4
from research_utils import bit_error_rate, set_seed


def test_generated_codewords_satisfy_parity_checks() -> None:
    code = BinaryLinearCode.from_parity_check(hamming_7_4())
    words = code.sample(32, torch.Generator().manual_seed(7))
    syndrome = (words @ torch.as_tensor(code.h, dtype=torch.int64).T) % 2
    assert not syndrome.any()


def test_ber_and_seed_are_deterministic() -> None:
    logits = torch.tensor([[2.0, -1.0], [-2.0, 3.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    assert bit_error_rate(logits, targets) == 0.25
    set_seed(3)
    first = torch.rand(4)
    set_seed(3)
    assert torch.equal(first, torch.rand(4))


def test_min_sum_decodes_clean_codewords() -> None:
    code = BinaryLinearCode.from_parity_check(hamming_7_4())
    generator = torch.Generator().manual_seed(4)
    words = code.sample(16, generator)
    _, llr = transmit_bpsk(words, ebn0_db=20.0, code_rate=code.rate, generator=generator)
    decoded = min_sum_decode(llr, torch.as_tensor(code.h), iterations=10)
    assert torch.equal(decoded, words)
