import pytest
import torch
import torch.nn.functional as F

from benchmark import base, consts


EMBEDDING_CASES = [
    ((16,), 4096, 16, "random"),
    ((1024,), 32768, 16, "random"),
    ((4096,), 32768, 64, "random"),
    ((8192,), 65536, 127, "random"),
    ((64, 128), 65536, 128, "random"),
    ((32, 128), 65536, 256, "repeat"),
    ((16, 256), 65536, 768, "hotspot"),
    ((8, 1024), 131072, 64, "repeat"),
]


def _normalize_case(case):
    if len(case) == 4 and isinstance(case[0], (list, tuple)):
        shape, vocab_size, embedding_dim, distribution = case
        return tuple(shape), vocab_size, embedding_dim, distribution
    vocab_size, embedding_dim = case
    return (vocab_size,), vocab_size, embedding_dim, "random"


def _estimate_peak_bytes(shape, vocab_size, embedding_dim, dtype):
    n_indices = torch.Size(shape).numel()
    elem_size = torch.empty((), dtype=dtype).element_size()
    return (
        n_indices * torch.empty((), dtype=torch.long).element_size()
        + vocab_size * embedding_dim * elem_size
        + n_indices * embedding_dim * elem_size
    )


def _make_indices(shape, vocab_size, distribution, device):
    if distribution == "repeat":
        local_vocab = min(vocab_size, 256)
        return torch.randint(
            0, local_vocab, shape, dtype=torch.long, device=device
        )
    if distribution == "hotspot":
        indices = torch.zeros(shape, dtype=torch.long, device=device)
        flat = indices.reshape(-1)
        if flat.numel() > 1:
            flat[1::8] = torch.randint(
                0,
                vocab_size,
                (flat[1::8].numel(),),
                dtype=torch.long,
                device=device,
            )
        return indices
    return torch.randint(0, vocab_size, shape, dtype=torch.long, device=device)


def embedding_input_fn(case, dtype, device, max_peak_bytes):
    shape, vocab_size, embedding_dim, distribution = _normalize_case(case)
    if (
        _estimate_peak_bytes(shape, vocab_size, embedding_dim, dtype)
        > max_peak_bytes
    ):
        return
    indices = _make_indices(shape, vocab_size, distribution, device)
    weight = torch.empty(
        (vocab_size, embedding_dim),
        dtype=dtype,
        device=device,
    ).normal_()
    yield indices, weight


@pytest.mark.embedding
def test_embedding():
    bench = base.EmbeddingBenchmark(
        op_name="embedding",
        torch_op=F.embedding,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=embedding_input_fn,
        cases=EMBEDDING_CASES,
        max_peak_bytes=6 * 1024**3,
    )
    bench.run()
