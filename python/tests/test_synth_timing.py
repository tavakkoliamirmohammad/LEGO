"""The shared timer warms up and batches by elapsed time, not only by call count."""
import pytest
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)
from lego_synth.bench.triton_matmul import time_fn  # noqa: E402


def test_time_fn_runs_enough_calls_to_cover_the_minimum_durations():
    calls = []
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    def fn():
        calls.append(1); a.matmul(a)
    ms = time_fn(fn, iters=5, warmup=2, batches=3, min_warmup_ms=50.0, min_batch_ms=30.0)
    assert ms > 0
    # 2 warm-up calls + 5 estimation calls + ~50 ms of warm-up + 3 batches of >= 30 ms each
    assert len(calls) > 2 + 5 + 3 * 5 and len(calls) * ms >= 0.5 * (50.0 + 3 * 30.0)
