"""GPU integration tests: transpose kernels built from synthesized layouts, measured on hardware."""
import shutil
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from lego_synth.layouts import RowMajor, XorSwizzle, Rotation
from lego_synth.harness.transpose_cuda import TransposeHarness, predicted_conflicts

LAYOUTS = {
    "naive": RowMajor(32, 32),
    "pad1": RowMajor(32, 32, pad=1),
    "xor": XorSwizzle(32, 32, shift=0, mask=31),
    "rot1": Rotation(32, 32, r=1),
}


@pytest.fixture(scope="module")
def harness():
    h = TransposeHarness(LAYOUTS, tile=32, block_rows=8)
    h.build()
    return h


def test_every_variant_transposes_correctly(harness):
    x = torch.randn(1024, 1024, device="cuda")
    for name in LAYOUTS:
        assert torch.equal(harness.run(name, x), x.t().contiguous()), name


def test_timing_returns_positive_milliseconds(harness):
    x = torch.randn(4096, 4096, device="cuda")
    ms = harness.time("xor", x, iters=20, warmup=5)
    assert 0 < ms < 100


@pytest.mark.skipif(shutil.which("ncu") is None, reason="needs Nsight Compute on PATH")
def test_measured_shared_conflicts_match_model_prediction(harness):
    import subprocess
    n = 4096
    try:
        measured = harness.ncu_shared_conflicts(n=n)      # {variant: (ld_conflicts, st_conflicts)}
    except subprocess.CalledProcessError as e:
        # Exit 9 = profiling counters unavailable (another tool such as dcgm-exporter holds them).
        pytest.skip(f"Nsight Compute could not read counters on this node (exit {e.returncode})")
    for name, L in LAYOUTS.items():
        pred_ld, pred_st = predicted_conflicts(L, n=n, tile=32)
        assert measured[name] == (pred_ld, pred_st), (name, measured[name], (pred_ld, pred_st))
