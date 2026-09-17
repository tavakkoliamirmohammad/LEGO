"""GPU validation of the wavefront model with a shared-memory-bound microbenchmark.

Counters are unavailable on nodes where a DCGM exporter holds them, so validate
the model through time: a loop dominated by shared loads runs proportionally to
the wavefronts the model predicts.
"""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from lego_synth.layouts import RowMajor, XorSwizzle, Rotation
from lego_synth.harness.smem_bench import SmemBench, predicted_wavefronts_per_instr

LAYOUTS = {
    "naive": RowMajor(32, 32),
    "pad1": RowMajor(32, 32, pad=1),
    "xor": XorSwizzle(32, 32, shift=0, mask=31),
    "rot1": Rotation(32, 32, r=1),
}


@pytest.fixture(scope="module")
def bench():
    return SmemBench(LAYOUTS, tile=32, block_rows=8).build()


def test_model_predicts_32_wavefronts_for_naive_and_1_for_the_rest():
    assert predicted_wavefronts_per_instr(LAYOUTS["naive"]) == 32
    for name in ("pad1", "xor", "rot1"):
        assert predicted_wavefronts_per_instr(LAYOUTS[name]) == 1


def test_conflicting_layout_is_an_order_of_magnitude_slower_on_smem_bound_loop(bench):
    ms = {name: bench.time(name, reps=2048) for name in LAYOUTS}
    free = [ms[n] for n in ("pad1", "xor", "rot1")]
    assert max(free) / min(free) < 1.15, ms          # conflict-free variants agree
    assert ms["naive"] / min(free) > 8, ms           # 32 predicted wavefronts vs 1
