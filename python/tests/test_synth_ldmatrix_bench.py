"""GPU validation of the staging result: ldmatrix.x4 reads from a b16 tile under synthesized layouts."""
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

from lego_synth.layouts import RowMajor, XorSwizzle
from lego_synth.harness.ldmatrix_bench import LdmatrixBench, predicted_wavefronts_per_ldmatrix

# 64 x 80 b16 tile (160 B rows): the width-80 staging case from the ablation.
LAYOUTS = {
    "naive": RowMajor(64, 80),                               # 160 B rows: 2-way conflicts on ldmatrix
    "pad8": RowMajor(64, 80, pad=8),                         # 176 B rows: conflict-free for ldmatrix
    "synth": XorSwizzle(64, 80, shift=2, mask=1, unit=8),    # solver's exact-footprint solution
    "pow2": XorSwizzle(64, 128, shift=0, mask=7, unit=8),    # Triton/CUTLASS: pad to 128, SW128
}


def test_model_predictions_for_the_four_layouts():
    assert predicted_wavefronts_per_ldmatrix(LAYOUTS["naive"]) == 8     # 2-way in each of 4 phases
    for name in ("pad8", "synth", "pow2"):
        assert predicted_wavefronts_per_ldmatrix(LAYOUTS[name]) == 4


@pytest.fixture(scope="module")
def bench():
    return LdmatrixBench(LAYOUTS, rows=64, cols=80).build()


def test_ldmatrix_loads_are_correct_under_every_layout(bench):
    # Each variant must reconstruct the same 16x16 fragments as a reference gather.
    for name in LAYOUTS:
        assert bench.check(name), name


def test_synthesized_layout_matches_pow2_swizzle_speed_with_smaller_footprint(bench):
    ms = {name: bench.time(name, reps=1024) for name in LAYOUTS}
    assert 1.6 < ms["naive"] / ms["synth"] < 2.4, ms              # 8 predicted wavefronts vs 4
    assert abs(ms["synth"] - ms["pow2"]) / ms["pow2"] < 0.15, ms  # same speed as the padded swizzle
    assert LAYOUTS["synth"].footprint * 1.5 < LAYOUTS["pow2"].footprint
