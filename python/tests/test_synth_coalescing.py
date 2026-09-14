"""Tests for the global-memory coalescing metric: 32-byte sectors touched per warp request."""
import numpy as np

from lego_synth.machine import ampere
from lego_synth.eval_np import sectors_per_request


def test_contiguous_f32_row_read_touches_the_ideal_four_sectors():
    lanes = np.arange(32)
    assert sectors_per_request(lanes * 4, access_bytes=4, machine=ampere()) == 4


def test_column_read_with_row_stride_1024_touches_one_sector_per_lane():
    lanes = np.arange(32)
    assert sectors_per_request(lanes * 1024, access_bytes=4, machine=ampere()) == 32


def test_misaligned_contiguous_read_spills_into_one_extra_sector():
    lanes = np.arange(32)
    assert sectors_per_request(16 + lanes * 4, access_bytes=4, machine=ampere()) == 5


def test_16_byte_vector_loads_by_eight_lanes_touch_four_sectors():
    lanes = np.arange(8)
    assert sectors_per_request(lanes * 16, access_bytes=16, machine=ampere()) == 4
