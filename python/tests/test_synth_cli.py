"""Tests for the `lego_synth.scripts.synth` CLI: one problem in, one certified layout out (JSON)."""
import json

from lego_synth.scripts.synth import synth_json


def test_synth_json_for_width_80_staging_reports_a_verified_exact_footprint_layout():
    out = json.loads(synth_json(kind="staging", rows=64, width=80, elem=2, tier="full", machine="ampere"))
    assert out["problem"] == {"kind": "staging", "rows": 64, "cols": 80, "elem_bytes": 2, "machine": "ampere",
                              "patterns": ["row_vector", "ldmatrix_x4"]}
    assert out["layout"]["type"] == "XorSwizzle" and out["layout"]["unit"] == 8
    assert out["footprint"] == 64 * 80
    assert out["verified"]["extra_wavefronts"] == 0 and out["verified"]["bijective"] is True
    assert out["cuda_index_expr"].startswith("((i)*80")
    assert out["solve_time_s"] > 0


def test_synth_json_reports_none_when_the_tier_has_no_solution():
    out = json.loads(synth_json(kind="staging", rows=64, width=80, elem=2, tier="affine_xor", machine="ampere"))
    assert out["layout"] is None
