"""Tests for the space-ablation table: same solver, three expressiveness tiers, many tile widths."""
from lego_synth.ablation import ablation_rows, to_markdown


def test_ablation_rows_cover_every_tier_for_every_width():
    rows = ablation_rows(widths=[24, 32], rows=32, elem_bytes=4)
    assert {(r["width"], r["tier"]) for r in rows} == {(24, "f2"), (24, "affine_xor"), (24, "full"),
                                                       (32, "f2"), (32, "affine_xor"), (32, "full")}
    for r in rows:
        assert r["footprint"] is not None and r["conflicts"] == 0


def test_full_tier_never_needs_more_footprint_than_the_narrower_tiers():
    rows = ablation_rows(widths=[24, 32], rows=32, elem_bytes=4)
    by = {(r["width"], r["tier"]): r["footprint"] for r in rows}
    for w in (24, 32):
        assert by[(w, "full")] <= by[(w, "affine_xor")] <= by[(w, "f2")]


def test_markdown_table_has_one_line_per_width():
    rows = ablation_rows(widths=[24], rows=32, elem_bytes=4)
    md = to_markdown(rows)
    assert md.count("\n| 24 ") == 1
