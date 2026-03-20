"""Shared LEGO dialect op emission helpers.

Used by both compiler.py and symbolic.py — kept separate to avoid circular imports.
"""
import os

from mlir.ir import IndexType, IntegerAttr, Type, InsertionPoint
from mlir import ir
from mlir.dialects import arith as arith_dialect

_LEGO_DEBUG = os.environ.get("LEGO_DEBUG", "")


def _lego_layout_type():
    return Type.parse("!lego.layout")


def _index_const(val):
    idx_ty = IndexType.get()
    return arith_dialect.ConstantOp(idx_ty, IntegerAttr.get(idx_ty, int(val))).result


def _emit_reg_p(perm, dim_vals):
    from lego.backend.dialects.lego_dialect import RegPOp
    return RegPOp(result=_lego_layout_type(), perm=perm, dims=dim_vals).result


def _emit_row(dim_vals):
    from lego.backend.dialects.lego_dialect import RowOp
    return RowOp(result=_lego_layout_type(), dims=dim_vals).result


def _emit_col(dim_vals):
    from lego.backend.dialects.lego_dialect import ColOp
    return ColOp(result=_lego_layout_type(), dims=dim_vals).result


def _emit_order_by(perm_vals):
    from lego.backend.dialects.lego_dialect import OrderByOp
    return OrderByOp(result=_lego_layout_type(), perms=perm_vals).result


def _emit_group_by(dim_vals, obj_vals):
    from lego.backend.dialects.lego_dialect import GroupByOp
    return GroupByOp(result=_lego_layout_type(), group_dims=dim_vals, objects=obj_vals).result


def _emit_tile_by(input_val, tile_dim_vals, tile_shape):
    from lego.backend.dialects.lego_dialect import TileByOp
    return TileByOp(result=_lego_layout_type(), input=input_val,
                    tile_dims=tile_dim_vals, tile_shape=tile_shape).result


def _emit_apply(layout_val, indices):
    from lego.backend.dialects.lego_dialect import ApplyOp
    return ApplyOp(flat_index=IndexType.get(), layout=layout_val,
                   indices=list(indices)).result


def _emit_apply_inverse(layout_val, flat_index, rank):
    from lego.backend.dialects.lego_dialect import ApplyInverseOp
    idx_ty = IndexType.get()
    return list(ApplyInverseOp(indices=[idx_ty] * rank, layout=layout_val,
                               flat_index=flat_index).results)


def _emit_cast_view(layout_val, memref_val, data_type):
    from lego.backend.dialects.lego_dialect import CastViewOp
    view_ty = ir.Type.parse(f"!lego.view<{data_type}>")
    return CastViewOp(view=view_ty, memref=memref_val, layout=layout_val).result


def _emit_load(view, data_type, indices):
    from lego.backend.dialects.lego_dialect import LoadOp
    return LoadOp(result=data_type, view=view, indices=indices).result


def _emit_store(value, view, indices):
    from lego.backend.dialects.lego_dialect import StoreOp
    StoreOp(value=value, view=view, indices=indices)


def _emit_assume_bounds(val, lb=None, ub=None):
    from lego.backend.dialects._lego_ops_gen import assume_bounds
    assume_bounds(val, lb=lb, ub=ub)
