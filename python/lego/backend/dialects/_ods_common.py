# Re-export from the MLIR Python bindings package so that the
# auto-generated _lego_ops_gen.py (which does ``from ._ods_common import …``)
# resolves correctly when imported from the lego.backend.dialects namespace.
from mlir.dialects._ods_common import *  # noqa: F401,F403
from mlir.dialects._ods_common import _cext  # noqa: F401 — needed by _lego_ops_gen
