"""JIT NumPy transform hello world — CPU, no SymPy."""
import numpy as np
from lego.frontends.python_mlir import TiledPermute

layout = TiledPermute((8, 8), tile_shape=(4, 4))
data = np.arange(64, dtype=np.float32)
transformed = layout.transform(data)
back = layout.inverse_transform(transformed)
print(f"Round-trip match: {np.allclose(data.reshape(8, 8), back)}")
