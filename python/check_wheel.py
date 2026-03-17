"""Verify wheel: build, install into a temp venv, and test imports."""
import subprocess, sys, os, tempfile, glob

wheel_dir = os.path.join(os.path.dirname(__file__), "..", "build", "python_packages", "lego")
setup_py = os.path.join(wheel_dir, "setup.py")
if not os.path.exists(setup_py):
    print("ERROR: setup.py not found — run lego-prepare-wheel first", file=sys.stderr)
    sys.exit(1)

# Build wheel
print("[check-lego-wheel] Building wheel...")
dist_dir = os.path.join(wheel_dir, "dist")
subprocess.check_call(
    [sys.executable, "setup.py", "bdist_wheel"],
    cwd=wheel_dir, stdout=subprocess.DEVNULL,
)
wheels = glob.glob(os.path.join(dist_dir, "*.whl"))
if not wheels:
    print("ERROR: no wheel produced", file=sys.stderr)
    sys.exit(1)
print(f"[check-lego-wheel] Built: {os.path.basename(wheels[0])}")

# Install into temp dir and verify
print("[check-lego-wheel] Installing into temp directory...")
with tempfile.TemporaryDirectory() as tmp:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", wheels[0],
         "--target", tmp, "--no-deps", "-q"],
    )
    print("[check-lego-wheel] Verifying imports from wheel...")
    env = os.environ.copy()
    env["PYTHONPATH"] = tmp
    subprocess.check_call(
        [sys.executable, "-c",
         "from lego import jit; "
         "from lego.core import OrderBy; "
         "from lego.backend.compiler import LayoutCompiler; "
         "from lego.backend.dialects.lego_dialect import register; "
         "from lego.frontends.python_mlir import Tiled; "
         "print('[check-lego-wheel] Wheel verification OK')"],
        env=env,
    )
