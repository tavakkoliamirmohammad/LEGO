"""Verify wheel: build, install into a temp venv, and test imports."""
import subprocess, sys, os, tempfile, glob, shutil

wheel_dir = os.path.join(os.path.dirname(__file__), "..", "build", "python_packages", "lego")
pyproject = os.path.join(wheel_dir, "pyproject.toml")
if not os.path.exists(pyproject):
    print("ERROR: pyproject.toml not found — run lego-prepare-wheel first", file=sys.stderr)
    sys.exit(1)

# Clean previous builds
dist_dir = os.path.join(wheel_dir, "dist")
for d in [dist_dir, os.path.join(wheel_dir, "build")]:
    if os.path.exists(d):
        shutil.rmtree(d)

# Build wheel using pip wheel with PEP 517
print("[check-lego-wheel] Building wheel...")
subprocess.check_call(
    [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--use-pep517",
     "-w", "dist", "-q"],
    cwd=wheel_dir,
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
        [sys.executable, "-P", os.path.join(os.path.dirname(__file__), "check_imports.py")],
        env=env,
    )
    print("[check-lego-wheel] Wheel verification OK")
