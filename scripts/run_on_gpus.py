"""Wrapper to set CUDA_VISIBLE_DEVICES before any CUDA imports."""
import os
import sys

if __name__ == "__main__":
    # First arg is GPU list, rest are forwarded to the module
    if any(a in ("-h", "--help") for a in sys.argv[1:2]):
        print(__doc__.strip() if __doc__ else "usage: run_on_gpus.py <gpu_ids> <module> [args...]")
        print("  e.g. python scripts/run_on_gpus.py 5,7 src.experiments.exp1_detection.run --model foo")
        sys.exit(0)
    if len(sys.argv) < 3:
        print("Usage: python scripts/run_on_gpus.py GPU_IDS MODULE [ARGS...]")
        print("  e.g. python scripts/run_on_gpus.py 5,7 src.experiments.exp1_detection.run --model foo")
        sys.exit(1)

    gpu_ids = sys.argv[1]
    module = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # Ensure project root is on the path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Remove our args so the target module sees only its own args
    sys.argv = sys.argv[2:]

    # Run the module
    import runpy
    runpy.run_module(module, run_name="__main__")
