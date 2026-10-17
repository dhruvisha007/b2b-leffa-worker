"""Build-time import sanity: try every Leffa/densepose import, print ALL failures, exit non-zero if any."""
import sys, traceback

sys.path.insert(0, "/Leffa")

CHECKS = [
    ("detectron2.engine", "DefaultPredictor"),
    ("densepose", "add_densepose_config"),
    ("leffa.model", "LeffaModel"),
    ("leffa.inference", "LeffaInference"),
    ("leffa.transform", "LeffaTransform"),
    ("leffa_utils.densepose_predictor", "DensePosePredictor"),
    ("leffa_utils.utils", "resize_and_center"),
    ("leffa_utils.utils", "get_agnostic_mask_hd"),
    ("preprocess.humanparsing.run_parsing", "Parsing"),
    ("preprocess.openpose.run_openpose", "OpenPose"),
]

failed = False

# ABI guard: torch 2.1.1 is built against numpy 1.x. A later pip step must not have pulled numpy 2,
# or torch.from_numpy raises "Numpy is not available" at runtime (detectron2 DensePose weight load).
try:
    import numpy, torch
    assert numpy.__version__.startswith("1."), f"numpy must stay 1.x, got {numpy.__version__}"
    assert int(torch.from_numpy(numpy.arange(3)).sum()) == 3
    print(f"OK   numpy ABI ({numpy.__version__}) + torch.from_numpy")
except Exception:
    failed = True
    print("FAIL numpy/torch ABI")
    traceback.print_exc()
    print("-" * 60)

for mod, attr in CHECKS:
    try:
        m = __import__(mod, fromlist=[attr])
        getattr(m, attr)
        print(f"OK   {mod}.{attr}")
    except Exception:
        failed = True
        print(f"FAIL {mod}.{attr}")
        traceback.print_exc()
        print("-" * 60)

sys.exit(1 if failed else 0)
