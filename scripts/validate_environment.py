"""
Startup Environment and Dependency Validation Script.

Validates system prerequisites, python dependencies, PyTorch CUDA/CPU execution device selection,
Ultralytics YOLO, scientific computing stack, and video capture/camera hardware accessibility.
"""

import argparse
import importlib.util
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("EnvironmentValidator")


def check_python_version() -> Tuple[bool, str]:
    """Validates Python runtime version (>= 3.8 required)."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    if sys.version_info >= (3, 8):
        return True, f"Python {version_str} (Compatible)"
    return False, f"Python {version_str} (Incompatible - Python 3.8+ required)"


def check_module_import(module_name: str, import_target: str = None) -> Tuple[bool, str]:
    """Checks if a module is importable and retrieves its version."""
    target = import_target or module_name
    spec = importlib.util.find_spec(target)
    if spec is None:
        return False, f"{module_name}: NOT INSTALLED"

    try:
        mod = importlib.import_module(target)
        version = getattr(mod, "__version__", "Installed")
        return True, f"{module_name}: v{version}"
    except Exception as e:
        return False, f"{module_name}: IMPORT ERROR ({e})"


def check_pytorch_and_cuda() -> Tuple[bool, str, str]:
    """
    Checks PyTorch, torchvision, and CUDA acceleration availability.
    Returns (Status, Status Details, Selected Device).
    """
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return False, "PyTorch: NOT INSTALLED", "cpu"

    try:
        import torch
        torch_ver = torch.__version__

        try:
            import torchvision
            tv_ver = torchvision.__version__
            torch_info = f"PyTorch v{torch_ver} | TorchVision v{tv_ver}"
        except ImportError:
            torch_info = f"PyTorch v{torch_ver} (TorchVision missing)"

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "N/A"
            details = (
                f"{torch_info}\n"
                f"  |- GPU Acceleration: AVAILABLE ({device_name})\n"
                f"  |- CUDA Version: {cuda_ver} | Devices: {device_count}"
            )
            selected_device = "cuda"
        else:
            details = (
                f"{torch_info}\n"
                f"  |- GPU Acceleration: UNAVAILABLE (CUDA not detected)\n"
                f"  |- Fallback Mode: Automatic CPU execution selected"
            )
            selected_device = "cpu"

        return True, details, selected_device

    except Exception as e:
        return False, f"PyTorch error: {e}", "cpu"


def check_camera_accessibility(camera_index: int = 0) -> Tuple[bool, str]:
    """Tests OpenCV camera hardware access."""
    cv2_spec = importlib.util.find_spec("cv2")
    if cv2_spec is None:
        return False, "Camera Check SKIPPED (OpenCV not installed)"

    try:
        import cv2
        # Use CAP_DSHOW backend on Windows for quick non-blocking camera probe
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            return False, f"Camera Index {camera_index}: NOT ACCESSIBLE (no hardware camera detected or in use)"

        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if ret and frame is not None:
            return True, f"Camera Index {camera_index}: ACCESSIBLE ({w}x{h} @ {fps:.1f} FPS)"
        return False, f"Camera Index {camera_index}: DETECTED BUT UNABLE TO CAPTURE FRAME"
    except Exception as e:
        return False, f"Camera check error: {e}"


def run_full_validation(camera_index: int = 0, skip_camera: bool = False) -> bool:
    """Runs end-to-end environment validation report."""
    print("\n" + "=" * 70)
    print(" [VALIDATION] WHEEL ALIGNMENT AI - STARTUP ENVIRONMENT VALIDATION REPORT")
    print("=" * 70)

    all_passed = True
    missing_packages = []

    # 1. Python Version
    py_ok, py_msg = check_python_version()
    print(f"[{'PASS' if py_ok else 'FAIL'}] Python Runtime: {py_msg}")
    if not py_ok:
        all_passed = False

    print("\n--- Core Computer Vision & Machine Learning Dependencies ---")
    
    # 2. PyTorch & CUDA check
    torch_ok, torch_msg, selected_device = check_pytorch_and_cuda()
    print(f"[{'PASS' if torch_ok else 'FAIL'}] {torch_msg}")
    if not torch_ok:
        all_passed = False
        missing_packages.append("torch torchvision")

    print(f"  -> Execution Device Selected: [{selected_device.upper()}]")

    # 3. Core Libraries Check
    libraries = [
        ("OpenCV", "cv2", "opencv-python"),
        ("Ultralytics YOLO", "ultralytics", "ultralytics"),
        ("NumPy", "numpy", "numpy"),
        ("Pandas", "pandas", "pandas"),
        ("Scikit-Learn", "sklearn", "scikit-learn"),
        ("Matplotlib", "matplotlib", "matplotlib"),
        ("Pillow (PIL)", "PIL", "pillow"),
        ("Joblib", "joblib", "joblib"),
        ("PyYAML", "yaml", "pyyaml")
    ]

    for display_name, module_name, pip_name in libraries:
        ok, msg = check_module_import(display_name, module_name)
        status_tag = "PASS" if ok else "WARN"
        print(f"[{status_tag}] {msg}")
        if not ok:
            missing_packages.append(pip_name)

    print("\n--- Hardware & Hardware Peripheral Verification ---")
    
    # 4. Camera Test
    if skip_camera:
        print("[INFO] Camera Check SKIPPED (--skip-camera requested)")
    else:
        cam_ok, cam_msg = check_camera_accessibility(camera_index)
        cam_tag = "PASS" if cam_ok else "INFO"
        print(f"[{cam_tag}] {cam_msg}")

    print("\n" + "=" * 70)
    if missing_packages:
        print(" [WARNING] ENVIRONMENT STATUS: MISSING DEPENDENCIES DETECTED")
        print("=" * 70)
        print("\nTo setup/repair your local Windows environment, run:")
        print("\n  1. Standard Installation (CPU / Default):")
        print("     pip install -r requirements.txt\n")
        print("  2. PyTorch CUDA GPU Installation (NVIDIA GPU users):")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("     pip install -r requirements.txt\n")
        print("  3. PyTorch CPU-Only Installation:")
        print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("     pip install -r requirements.txt\n")
    else:
        print(" [OK] ENVIRONMENT STATUS: ALL SYSTEMS OPERATIONAL")
        print(f" Ready for execution on device: [{selected_device.upper()}]")
        print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate execution environment and dependencies.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to test (default: 0)")
    parser.add_argument("--skip-camera", action="store_true", help="Skip camera hardware test")
    args = parser.parse_args()

    success = run_full_validation(camera_index=args.camera, skip_camera=args.skip_camera)
    sys.exit(0 if success else 1)
