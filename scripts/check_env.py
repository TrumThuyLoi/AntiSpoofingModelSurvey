#!/usr/bin/env python3
"""Kiểm tra môi trường khởi tạo dự án Face Anti-Spoofing Survey."""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MIN_PYTHON = (3, 10)

PROJECT_DIRS = (
    "configs",
    "data/raw",
    "data/sampled",
    "models",
    "notebooks",
    "reports",
    "reports/predictions",
    "reports/models",
    "reports/failure_cases",
    "scripts",
    "src",
    "src/datasets",
    "src/preprocessing",
    "src/models",
    "src/inference",
    "src/evaluation",
    "src/utils",
    "third_party",
    "docker",
    "label-studio",
)

SUBMODULE_REL = "third_party/Silent-Face-Anti-Spoofing"
SUBMODULE_MARKERS = (
    "src/anti_spoof_predict.py",
    "src/generate_patches.py",
    "src/model_lib/MiniFASNet.py",
    "resources/anti_spoof_models",
)

PREPROCESSING_MODULE = "src/preprocessing/minifasnet.py"

REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("PIL", "Pillow"),
    ("torch", "torch"),
    ("tqdm", "tqdm"),
    ("sklearn", "scikit-learn"),
    ("dotenv", "dotenv"),
    ("requests", "requests"),
    ("datasets", "datasets"),
    ("huggingface_hub", "huggingface_hub"),
)

OPTIONAL_PACKAGES: tuple[tuple[str, str, str], ...] = (
    ("torchvision", "torchvision", "cần cho một số model/pipeline ảnh nâng cao"),
    ("matplotlib", "matplotlib", "cần khi vẽ biểu đồ/figure báo cáo"),
    ("cv2", "opencv-python-headless", "cần cho MiniFASNet preprocessing"),
    ("label_studio", "label-studio", "chỉ cần khi chạy local Label Studio"),
)


@dataclass
class CheckResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def find_project_root() -> Path:
    """Tìm root repo (thư mục chứa configs/ và src/)."""
    candidates: list[Path] = []

    script_dir = Path(__file__).resolve().parent
    candidates.append(script_dir.parent)

    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    if cwd.name == "scripts" and cwd.parent.exists():
        candidates.append(cwd.parent)

    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        if (root / "configs").is_dir() and (root / "src").is_dir():
            return root

    raise FileNotFoundError(
        "Không tìm thấy project root (cần thư mục configs/ và src/). "
        "Chạy script từ root repo: python scripts/check_env.py"
    )


def check_python_version(result: CheckResult) -> None:
    version = sys.version_info[:3]
    label = f"{version[0]}.{version[1]}.{version[2]}"
    print(f"Python: {label} ({sys.executable})")

    if version[:2] < MIN_PYTHON:
        result.fail(
            f"Python {label} < yêu cầu {MIN_PYTHON[0]}.{MIN_PYTHON[1]} (AGENT.md)"
        )


def check_directories(root: Path, result: CheckResult) -> None:
    print("\n[Thư mục]")
    for rel_path in PROJECT_DIRS:
        path = root / rel_path
        if path.is_dir():
            print(f"  OK  {rel_path}/")
        else:
            result.fail(f"Thiếu thư mục: {rel_path}/")
            print(f"  FAIL  {rel_path}/")

    prep_file = root / PREPROCESSING_MODULE
    if prep_file.is_file():
        print(f"  OK  {PREPROCESSING_MODULE}")
    else:
        result.fail(f"Thiếu file: {PREPROCESSING_MODULE}")
        print(f"  FAIL  {PREPROCESSING_MODULE}")


def check_submodule(root: Path, result: CheckResult) -> None:
    print("\n[Submodule Silent-Face-Anti-Spoofing]")
    submodule = root / SUBMODULE_REL
    if not submodule.is_dir():
        result.warn(
            f"Thiếu submodule: {SUBMODULE_REL}/ — "
            "cần nếu chạy MiniFASNet (chạy: git submodule update --init --recursive)"
        )
        print(f"  WARN  {SUBMODULE_REL}/")
        return

    git_meta = submodule / ".git"
    if git_meta.is_file() or git_meta.is_dir():
        print(f"  OK  {SUBMODULE_REL}/ (.git)")
    else:
        result.warn(
            f"{SUBMODULE_REL}/ không có .git — có thể là bản copy thường, "
            "nên dùng git submodule"
        )
        print(f"  WARN  {SUBMODULE_REL}/ — không phải submodule git")

    for rel_marker in SUBMODULE_MARKERS:
        marker = submodule / rel_marker
        if marker.exists():
            print(f"  OK  {SUBMODULE_REL}/{rel_marker}")
        else:
            result.warn(f"Submodule thiếu: {SUBMODULE_REL}/{rel_marker}")
            print(f"  WARN  {SUBMODULE_REL}/{rel_marker}")


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Cần cài pyyaml để đọc config") from exc

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_absolute_path(value: str) -> bool:
    p = Path(value)
    return p.is_absolute() or (len(value) > 1 and value[1] == ":")


def check_config_file(root: Path, rel_path: str, result: CheckResult) -> dict[str, Any] | None:
    path = root / rel_path
    if not path.is_file():
        result.fail(f"Thiếu file config: {rel_path}")
        print(f"  FAIL  {rel_path}")
        return None

    try:
        data = _load_yaml(path)
    except Exception as exc:
        result.fail(f"Không đọc được {rel_path}: {exc}")
        print(f"  FAIL  {rel_path} — {exc}")
        return None

    if not isinstance(data, dict):
        result.fail(f"{rel_path}: nội dung YAML phải là mapping (dict)")
        print(f"  FAIL  {rel_path} — root không phải dict")
        return None

    print(f"  OK  {rel_path}")
    return data


def _check_path_fields(
    root: Path,
    config_rel: str,
    data: dict[str, Any],
    path_keys: tuple[str, ...],
    result: CheckResult,
    *,
    must_exist: bool = False,
) -> None:
    for key in path_keys:
        value = data.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            result.warn(f"{config_rel}: '{key}' nên là chuỗi path")
            continue
        if _is_absolute_path(value):
            result.fail(f"{config_rel}: '{key}' không được dùng absolute path: {value}")
        target = root / value
        if must_exist and not target.exists():
            result.warn(f"{config_rel}: path chưa tồn tại — {key}={value}")


def _check_weights_path(
    root: Path,
    weights_rel: str,
    result: CheckResult,
) -> None:
    """weights_dir có thể là file .pth hoặc thư mục chứa weight."""
    weights_path = root / weights_rel
    if weights_path.is_file():
        if weights_path.suffix.lower() == ".pth":
            print(f"  OK  weight file: {weights_rel}")
        else:
            result.warn(f"configs/model_minifasnet.yaml: weights_dir không phải .pth: {weights_rel}")
        return

    if weights_path.is_dir():
        pth_files = list(weights_path.glob("*.pth"))
        if pth_files:
            print(f"  OK  weight dir: {weights_rel} ({len(pth_files)} file .pth)")
        else:
            result.warn(
                f"Thư mục weight trống ({weights_rel}) — "
                "tải pretrained vào đây (xem MiniFASNet_GUIDE.md)"
            )
            print(f"  WARN  {weights_rel}/ — chưa có file .pth")
        return

    result.warn(
        f"Chưa có weight tại {weights_rel} — "
        "tải 2.7_80x80_MiniFASNetV2.pth (xem MiniFASNet_GUIDE.md)"
    )
    print(f"  WARN  weight chưa tồn tại: {weights_rel}")


def _validate_input_size(model: dict[str, Any], result: CheckResult) -> None:
    raw = model.get("input_size")
    if raw is None:
        result.fail("configs/model_minifasnet.yaml thiếu key: input_size")
        return
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        result.fail(f"configs/model_minifasnet.yaml: input_size phải là [height, width], nhận {raw!r}")
        return
    try:
        h, w = int(raw[0]), int(raw[1])
    except (TypeError, ValueError):
        result.fail(f"configs/model_minifasnet.yaml: input_size không phải số nguyên: {raw!r}")
        return
    if h <= 0 or w <= 0:
        result.fail(f"configs/model_minifasnet.yaml: input_size phải dương: {[h, w]}")


def check_configs(root: Path, result: CheckResult) -> dict[str, Any] | None:
    print("\n[Config YAML]")

    model = check_config_file(root, "configs/model_minifasnet.yaml", result)
    inference = check_config_file(root, "configs/inference.yaml", result)
    evaluation = check_config_file(root, "configs/evaluation.yaml", result)
    model_data: dict[str, Any] | None = model

    dataset_candidates = sorted((root / "configs").glob("dataset*.yaml"))
    if not dataset_candidates:
        result.warn("Không tìm thấy dataset config nào trong configs/ (dataset*.yaml)")
        print("  WARN  configs/dataset*.yaml")
    else:
        print(f"  OK  tìm thấy {len(dataset_candidates)} dataset config")
        for ds_path in dataset_candidates:
            rel = str(ds_path.relative_to(root))
            ds = check_config_file(root, rel, result)
            if not ds:
                continue
            for key in ("name", "annotation_path", "image_root", "source_dataset"):
                if key not in ds:
                    result.warn(f"{rel} thiếu key: {key}")
            _check_path_fields(
                root,
                rel,
                ds,
                ("annotation_path", "image_root"),
                result,
            )

    if model:
        for key in ("name", "weights_dir", "device", "threshold", "input_size"):
            if key not in model:
                result.fail(f"configs/model_minifasnet.yaml thiếu key: {key}")
        _validate_input_size(model, result)
        weights_dir = model.get("weights_dir")
        if isinstance(weights_dir, str):
            _check_path_fields(
                root,
                "configs/model_minifasnet.yaml",
                model,
                ("weights_dir",),
                result,
            )
            _check_weights_path(root, weights_dir, result)

    if inference:
        for key in ("dataset_config", "model_config", "output_dir", "batch_size"):
            if key not in inference:
                result.fail(f"configs/inference.yaml thiếu key: {key}")
        for ref_key in ("dataset_config", "model_config"):
            ref = inference.get(ref_key)
            if isinstance(ref, str) and not (root / ref).is_file():
                result.warn(
                    f"configs/inference.yaml: {ref_key} trỏ tới file không tồn tại: {ref}"
                )
        _check_path_fields(
            root,
            "configs/inference.yaml",
            inference,
            ("output_dir",),
            result,
        )

    if evaluation:
        if "thresholds" not in evaluation:
            result.fail("configs/evaluation.yaml thiếu key: thresholds")
        if not evaluation.get("model_config") and not evaluation.get("predictions_path"):
            result.fail(
                "configs/evaluation.yaml: cần model_config hoặc predictions_path",
            )
        if evaluation.get("predictions_path"):
            _check_path_fields(
                root,
                "configs/evaluation.yaml",
                evaluation,
                ("predictions_path",),
                result,
            )
        if evaluation.get("model_config"):
            ref = evaluation.get("model_config")
            if isinstance(ref, str) and not (root / ref).is_file():
                result.fail(
                    f"configs/evaluation.yaml: model_config không tồn tại: {ref}",
                )
        thresholds = evaluation.get("thresholds")
        if thresholds is not None and not isinstance(thresholds, list):
            result.fail("configs/evaluation.yaml: 'thresholds' phải là list")

    return model_data


def check_device_config(model: dict[str, Any] | None, result: CheckResult) -> None:
    """Cảnh báo khi config yêu cầu GPU nhưng CUDA không khả dụng."""
    if not model:
        return

    print("\n[Device vs CUDA]")
    device = str(model.get("device", "")).strip().lower()
    print(f"  configs/model_minifasnet.yaml device = {device!r}")

    if device not in ("gpu", "cuda"):
        print("  OK  device không yêu cầu CUDA")
        return

    try:
        import torch
    except ImportError:
        print("  SKIP  chưa import torch — bỏ qua kiểm tra CUDA")
        return

    cuda_ok = torch.cuda.is_available()
    print(f"  torch.cuda.is_available() = {cuda_ok}")
    if not cuda_ok:
        result.warn(
            "configs/model_minifasnet.yaml: device=gpu nhưng CUDA không khả dụng — "
            "đổi device: cpu trong configs/model_minifasnet.yaml hoặc cài driver/CUDA"
        )
        print("  WARN  GPU được cấu hình nhưng CUDA không sẵn sàng")
    else:
        print(f"  OK  CUDA — {torch.cuda.get_device_name(0)}")


def check_requirements_file(root: Path, result: CheckResult) -> None:
    print("\n[requirements.txt]")
    req_path = root / "requirements.txt"
    if not req_path.is_file():
        result.warn("Chưa có requirements.txt ở root repo")
        print("  WARN  requirements.txt không tồn tại")
        return

    content = req_path.read_text(encoding="utf-8").strip()
    if not content:
        result.warn("requirements.txt đang trống — chưa pin dependency")
        print("  WARN  requirements.txt trống")
    else:
        line_count = sum(1 for line in content.splitlines() if line.strip() and not line.strip().startswith("#"))
        print(f"  OK  requirements.txt ({line_count} dòng package)")


def check_imports(result: CheckResult) -> None:
    print("\n[Import packages — bắt buộc]")
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK  {module_name} ({pip_name}) — {version}")
        except ImportError:
            result.fail(f"Thiếu package: {pip_name} (import {module_name})")
            print(f"  FAIL  {module_name} — cài: pip install {pip_name}")

    print("\n[Import packages — tùy chọn]")
    for module_name, pip_name, reason in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK  {module_name} ({pip_name}) — {version}")
        except ImportError:
            result.warn(f"Thiếu package tùy chọn: {pip_name} ({reason})")
            print(f"  WARN  {module_name} — tùy chọn, cài nếu cần: pip install {pip_name}")


def print_summary(result: CheckResult) -> int:
    print("\n" + "=" * 60)
    if result.warnings:
        print("Cảnh báo:")
        for msg in result.warnings:
            print(f"  - {msg}")

    if result.errors:
        print("Lỗi:")
        for msg in result.errors:
            print(f"  - {msg}")
        print("\nKết quả: FAIL")
        return 1

    if result.warnings:
        print("\nKết quả: PASS (có cảnh báo)")
    else:
        print("\nKết quả: PASS")
    return 0


def main() -> int:
    result = CheckResult()

    print("Face Anti-Spoofing Survey — Environment Check")
    print(f"Platform: {platform.platform()}")

    try:
        root = find_project_root()
    except FileNotFoundError as exc:
        print(f"\nLỗi: {exc}")
        return 1

    print(f"Project root: {root}")

    check_python_version(result)
    check_directories(root, result)
    check_submodule(root, result)
    check_requirements_file(root, result)

    model_config: dict[str, Any] | None = None
    try:
        model_config = check_configs(root, result)
    except ImportError as exc:
        result.fail(str(exc))

    check_imports(result)
    check_device_config(model_config, result)

    return print_summary(result)


if __name__ == "__main__":
    sys.exit(main())
