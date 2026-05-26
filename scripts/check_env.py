#!/usr/bin/env python3
"""Kiểm tra môi trường khởi tạo dự án MiniFASNet Anti-Spoofing Survey."""

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
    "data/labeled",
    "data/processed",
    "models",
    "models/minifasnet",
    "notebooks",
    "reports",
    "reports/predictions",
    "reports/metrics",
    "reports/failure_cases",
    "scripts",
    "src",
    "src/datasets",
    "src/models",
    "src/inference",
    "src/evaluation",
    "src/utils",
    "docker",
    "label-studio",
)

CONFIG_FILES = (
    "configs/dataset.yaml",
    "configs/model.yaml",
    "configs/inference.yaml",
    "configs/evaluation.yaml",
)

REQUIRED_PACKAGES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "Pillow"),
)

OPTIONAL_PACKAGES: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
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


def check_configs(root: Path, result: CheckResult) -> None:
    print("\n[Config YAML]")

    dataset = check_config_file(root, "configs/dataset.yaml", result)
    model = check_config_file(root, "configs/model.yaml", result)
    inference = check_config_file(root, "configs/inference.yaml", result)
    evaluation = check_config_file(root, "configs/evaluation.yaml", result)

    if dataset:
        for key in ("name", "annotation_path", "image_root", "source_dataset"):
            if key not in dataset:
                result.fail(f"configs/dataset.yaml thiếu key: {key}")
        _check_path_fields(
            root,
            "configs/dataset.yaml",
            dataset,
            ("annotation_path", "image_root"),
            result,
        )

    if model:
        for key in ("name", "weights_dir", "device", "threshold"):
            if key not in model:
                result.fail(f"configs/model.yaml thiếu key: {key}")
        weights_dir = model.get("weights_dir")
        if isinstance(weights_dir, str):
            _check_path_fields(
                root,
                "configs/model.yaml",
                model,
                ("weights_dir",),
                result,
            )
            weights_path = root / weights_dir
            if weights_path.is_dir() and not any(weights_path.iterdir()):
                result.warn(
                    f"models/minifasnet: thư mục weight trống "
                    f"({weights_dir}) — bình thường trước khi tải pretrained"
                )

    if inference:
        for key in ("dataset_config", "model_config", "output_dir", "batch_size"):
            if key not in inference:
                result.fail(f"configs/inference.yaml thiếu key: {key}")
        for ref_key in ("dataset_config", "model_config"):
            ref = inference.get(ref_key)
            if isinstance(ref, str) and not (root / ref).is_file():
                result.fail(
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
        for key in ("predictions_path", "metrics_output_dir", "thresholds"):
            if key not in evaluation:
                result.fail(f"configs/evaluation.yaml thiếu key: {key}")
        _check_path_fields(
            root,
            "configs/evaluation.yaml",
            evaluation,
            ("predictions_path", "metrics_output_dir"),
            result,
        )
        thresholds = evaluation.get("thresholds")
        if thresholds is not None and not isinstance(thresholds, list):
            result.fail("configs/evaluation.yaml: 'thresholds' phải là list")


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
    for module_name, pip_name in OPTIONAL_PACKAGES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK  {module_name} ({pip_name}) — {version}")
            if module_name == "torch":
                cuda = mod.cuda.is_available()
                print(f"        torch.cuda.is_available() = {cuda}")
        except ImportError:
            result.warn(f"Chưa cài (tùy chọn): {pip_name}")
            print(f"  SKIP  {module_name} — pip install {pip_name}")


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

    print("MiniFASNet Anti-Spoofing Survey — Environment Check")
    print(f"Platform: {platform.platform()}")

    try:
        root = find_project_root()
    except FileNotFoundError as exc:
        print(f"\nLỗi: {exc}")
        return 1

    print(f"Project root: {root}")

    check_python_version(result)
    check_directories(root, result)
    check_requirements_file(root, result)

    try:
        check_configs(root, result)
    except ImportError as exc:
        result.fail(str(exc))

    check_imports(result)

    return print_summary(result)


if __name__ == "__main__":
    sys.exit(main())
