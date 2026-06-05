"""Batch inference runner — đọc annotation CSV, gọi MiniFASNetWrapper, ghi predictions."""

from __future__ import annotations

import sys
import csv
import json
import shutil
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.minifasnet import MiniFASNetWrapper
from src.models.vit_fas import ViTFASWrapper
from src.models.face_antispoof_onnx import FaceAntispoofONNXWrapper
from src.models.hairymax_onnx import HairymaxONNXWrapper
from src.reports.layout import model_report_paths

BASE_FIELDNAMES = (
    "image_path",
    "label_true",
    "label_pred",
    "live_score",
    "spoof_score",
    "error",
)


def _repo_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs").is_dir():
        raise FileNotFoundError(f"Không tìm thấy configs/ tại {root}")
    return root


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML root phải là dict")
    return data


def _resolve_path(repo_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else repo_root / p


def load_batch_configs(
    dataset_config_path: Path,
    model_config_path: Path,
    inference_config_path: Path,
    *,
    repo_root: Path | None = None,
    annotation_csv: Path | None = None,
) -> dict[str, Any]:
    """
    Đọc 3 file config và xác định đường dẫn annotation CSV.

    ``annotation_csv`` override ``annotation_path`` trong dataset config
    (dùng khi chạy casia_fasd hoặc celeba_spoof mà chưa đổi dataset.yaml).
    """
    root = repo_root or _repo_root()
    dataset_cfg = _load_yaml(_resolve_path(root, str(dataset_config_path)))
    model_cfg = _load_yaml(_resolve_path(root, str(model_config_path)))
    inference_cfg = _load_yaml(_resolve_path(root, str(inference_config_path)))

    if annotation_csv is not None:
        ann_path = _resolve_path(root, str(annotation_csv))
    else:
        ann_path = _resolve_path(root, str(dataset_cfg["annotation_path"]))

    return {
        "repo_root": root,
        "dataset": dataset_cfg,
        "model": model_cfg,
        "inference": inference_cfg,
        "annotation_path": ann_path,
        "output_dir": _resolve_path(root, str(inference_cfg["output_dir"])),
        "batch_size": int(inference_cfg.get("batch_size", 32)),
        "save_raw_output": bool(inference_cfg.get("save_raw_output", False)),
    }


def _is_valid_row(row: dict[str, str]) -> bool:
    return str(row.get("is_valid", "")).strip().lower() == "true"


def _resolve_image_path(repo_root: Path, image_path: str) -> Path:
    p = Path(image_path)
    return p if p.is_absolute() else repo_root / p


def _read_annotation_rows(annotation_path: Path) -> list[dict[str, str]]:
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy annotation CSV: {annotation_path}")
    with annotation_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _metadata_lines(meta: dict[str, Any]) -> list[str]:
    return [f"# {key}={value}" for key, value in meta.items()]


def _empty_result_row(rel_path: str, label_true: str) -> dict[str, Any]:
    return {
        "image_path": rel_path,
        "label_true": label_true,
        "label_pred": "",
        "live_score": "",
        "spoof_score": "",
        "error": "",
    }


def _result_row_from_scores(
    rel_path: str,
    label_true: str,
    live_score: float,
    spoof_score: float,
    label_pred: str,
    raw_output: Any | None,
    *,
    save_raw_output: bool,
) -> dict[str, Any]:
    row = _empty_result_row(rel_path, label_true)
    row["label_pred"] = label_pred
    row["live_score"] = live_score
    row["spoof_score"] = spoof_score
    if save_raw_output and raw_output is not None:
        row["raw_output"] = json.dumps(raw_output)
    return row


def _predict_chunk(
    wrapper: Any,
    chunk: list[dict[str, str]],
    repo_root: Path,
    *,
    save_raw_output: bool,
) -> list[dict[str, Any]]:
    """Predict theo wrapper chuẩn hóa interface (load/predict/predict_batch)."""
    output_rows: list[dict[str, Any]] = []
    for row in chunk:
        rel_path = row["image_path"]
        label_true = row.get("label", "")
        try:
            abs_path = _resolve_image_path(repo_root, rel_path)
            pred = wrapper.predict(abs_path)
            output_rows.append(
                _result_row_from_scores(
                    rel_path=rel_path,
                    label_true=label_true,
                    live_score=float(pred["live_score"]),
                    spoof_score=float(pred["spoof_score"]),
                    label_pred=str(pred["label_pred"]),
                    raw_output=pred.get("raw_output"),
                    save_raw_output=save_raw_output,
                )
            )
        except Exception as exc:
            err_row = _empty_result_row(rel_path, label_true)
            err_row["error"] = str(exc)
            output_rows.append(err_row)
    return output_rows


def _build_model_wrapper(model_cfg: dict[str, Any], model_config_abs: Path):
    model_name = str(model_cfg.get("name", "")).strip().lower()
    device_cfg = str(model_cfg.get("device", "cpu")).strip().lower()
    prefer_cpu = device_cfg == "cpu"

    if model_name == "minifasnet":
        wrapper = MiniFASNetWrapper(model_config_path=model_config_abs, prefer_cpu=prefer_cpu)
        wrapper.load()
        return wrapper
    if model_name == "vit_fas":
        wrapper = ViTFASWrapper(
            weights_path=model_cfg["weights_dir"],
            input_size=tuple(model_cfg.get("input_size", [224, 224])),
            threshold=float(model_cfg.get("threshold", 0.5)),
            prefer_cpu=prefer_cpu,
        )
        wrapper.load()
        return wrapper
    if model_name == "face_antispoof_onnx":
        wrapper = FaceAntispoofONNXWrapper(
            weights_path=model_cfg["weights_dir"],
            input_size=tuple(model_cfg.get("input_size", [128, 128])),
            threshold=float(model_cfg.get("threshold", 0.5)),
            prefer_cpu=prefer_cpu,
        )
        wrapper.load()
        return wrapper
    if model_name == "hairymax_onnx":
        wrapper = HairymaxONNXWrapper(
            weights_path=model_cfg["weights_dir"],
            input_size=tuple(model_cfg.get("input_size", [128, 128])),
            threshold=float(model_cfg.get("threshold", 0.5)),
            prefer_cpu=prefer_cpu,
        )
        wrapper.load()
        return wrapper
    raise ValueError(
        f"Model chưa được hỗ trợ trong runner: name={model_name!r}. "
        "Hỗ trợ: 'minifasnet', 'vit_fas', 'face_antispoof_onnx'."
    )


def run_batch_inference(
    dataset_config_path: Path | str = "configs/dataset.yaml",
    model_config_path: Path | str = "configs/model_minifasnet.yaml",
    inference_config_path: Path | str = "configs/inference.yaml",
    *,
    annotation_csv: Path | str | None = None,
    repo_root: Path | None = None,
) -> Path:
    """
    Chạy batch inference trên annotation CSV và ghi ``run_YYYYMMDD_HHMMSS.csv``.

    Trả về đường dẫn file predictions vừa ghi.
    """
    ann_override = Path(annotation_csv) if annotation_csv is not None else None
    cfg = load_batch_configs(
        Path(dataset_config_path),
        Path(model_config_path),
        Path(inference_config_path),
        repo_root=repo_root,
        annotation_csv=ann_override,
    )

    root: Path = cfg["repo_root"]
    model_cfg: dict[str, Any] = cfg["model"]
    dataset_cfg: dict[str, Any] = cfg["dataset"]
    inference_cfg: dict[str, Any] = cfg["inference"]

    source_dataset = str(dataset_cfg.get("source_dataset", "")).strip()
    if not source_dataset:
        raise ValueError("dataset config thiếu source_dataset")

    report_paths = model_report_paths(model_cfg, root)
    cfg["output_dir"] = report_paths.predictions_dir(source_dataset)

    model_config_abs = _resolve_path(root, str(model_config_path))
    wrapper = _build_model_wrapper(model_cfg, model_config_abs)

    rows_in = _read_annotation_rows(cfg["annotation_path"])
    valid_rows = [r for r in rows_in if _is_valid_row(r)]
    skipped_invalid = len(rows_in) - len(valid_rows)

    fieldnames = list(BASE_FIELDNAMES)
    if cfg["save_raw_output"]:
        fieldnames.append("raw_output")

    started_at = datetime.now()
    output_rows: list[dict[str, Any]] = []
    num_errors = 0

    batch_size = max(1, cfg["batch_size"])
    with tqdm(total=len(valid_rows), desc="Inference", unit="img") as pbar:
        for i in range(0, len(valid_rows), batch_size):
            chunk = valid_rows[i : i + batch_size]
            for out_row in _predict_chunk(
                wrapper,
                chunk,
                root,
                save_raw_output=cfg["save_raw_output"],
            ):
                if out_row["error"]:
                    num_errors += 1
                output_rows.append(out_row)
                pbar.update(1)

    finished_at = datetime.now()
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    out_name = f"run_{dataset_cfg.get('source_dataset', '')}_{started_at.strftime('%Y%m%d_%H%M%S')}.csv"
    out_path = cfg["output_dir"] / out_name

    metadata = {
        "model_id": report_paths.model_id,
        "model_name": model_cfg.get("name", ""),
        "weights_path": str(wrapper.weights_path),
        "threshold": model_cfg.get("threshold", ""),
        "device": str(wrapper.device),
        "dataset_config": str(dataset_config_path),
        "model_config": str(model_config_path),
        "inference_config": str(inference_config_path),
        "annotation_path": str(cfg["annotation_path"]),
        "dataset_name": dataset_cfg.get("name", ""),
        "source_dataset": dataset_cfg.get("source_dataset", ""),
        "batch_size": batch_size,
        "save_raw_output": cfg["save_raw_output"],
        "num_input_rows": len(rows_in),
        "num_valid_rows": len(valid_rows),
        "num_skipped_invalid": skipped_invalid,
        "num_errors": num_errors,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
    }

    report_paths.write_manifest(
        {
            "model_id": report_paths.model_id,
            "model_name": metadata["model_name"],
            "weights_path": metadata["weights_path"],
            "threshold": metadata["threshold"],
            "device": metadata["device"],
            "source_dataset": source_dataset,
            "last_run_csv": out_name,
            "updated_at": metadata["finished_at"],
        }
    )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        for line in _metadata_lines(metadata):
            f.write(line + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    latest_path = report_paths.predictions_latest(source_dataset)
    shutil.copy2(out_path, latest_path)
    return latest_path


if __name__ == "__main__":
    out = run_batch_inference(
        annotation_csv="data/sampled/casia_fasd_sample.csv",
    )
    print(out)