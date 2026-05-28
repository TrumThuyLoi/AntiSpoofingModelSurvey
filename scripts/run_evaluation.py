#!/usr/bin/env python3
"""CLI tính metrics anti-spoofing từ file predictions CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

VALID_LABELS = {"live", "spoof"}
_RUN_STEM_RE = re.compile(r"^run_(.+)_\d{8}_\d{6}$")


@dataclass
class EvalStats:
    total_rows: int = 0
    dropped_error_rows: int = 0
    dropped_invalid_label_rows: int = 0
    kept_rows: int = 0
    kept_live_rows: int = 0
    kept_spoof_rows: int = 0


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML root phải là dict")
    return data


def _resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def parse_predictions_metadata(path: Path) -> dict[str, str]:
    """Đọc dòng comment `# key=value` ở đầu predictions CSV (giống run_batch)."""
    meta: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped.startswith("#"):
                break
            body = stripped[1:].strip()
            if "=" in body:
                key, _, value = body.partition("=")
                meta[key.strip()] = value.strip()
    return meta


def resolve_dataset_slug(predictions_path: Path, override: str | None = None) -> str:
    """
    Suy ra tên dataset cho thư mục metrics.

    Ưu tiên: --dataset / config → *_latest.csv → run_*_timestamp → # source_dataset=
    """
    if override and override.strip():
        return override.strip()

    stem = predictions_path.stem
    if stem.endswith("_latest"):
        return stem[: -len("_latest")]

    run_match = _RUN_STEM_RE.match(stem)
    if run_match:
        return run_match.group(1)

    meta = parse_predictions_metadata(predictions_path)
    if meta.get("source_dataset"):
        return meta["source_dataset"]

    raise ValueError(
        f"Không suy ra được dataset từ {predictions_path}. "
        "Dùng --dataset, file <dataset>_latest.csv, run_<dataset>_YYYYMMDD_HHMMSS.csv, "
        "hoặc CSV có dòng # source_dataset=...",
    )


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_prediction_rows(path: Path, stats: EvalStats) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        csv_lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(csv_lines)

    required_columns = {"label_true", "live_score", "error"}
    if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
        raise ValueError(
            f"{path}: thiếu cột bắt buộc. Cần có ít nhất: {sorted(required_columns)}",
        )

    for row in reader:
        stats.total_rows += 1

        error_value = (row.get("error") or "").strip()
        if error_value:
            stats.dropped_error_rows += 1
            continue

        label_true = (row.get("label_true") or "").strip().lower()
        if label_true not in VALID_LABELS:
            stats.dropped_invalid_label_rows += 1
            continue

        live_score_text = (row.get("live_score") or "").strip()
        try:
            live_score = float(live_score_text)
        except ValueError:
            stats.dropped_invalid_label_rows += 1
            continue

        clean_row = {
            "label_true": label_true,
            "live_score": str(live_score),
        }
        rows.append(clean_row)

    stats.kept_rows = len(rows)
    stats.kept_live_rows = sum(1 for r in rows if r["label_true"] == "live")
    stats.kept_spoof_rows = sum(1 for r in rows if r["label_true"] == "spoof")
    return rows


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _evaluate_threshold(rows: list[dict[str, str]], threshold: float) -> dict[str, float | int]:
    tp_live = 0
    fn_live = 0
    fp_live = 0
    tn_live = 0

    for row in rows:
        label_true = row["label_true"]
        live_score = float(row["live_score"])
        label_pred = "live" if live_score >= threshold else "spoof"
        label_pred = label_pred.lower()

        if label_true == "live" and label_pred == "live":
            tp_live += 1
        elif label_true == "live" and label_pred == "spoof":
            fn_live += 1
        elif label_true == "spoof" and label_pred == "live":
            fp_live += 1
        else:
            tn_live += 1

    total = tp_live + fn_live + fp_live + tn_live
    n_live = tp_live + fn_live
    n_spoof = fp_live + tn_live

    accuracy = _safe_div(tp_live + tn_live, total)
    precision_live = _safe_div(tp_live, tp_live + fp_live)
    recall_live = _safe_div(tp_live, tp_live + fn_live)
    f1_live = _safe_div(2 * precision_live * recall_live, precision_live + recall_live)

    tp_spoof = tn_live
    fp_spoof = fn_live
    fn_spoof = fp_live
    precision_spoof = _safe_div(tp_spoof, tp_spoof + fp_spoof)
    recall_spoof = _safe_div(tp_spoof, tp_spoof + fn_spoof)
    f1_spoof = _safe_div(2 * precision_spoof * recall_spoof, precision_spoof + recall_spoof)

    apcer = _safe_div(fp_live, n_spoof)
    bpcer = _safe_div(fn_live, n_live)
    acer = 0.5 * (apcer + bpcer)

    return {
        "threshold": threshold,
        "num_samples": total,
        "num_live": n_live,
        "num_spoof": n_spoof,
        "cm_true_live_pred_live": tp_live,
        "cm_true_live_pred_spoof": fn_live,
        "cm_true_spoof_pred_live": fp_live,
        "cm_true_spoof_pred_spoof": tn_live,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "accuracy": accuracy,
        "precision_live": precision_live,
        "recall_live": recall_live,
        "f1_live": f1_live,
        "precision_spoof": precision_spoof,
        "recall_spoof": recall_spoof,
        "f1_spoof": f1_spoof,
    }


def _write_summary_csv(path: Path, stats: EvalStats, results: list[dict[str, float | int]]) -> None:
    fieldnames = [
        "threshold",
        "num_samples",
        "num_live",
        "num_spoof",
        "dropped_error_rows",
        "dropped_invalid_label_rows",
        "cm_true_live_pred_live",
        "cm_true_live_pred_spoof",
        "cm_true_spoof_pred_live",
        "cm_true_spoof_pred_spoof",
        "apcer",
        "bpcer",
        "acer",
        "accuracy",
        "precision_live",
        "recall_live",
        "f1_live",
        "precision_spoof",
        "recall_spoof",
        "f1_spoof",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            row_out = dict(row)
            row_out["dropped_error_rows"] = stats.dropped_error_rows
            row_out["dropped_invalid_label_rows"] = stats.dropped_invalid_label_rows
            writer.writerow(row_out)


def _write_threshold_json(
    path: Path,
    threshold_result: dict[str, float | int],
    stats: EvalStats,
    provenance: dict[str, str],
) -> None:
    payload = {
        "provenance": provenance,
        "filtering": {
            "total_rows": stats.total_rows,
            "dropped_error_rows": stats.dropped_error_rows,
            "dropped_invalid_label_rows": stats.dropped_invalid_label_rows,
            "kept_rows": stats.kept_rows,
            "kept_live_rows": stats.kept_live_rows,
            "kept_spoof_rows": stats.kept_spoof_rows,
        },
        "metrics": threshold_result,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _plot_confusion_matrix(path: Path, result: dict[str, float | int], threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = [
        [
            int(result["cm_true_live_pred_live"]),
            int(result["cm_true_live_pred_spoof"]),
        ],
        [
            int(result["cm_true_spoof_pred_live"]),
            int(result["cm_true_spoof_pred_spoof"]),
        ],
    ]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred_live", "pred_spoof"])
    ax.set_yticklabels(["true_live", "true_spoof"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Confusion Matrix @ threshold={threshold}")

    max_val = max(max(row) for row in cm) if cm else 0
    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            color = "white" if value > max_val / 2 else "black"
            ax.text(j, i, f"{value}", ha="center", va="center", color=color, fontsize=12)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _threshold_suffix(threshold: float) -> str:
    return f"{threshold:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tính metric anti-spoofing từ predictions CSV.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/evaluation.yaml",
        help="Đường dẫn evaluation config (mặc định: configs/evaluation.yaml).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Override predictions_path trong evaluation config.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Tên dataset cho thư mục output (override auto-detect từ predictions).",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Ghi thẳng vào metrics_output_dir (không tạo subfolder theo dataset).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(args.config)

    predictions_path = args.predictions if args.predictions else _resolve_path(str(cfg["predictions_path"]))
    if not predictions_path.is_absolute():
        predictions_path = _resolve_path(str(predictions_path))

    metrics_base = _resolve_path(str(cfg["metrics_output_dir"]))
    thresholds_raw = cfg.get("thresholds", [])
    thresholds = [float(t) for t in thresholds_raw]
    dataset_override = args.dataset or cfg.get("dataset")
    flat_output = args.flat_output or bool(cfg.get("flat_output"))

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy predictions file: {predictions_path}. "
            "Dùng --predictions để chỉ đúng file cần đánh giá.",
        )
    if not thresholds:
        raise ValueError("Cấu hình thresholds rỗng.")

    dataset_slug = resolve_dataset_slug(predictions_path, dataset_override)
    output_dir = metrics_base if flat_output else metrics_base / dataset_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "source_dataset": dataset_slug,
        "predictions_path": _repo_relative_path(predictions_path),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    stats = EvalStats()
    rows = _load_prediction_rows(predictions_path, stats)
    if not rows:
        raise ValueError("Không còn dữ liệu hợp lệ sau khi lọc.")

    results = [_evaluate_threshold(rows, t) for t in thresholds]

    summary_path = output_dir / "metrics_summary.csv"
    _write_summary_csv(summary_path, stats, results)

    saved_json_paths: list[Path] = []
    saved_cm_paths: list[Path] = []
    for result in results:
        threshold = float(result["threshold"])
        suffix = _threshold_suffix(threshold)

        json_path = output_dir / f"metrics_threshold_{suffix}.json"
        _write_threshold_json(json_path, result, stats, provenance)
        saved_json_paths.append(json_path)

        cm_path = output_dir / f"confusion_matrix_{suffix}.png"
        _plot_confusion_matrix(cm_path, result, threshold=threshold)
        saved_cm_paths.append(cm_path)

    print(f"dataset: {dataset_slug}")
    print(f"output_dir: {output_dir}")
    print(f"predictions: {predictions_path}")
    print(f"kept_rows: {stats.kept_rows}")
    print(f"dropped_error_rows: {stats.dropped_error_rows}")
    print(f"dropped_invalid_label_rows: {stats.dropped_invalid_label_rows}")
    print(f"saved: {summary_path}")
    for path in saved_json_paths:
        print(f"saved: {path}")
    for path in saved_cm_paths:
        print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
