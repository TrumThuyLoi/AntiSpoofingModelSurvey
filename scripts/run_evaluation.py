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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reports.layout import (  # noqa: E402
    ModelReportPaths,
    resolve_model_id,
    sanitize_model_id,
)

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


def discover_config_paths(repo_root: Path, prefix: str) -> list[Path]:
    """Liệt kê configs/<prefix>_*.yaml (sorted)."""
    return sorted((repo_root / "configs").glob(f"{prefix}_*.yaml"))


def source_dataset_from_config(dataset_config: Path) -> str:
    slug = _load_yaml(dataset_config).get("source_dataset")
    if not slug or not str(slug).strip():
        raise ValueError(f"{dataset_config}: thiếu source_dataset")
    return str(slug).strip()


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
    if stem == "latest":
        parent = predictions_path.parent
        if parent.parent.name == "predictions":
            return parent.name

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


def resolve_model_id_for_eval(
    *,
    predictions_path: Path | None,
    eval_cfg: dict[str, Any],
    cli_model_id: str | None,
) -> str | None:
    """model_id: CLI → eval config → metadata CSV → derive từ model_config."""
    if cli_model_id and cli_model_id.strip():
        return sanitize_model_id(cli_model_id)
    if eval_cfg.get("model_id") and str(eval_cfg["model_id"]).strip():
        return sanitize_model_id(str(eval_cfg["model_id"]))
    if predictions_path is not None and predictions_path.is_file():
        meta = parse_predictions_metadata(predictions_path)
        if meta.get("model_id"):
            return sanitize_model_id(meta["model_id"])
    model_config = eval_cfg.get("model_config")
    if model_config:
        model_cfg = _load_yaml(_resolve_path(str(model_config)))
        return resolve_model_id(model_cfg, REPO_ROOT)
    return None


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

        clean_row: dict[str, str] = {
            "label_true": label_true,
            "live_score": str(live_score),
        }
        image_path = (row.get("image_path") or "").strip()
        if image_path:
            clean_row["image_path"] = image_path
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


def _thresholds_for_curve_plot(configured: list[float]) -> list[float]:
    """Ngưỡng dày cho biểu đồ; CSV/JSON vẫn dùng `configured` từ config."""
    dense = [round(i * 0.01, 2) for i in range(101)]
    extra = [float(t) for t in configured]
    return sorted(set(dense + extra))


def _scores_by_true_label(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    live_scores = [float(r["live_score"]) for r in rows if r["label_true"] == "live"]
    spoof_scores = [float(r["live_score"]) for r in rows if r["label_true"] == "spoof"]
    return live_scores, spoof_scores


LIVE_ACCEPT_THRESHOLD = 0.5
MONTAGE_COLS = 6
MONTAGE_ROWS = 4
MONTAGE_MAX_IMAGES = MONTAGE_COLS * MONTAGE_ROWS
# Bin §7.5.2 histogram toàn tập live trên drivers (step 0.1)
LIVE_SCORE_HISTOGRAM_BIN_STEP = 0.1
LIVE_SCORE_COARSE_BIN_EDGES = [
    round(i * LIVE_SCORE_HISTOGRAM_BIN_STEP, 2) for i in range(int(1 / LIVE_SCORE_HISTOGRAM_BIN_STEP) + 1)
]


def _live_scores_from_rows(rows: list[dict[str, str]]) -> list[float]:
    return [float(r["live_score"]) for r in rows if r["label_true"] == "live"]


def _is_drivers_dataset_slug(dataset_slug: str) -> bool:
    return dataset_slug.startswith("drivers_250_fn")


def _resolve_repo_image_path(image_path: str) -> Path:
    p = Path(image_path)
    if p.is_file():
        return p
    return REPO_ROOT / image_path


def _misclassified_rows_at_threshold(
    rows: list[dict[str, str]],
    threshold: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """(âm tính giả/FN: live→spoof, dương tính giả/FP: spoof→live)."""
    fn_rows: list[dict[str, str]] = []
    fp_rows: list[dict[str, str]] = []
    for row in rows:
        score = float(row["live_score"])
        pred_live = score >= threshold
        if row["label_true"] == "live" and not pred_live:
            fn_rows.append(row)
        elif row["label_true"] == "spoof" and pred_live:
            fp_rows.append(row)
    return fn_rows, fp_rows


def _montage_cases_with_images(cases: list[dict[str, str]], *, limit: int) -> list[dict[str, str]]:
    picked: list[dict[str, str]] = []
    for row in cases:
        if len(picked) >= limit:
            break
        rel = row.get("image_path", "").strip()
        if not rel:
            continue
        if _resolve_repo_image_path(rel).is_file():
            picked.append(row)
    return picked


def _plot_error_montage(
    path: Path,
    cases: list[dict[str, str]],
    *,
    title: str,
    subtitle_fn: callable[[dict[str, str]], str],
    reverse_score: bool = False,
) -> bool:
    """Lưới 6×4 (tối đa 24 ảnh). Trả về False nếu không vẽ (không có ảnh hợp lệ)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    ordered = sorted(cases, key=lambda r: float(r["live_score"]), reverse=reverse_score)
    selected = _montage_cases_with_images(ordered, limit=MONTAGE_MAX_IMAGES)
    if not selected:
        return False

    fig, axes = plt.subplots(MONTAGE_ROWS, MONTAGE_COLS, figsize=(14, 9))
    for ax in axes.ravel():
        ax.axis("off")

    for idx, row in enumerate(selected):
        ax = axes.ravel()[idx]
        img_path = _resolve_repo_image_path(row["image_path"])
        with Image.open(img_path) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(subtitle_fn(row), fontsize=6)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _plot_misclassification_montages_at_threshold(
    output_dir: Path,
    rows: list[dict[str, str]],
    *,
    threshold: float,
    dataset_slug: str,
    model_id: str | None,
) -> list[Path]:
    suffix = _threshold_suffix(threshold)
    fn_rows, fp_rows = _misclassified_rows_at_threshold(rows, threshold)
    base = f"{dataset_slug}"
    if model_id:
        base += f" ({model_id})"

    def _fn_subtitle(row: dict[str, str]) -> str:
        s = float(row["live_score"])
        return f"nhãn: live\npred: spoof\nscore={s:.3f}"

    def _fp_subtitle(row: dict[str, str]) -> str:
        s = float(row["live_score"])
        return f"nhãn: spoof\npred: live\nscore={s:.3f}"

    saved: list[Path] = []
    fn_path = output_dir / f"false_negative_montage_{suffix}.png"
    if _plot_error_montage(
        fn_path,
        fn_rows,
        title=(
            f"FN (âm tính giả): live thật bị reject @ {threshold:g} — {base}\n"
            "(ảnh là người thật; model gán spoof vì score < ngưỡng)"
        ),
        subtitle_fn=_fn_subtitle,
        reverse_score=False,
    ):
        saved.append(fn_path)

    fp_path = output_dir / f"false_positive_montage_{suffix}.png"
    if _plot_error_montage(
        fp_path,
        fp_rows,
        title=(
            f"FP (dương tính giả): spoof bị chấp nhận @ {threshold:g} — {base}\n"
            "(ảnh là spoof; model gán live vì score ≥ ngưỡng)"
        ),
        subtitle_fn=_fp_subtitle,
        reverse_score=True,
    ):
        saved.append(fp_path)
    return saved


def _plot_live_score_true_false_at_threshold(
    path: Path,
    rows: list[dict[str, str]],
    *,
    threshold: float = LIVE_ACCEPT_THRESHOLD,
    dataset_slug: str,
    model_id: str | None = None,
) -> None:
    """Boxplot live_score trên mẫu live: True (≥ threshold) vs False (< threshold)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = _live_scores_from_rows(rows)
    if not scores:
        return

    true_scores = [s for s in scores if s >= threshold]
    false_scores = [s for s in scores if s < threshold]
    if not true_scores and not false_scores:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    data: list[list[float]] = []
    labels: list[str] = []
    colors: list[str] = []
    if true_scores:
        data.append(true_scores)
        labels.append(f"True (n={len(true_scores)})")
        colors.append("#2E86AB")
    if false_scores:
        data.append(false_scores)
        labels.append(f"False (n={len(false_scores)})")
        colors.append("#C73E1D")

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.45)
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.axhline(threshold, color="#555555", linestyle="--", linewidth=1.0, label=f"threshold={threshold:g}")
    ax.set_ylabel("live_score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    title = f"live @ {threshold:g} — True / False — {dataset_slug}"
    if model_id:
        title += f" ({model_id})"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_live_score_histogram_coarse(
    path: Path,
    rows: list[dict[str, str]],
    *,
    threshold: float = LIVE_ACCEPT_THRESHOLD,
    dataset_slug: str,
    model_id: str | None = None,
) -> None:
    """Histogram live_score (chỉ live), 5 bin thô + đường ngưỡng."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = _live_scores_from_rows(rows)
    if not scores:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(
        scores,
        bins=LIVE_SCORE_COARSE_BIN_EDGES,
        color="#2E86AB",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.85,
    )
    ax.axvline(threshold, color="#C73E1D", linestyle="--", linewidth=1.2, label=f"threshold={threshold:g}")
    ax.set_xlabel("live_score")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    title = f"live histogram (step {LIVE_SCORE_HISTOGRAM_BIN_STEP:g}) — {dataset_slug}"
    if model_id:
        title += f" ({model_id})"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_live_score_distribution(
    path: Path,
    rows: list[dict[str, str]],
    *,
    dataset_slug: str,
    model_id: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live_scores, spoof_scores = _scores_by_true_label(rows)
    if not live_scores and not spoof_scores:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = [i / 100 for i in range(101)]  # 0.00 .. 1.00 step 0.01

    if live_scores:
        ax.hist(
            live_scores,
            bins=bins,
            alpha=0.55,
            label=f"true live (n={len(live_scores)})",
            color="#2E86AB",
            edgecolor="white",
            linewidth=0.4,
        )
    if spoof_scores:
        ax.hist(
            spoof_scores,
            bins=bins,
            alpha=0.55,
            label=f"true spoof (n={len(spoof_scores)})",
            color="#A23B72",
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xlabel("live_score")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    title = f"live_score distribution — {dataset_slug}"
    if model_id:
        title += f" ({model_id})"
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_apcer_bpcer_curve(
    path: Path,
    results: list[dict[str, float | int]],
    *,
    dataset_slug: str,
    model_id: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(results, key=lambda r: float(r["threshold"]))
    thresholds = [float(r["threshold"]) for r in ordered]
    apcer = [float(r["apcer"]) for r in ordered]
    bpcer = [float(r["bpcer"]) for r in ordered]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, apcer, marker="o", label="APCER")
    ax.plot(thresholds, bpcer, marker="s", label="BPCER")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    title = f"APCER / BPCER vs threshold — {dataset_slug}"
    if model_id:
        title += f" ({model_id})"
    ax.set_title(title)

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
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Namespace metrics theo model (override config / metadata predictions).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Đánh giá mọi cặp model_*.yaml × dataset_*.yaml (predictions/latest.csv đã có).",
    )
    return parser.parse_args()


def evaluate_predictions(
    *,
    predictions_path: Path,
    dataset_override: str | None,
    model_id: str | None,
    cfg: dict[str, Any],
    thresholds: list[float],
    flat_output: bool,
    repo_root: Path,
) -> None:
    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy predictions file: {predictions_path}. "
            "Chạy inference trước hoặc dùng --predictions.",
        )
    if not thresholds:
        raise ValueError("Cấu hình thresholds rỗng.")

    resolved_model_id = model_id
    if resolved_model_id is None:
        meta = parse_predictions_metadata(predictions_path)
        if meta.get("model_id"):
            resolved_model_id = sanitize_model_id(meta["model_id"])

    dataset_slug = resolve_dataset_slug(predictions_path, dataset_override)

    if resolved_model_id:
        report_paths = ModelReportPaths(resolved_model_id, repo_root)
        output_dir = (
            report_paths.model_root / "metrics"
            if flat_output
            else report_paths.metrics_dir(dataset_slug)
        )
    else:
        metrics_base = _resolve_path(str(cfg.get("metrics_output_dir", "reports/metrics")))
        output_dir = metrics_base if flat_output else metrics_base / dataset_slug

    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        "model_id": resolved_model_id or "",
        "source_dataset": dataset_slug,
        "predictions_path": _repo_relative_path(predictions_path),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
    }

    stats = EvalStats()
    rows = _load_prediction_rows(predictions_path, stats)
    if not rows:
        raise ValueError("Không còn dữ liệu hợp lệ sau khi lọc.")

    results = [_evaluate_threshold(rows, t) for t in thresholds]
    curve_results = [_evaluate_threshold(rows, t) for t in _thresholds_for_curve_plot(thresholds)]

    summary_path = output_dir / "metrics_summary.csv"
    _write_summary_csv(summary_path, stats, results)

    curve_path = output_dir / "apcer_bpcer_vs_threshold.png"
    _plot_apcer_bpcer_curve(
        curve_path,
        curve_results,
        dataset_slug=dataset_slug,
        model_id=resolved_model_id,
    )

    dist_path = output_dir / "live_score_distribution.png"
    _plot_live_score_distribution(
        dist_path,
        rows,
        dataset_slug=dataset_slug,
        model_id=resolved_model_id,
    )

    if _is_drivers_dataset_slug(dataset_slug):
        _plot_live_score_true_false_at_threshold(
            output_dir / "live_score_true_false_0.5.png",
            rows,
            threshold=LIVE_ACCEPT_THRESHOLD,
            dataset_slug=dataset_slug,
            model_id=resolved_model_id,
        )
        _plot_live_score_histogram_coarse(
            output_dir / "live_score_histogram_bins_0.5.png",
            rows,
            threshold=LIVE_ACCEPT_THRESHOLD,
            dataset_slug=dataset_slug,
            model_id=resolved_model_id,
        )

    montage_paths: list[Path] = []
    if LIVE_ACCEPT_THRESHOLD in thresholds:
        montage_paths = _plot_misclassification_montages_at_threshold(
            output_dir,
            rows,
            threshold=LIVE_ACCEPT_THRESHOLD,
            dataset_slug=dataset_slug,
            model_id=resolved_model_id,
        )

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

    print(f"model_id: {resolved_model_id or '(legacy)'}")
    print(f"dataset: {dataset_slug}")
    print(f"output_dir: {output_dir}")
    print(f"predictions: {predictions_path}")
    print(f"kept_rows: {stats.kept_rows}")
    print(f"dropped_error_rows: {stats.dropped_error_rows}")
    print(f"dropped_invalid_label_rows: {stats.dropped_invalid_label_rows}")
    print(f"saved: {summary_path}")
    print(f"saved: {curve_path}")
    print(f"saved: {dist_path}")
    for path in montage_paths:
        print(f"saved: {path}")
    for path in saved_json_paths:
        print(f"saved: {path}")
    for path in saved_cm_paths:
        print(f"saved: {path}")


def main() -> int:
    args = parse_args()
    cfg = _load_yaml(args.config)

    thresholds_raw = cfg.get("thresholds", [])
    thresholds = [float(t) for t in thresholds_raw]
    flat_output = args.flat_output or bool(cfg.get("flat_output"))

    if args.all:
        if args.predictions:
            raise ValueError("--all không dùng cùng --predictions.")
        model_configs = discover_config_paths(REPO_ROOT, "model")
        dataset_configs = discover_config_paths(REPO_ROOT, "dataset")
        if not model_configs:
            raise ValueError(f"Không tìm thấy configs/model_*.yaml trong {REPO_ROOT / 'configs'}")
        if not dataset_configs:
            raise ValueError(f"Không tìm thấy configs/dataset_*.yaml trong {REPO_ROOT / 'configs'}")

        for model_config in model_configs:
            model_id = resolve_model_id(_load_yaml(model_config), REPO_ROOT)
            for dataset_config in dataset_configs:
                dataset_slug = source_dataset_from_config(dataset_config)
                predictions_path = ModelReportPaths(model_id, REPO_ROOT).predictions_latest(dataset_slug)
                if not predictions_path.is_file():
                    print(
                        f"\n=== skip {model_config.name} × {dataset_config.name} "
                        f"(thiếu {predictions_path}) ===",
                    )
                    continue
                print(f"\n=== {model_config.name} × {dataset_config.name} ===")
                evaluate_predictions(
                    predictions_path=predictions_path,
                    dataset_override=dataset_slug,
                    model_id=model_id,
                    cfg=cfg,
                    thresholds=thresholds,
                    flat_output=flat_output,
                    repo_root=REPO_ROOT,
                )
        return 0

    dataset_override = args.dataset or cfg.get("dataset")

    predictions_path: Path | None = None
    if args.predictions:
        predictions_path = args.predictions if args.predictions.is_absolute() else _resolve_path(str(args.predictions))
    elif cfg.get("predictions_path"):
        predictions_path = _resolve_path(str(cfg["predictions_path"]))

    model_id = resolve_model_id_for_eval(
        predictions_path=predictions_path,
        eval_cfg=cfg,
        cli_model_id=args.model_id,
    )

    if predictions_path is None:
        if not model_id or not dataset_override:
            raise ValueError(
                "Cần --predictions, hoặc model_id + dataset trong config/CLI "
                "(vd. model_config + dataset: celeba_spoof).",
            )
        predictions_path = ModelReportPaths(model_id, REPO_ROOT).predictions_latest(str(dataset_override))

    if not predictions_path.is_absolute():
        predictions_path = _resolve_path(str(predictions_path))

    evaluate_predictions(
        predictions_path=predictions_path,
        dataset_override=str(dataset_override) if dataset_override else None,
        model_id=model_id,
        cfg=cfg,
        thresholds=thresholds,
        flat_output=flat_output,
        repo_root=REPO_ROOT,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
