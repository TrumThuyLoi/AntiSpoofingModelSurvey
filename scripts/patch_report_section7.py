#!/usr/bin/env python3
"""Regenerate REPORT.md §7 (đến trước §7.5.1): metric + biểu đồ.

- Phạm vi báo cáo: **face_antispoofing_vn_exp1.5** + **drivers_250_fn_exp1.5** (SFAS bbox **1.5**).
- Confusion matrix: chỉ **ảnh** `confusion_matrix_0.5.png` (không bảng markdown TP/FN).
- §7.5.2 (drivers): `live_score_true_false_0.5.png`, `live_score_histogram_bins_0.5.png`.

Cần chạy `python3 scripts/run_evaluation.py --all` trước để có PNG trong
`reports/models/<model_id>/metrics/<dataset>/`.
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "REPORT.md"
MODELS = [
    ("minifasnet_v2_2p7", "MiniFASNet"),
    ("face_antispoof_onnx_9820", "ONNX"),
    ("hairymax_bin_1p5_pretrained", "Hairymax pre"),
    ("hairymax_bin_1p5_retrain", "Hairymax retrain"),
]
THRESHOLDS = (0.3, 0.5, 0.7)
ROOT = REPO / "reports" / "models"


def metrics_glossary_block() -> str:
    """Chú thích chỉ số — chèn đầu §7 (giữ khi chạy lại patch)."""
    return "\n".join(
        [
            "### 7.0 Chú thích chỉ số đo lường",
            "",
            "Quy ước trong repo (`scripts/run_evaluation.py`): ảnh có nhãn thật **live** hoặc **spoof**; "
            "model xuất **`live_score`** ∈ [0, 1] (càng cao càng nghiêng về live). "
            "Với ngưỡng **Threshold** *t*: dự đoán **live** nếu `live_score ≥ t`, ngược lại **spoof**.",
            "",
            "| Chỉ số / thuật ngữ | Ý nghĩa | Công thức (theo confusion matrix) | Ghi chú vận hành |",
            "|-------------------|---------|-----------------------------------|------------------|",
            "| **Threshold** | Ngưỡng cắt `live_score` | — | Báo cáo dùng **0.3, 0.5, 0.7**; đường cong APCER/BPCER vẽ dày hơn (0.00–1.00). |",
            "| **live_score** | Điểm “độ live” do model | — | Biểu đồ phân phối: tách theo nhãn thật live vs spoof. |",
            "| **Accuracy** | Tỷ lệ dự đoán đúng (cả hai lớp) | (TP + TN) / tổng mẫu | Dễ lệch nếu mất cân bằng lớp; không thay BPCER/APCER trên production. |",
            "| **APCER** | Attack Presentation Classification Error Rate | FP / (FP + TN) = spoof bị gán **live** | **Spoof lọt** (false accept). Càng **thấp** càng chặt spoof. |",
            "| **BPCER** | Bona Fide Presentation Classification Error Rate | FN / (TP + FN) = live bị gán **spoof** | **Live bị reject** (false reject). Metric **ưu tiên** trên `drivers_250_fn` (toàn live). |",
            "| **TP** | True Positive (live→live) | Live đúng, pass | Trên drivers: số live được chấp nhận. |",
            "| **FN** | False Negative (live→spoof) | Live sai, reject | = BPCER × số live; ví dụ BPCER 0.059, 393 live → ~23 FN. |",
            "| **FP** | False Positive (spoof→live) | Spoof sai, pass | = APCER × số spoof; trên drivers (0 spoof) → APCER = 0. |",
            "| **TN** | True Negative (spoof→spoof) | Spoof đúng, reject | |",
            "",
            "**Đọc biểu đồ:**",
            "- `apcer_bpcer_vs_threshold.png`: đổi **Threshold** → trade-off APCER ↔ BPCER.",
            "- `live_score_distribution.png`: live thật nên dồn **cao**; spoof thật nên dồn **thấp** (tách càng rõ càng tốt).",
            "- `confusion_matrix_0.5.png`: ma trận 2×2 @ ngưỡng **0.5** (trục: true live/spoof × pred live/spoof).",
            "",
            "**Định dạng bảng §7.1–7.2:** mỗi ô Accuracy / APCER / BPCER = **`tỷ lệ (số_lỗi/tổng_lớp)`** "
            "(vd. BPCER `0.059 (23/393)` = 23 live bị reject trên 393 live).",
            "",
            "**Production (verify tài xế):** gate đề xuất **BPCER ≤ 0.10** trên tập live thực tế; kiểm tra thêm APCER khi có tập spoof.",
            "",
        ]
    )


def metrics_from_predictions(path: Path) -> dict[float, dict] | None:
    if not path.is_file():
        return None
    rows: list[tuple[str, float]] = []
    with path.open(encoding="utf-8") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        if (row.get("error") or "").strip():
            continue
        rows.append((row["label_true"].strip().lower(), float(row["live_score"])))
    if not rows:
        return None
    n_live = sum(1 for label, _ in rows if label == "live")
    n_spoof = len(rows) - n_live
    out: dict[float, dict] = {}
    for th in THRESHOLDS:
        tp = fn = fp = tn = 0
        for label, live_score in rows:
            pred = "live" if live_score >= th else "spoof"
            if label == "live":
                if pred == "live":
                    tp += 1
                else:
                    fn += 1
            else:
                if pred == "live":
                    fp += 1
                else:
                    tn += 1
        bpcer = fn / n_live if n_live else 0.0
        apcer = fp / n_spoof if n_spoof else 0.0
        acc = (tp + tn) / len(rows)
        out[th] = {
            "accuracy": acc,
            "apcer": apcer,
            "bpcer": bpcer,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }
    return out


def metrics_from_summary(mid: str, ds: str) -> dict[float, dict] | None:
    path = ROOT / mid / "metrics" / ds / "metrics_summary.csv"
    if not path.is_file():
        return None
    out: dict[float, dict] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            th = float(row["threshold"])
            if th not in THRESHOLDS:
                continue
            out[th] = {
                "accuracy": float(row["accuracy"]),
                "apcer": float(row["apcer"]),
                "bpcer": float(row["bpcer"]),
                "tp": int(row["cm_true_live_pred_live"]),
                "fn": int(row["cm_true_live_pred_spoof"]),
                "fp": int(row["cm_true_spoof_pred_live"]),
                "tn": int(row["cm_true_spoof_pred_spoof"]),
            }
    return out or None


def load_metrics(ds: str, *, from_predictions: bool) -> dict[str, dict[float, dict]]:
    """Metrics theo model; thiếu artifact → bỏ qua model đó (không fail cả dataset)."""
    merged: dict[str, dict[float, dict]] = {}
    for mid, _ in MODELS:
        if from_predictions:
            data = metrics_from_predictions(ROOT / mid / "predictions" / ds / "latest.csv")
        else:
            data = metrics_from_summary(mid, ds)
        if data:
            merged[mid] = data
    return merged


def model_table_header() -> tuple[str, str]:
    """Header markdown 2 hàng cho bảng/biểu đồ theo MODELS."""
    labels = [label for _, label in MODELS]
    cols = " | ".join(labels)
    sep = " | ".join(":---:" for _ in labels)
    return f"| {cols} |", f"|{sep}|"


def benchmark_matrix_header() -> tuple[str, str]:
    """Header bảng Dataset × models (§7.5, §7.5.1)."""
    labels = [label for _, label in MODELS]
    return (
        "| Dataset | " + " | ".join(labels) + " |",
        "|---------|" + "|".join(":---:" for _ in labels) + "|",
    )


def fmt4(value: float) -> str:
    return f"{float(value):.4f}"


def fmt_rate_count(rate: float, num: int, denom: int) -> str:
    if denom <= 0:
        return f"{fmt4(rate)} (—)"
    return f"{fmt4(rate)} ({num}/{denom})"


def metric_table(ds: str, *, from_predictions: bool) -> str:
    by_model = load_metrics(ds, from_predictions=from_predictions)
    if not by_model:
        return "*Chưa có metrics cho dataset này.*\n"
    lines = [
        "| Model | Threshold | Accuracy | APCER | BPCER |",
        "|-------|-----------|----------|-------|-------|",
    ]
    for mid, label in MODELS:
        data = by_model.get(mid)
        if not data:
            continue
        for th in THRESHOLDS:
            row = data[th]
            tp = int(row["tp"])
            fn = int(row["fn"])
            fp = int(row["fp"])
            tn = int(row["tn"])
            n_live = tp + fn
            n_spoof = fp + tn
            total = n_live + n_spoof
            lines.append(
                f"| {label} | {th:g} | "
                f"{fmt_rate_count(float(row['accuracy']), tp + tn, total)} | "
                f"{fmt_rate_count(float(row['apcer']), fp, n_spoof)} | "
                f"{fmt_rate_count(float(row['bpcer']), fn, n_live)} |"
            )
    return "\n".join(lines)


def _metrics_rel_path(model_id: str, ds: str, filename: str) -> str:
    return f"./reports/models/{model_id}/metrics/{ds}/{filename}"


def plot_row(ds: str, filename: str, *, alt: str) -> str:
    h1, h2 = model_table_header()
    header = f"{h1}\n{h2}\n|"
    cells: list[str] = []
    for mid, label in MODELS:
        rel = _metrics_rel_path(mid, ds, filename)
        path = ROOT / mid / "metrics" / ds / filename
        if path.is_file():
            cells.append(f" ![{label} {alt}]({rel}) ")
        else:
            cells.append(f" *({label}: chưa có `{filename}`)* ")
    return header + "|".join(cells) + "|"


def charts_block(ds: str) -> list[str]:
    """Biểu đồ từ `run_evaluation.py` (metrics/<dataset>/)."""
    return [
        "Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):",
        "",
        plot_row(ds, "apcer_bpcer_vs_threshold.png", alt="APCER/BPCER"),
        "",
        "Phân phối `live_score` (`live_score_distribution.png`):",
        "",
        plot_row(ds, "live_score_distribution.png", alt="live_score"),
        "",
        "Confusion matrix @0.5 (`confusion_matrix_0.5.png`):",
        "",
        plot_row(ds, "confusion_matrix_0.5.png", alt="CM@0.5"),
        "",
    ]


def error_montages_block(ds: str, *, include_fp: bool = True) -> list[str]:
    """FN/FP montage @0.5 (`run_evaluation.py`). Drivers (0 spoof): chỉ FN."""
    lines = [
        "**FN (âm tính giả)** — nhãn **live**, model **reject** (`live_score` < 0.5). "
        "Ảnh là **người thật** (không phải spoof). `false_negative_montage_0.5.png`:",
        "",
        plot_row(ds, "false_negative_montage_0.5.png", alt="FN"),
        "",
    ]
    if include_fp:
        lines += [
            "**FP (dương tính giả)** — nhãn **spoof**, model **chấp nhận** (`live_score` ≥ 0.5). "
            "Ảnh là **spoof** (thư mục `not_live` / spoof trên Face VN). `false_positive_montage_0.5.png`:",
            "",
            plot_row(ds, "false_positive_montage_0.5.png", alt="FP"),
            "",
        ]
    return lines


def dataset_block(title: str, ds: str, intro: list[str], *, drivers: bool = False) -> str:
    parts = (
        [f"### {title}", ""]
        + intro
        + ["", metric_table(ds, from_predictions=drivers), ""]
        + charts_block(ds)
    )
    return "\n".join(parts)


def expansion_block(
    heading: str,
    ds: str,
    *,
    drivers: bool,
    error_montages: bool = False,
    fp_montage: bool = True,
) -> str:
    parts = [f"#### {heading}", "", metric_table(ds, from_predictions=drivers), ""] + charts_block(ds)
    if error_montages:
        parts += error_montages_block(ds, include_fp=fp_montage)
    return "\n".join(parts)


def build_section7() -> str:
    parts = [
        "## 7. Kết quả",
        "",
        "Phạm vi: **Face Anti-Spoofing VN** + **drivers_250_fn** (ảnh live thật), crop SFAS **exp 1.5**.",
        "Ngưỡng: **0.3, 0.5, 0.7** (`configs/evaluation.yaml`). Metric từ `predictions/latest.csv` "
        "(Face VN: live+spoof; drivers: 393 live, APCER = 0).",
        "",
        metrics_glossary_block(),
        "### 7.1 Face Anti-Spoofing VN (SFAS exp 1.5)",
        "",
        "- **11376** mẫu (7541 live, 3835 spoof). `source_dataset`: `face_antispoofing_vn_exp1.5`.",
        "",
    ]
    parts.append(
        expansion_block(
            "SFAS exp 1.5 (`face_antispoofing_vn_exp1.5`)",
            "face_antispoofing_vn_exp1.5",
            drivers=True,
            error_montages=True,
        )
    )
    parts.extend(
        [
            "### 7.2 Drivers 250 FN (SFAS exp 1.5)",
            "",
            "- **393** live. **APCER = 0**. `source_dataset`: `drivers_250_fn_exp1.5`.",
            "- Hairymax ONNX (pretrained / retrain) benchmark trên tập này.",
            "",
        ]
    )
    parts.append(
        expansion_block(
            "SFAS exp 1.5 (`drivers_250_fn_exp1.5`)",
            "drivers_250_fn_exp1.5",
            drivers=True,
            error_montages=True,
            fp_montage=False,
        )
    )
    return "\n".join(parts)


BENCHMARK_ROWS: list[tuple[str, str, bool]] = [
    ("face_antispoofing_vn_exp1.5", "face_antispoofing_vn_exp1.5", True),
    ("drivers_250_fn_exp1.5", "drivers_250_fn_exp1.5", True),
]

SECTION751_ROWS = BENCHMARK_ROWS


def _bpcer_apcer_cells(ds: str, *, from_predictions: bool) -> list[str]:
    by_model = load_metrics(ds, from_predictions=from_predictions)
    cells: list[str] = []
    for mid, _ in MODELS:
        if mid not in by_model:
            cells.append("— / —")
            continue
        row = by_model[mid][0.5]
        tp, fn, fp, tn = int(row["tp"]), int(row["fn"]), int(row["fp"]), int(row["tn"])
        n_live, n_spoof = tp + fn, fp + tn
        bpcer = fmt_rate_count(float(row["bpcer"]), fn, n_live)
        apcer = fmt_rate_count(float(row["apcer"]), fp, n_spoof)
        cells.append(f"{bpcer} / {apcer}")
    return cells


def _tp_fn_live_at_05(ds: str, mid: str) -> tuple[int, int, int, int]:
    """(tp_live, fn_live, n_live, n_total) @ threshold 0.5 từ predictions."""
    path = ROOT / mid / "predictions" / ds / "latest.csv"
    tp = fn = n_live = n_total = 0
    with path.open(encoding="utf-8") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    for row in csv.DictReader(lines):
        if (row.get("error") or "").strip():
            continue
        n_total += 1
        label = (row["label_true"] or "").strip().lower()
        if label != "live":
            continue
        n_live += 1
        live_score = float(row["live_score"])
        if live_score >= 0.5:
            tp += 1
        else:
            fn += 1
    return tp, fn, n_live, n_total


def build_section751() -> str:
    lines = [
        "#### 7.5.1 Dataset × dự đoán @0.5 (`True` / `False`)",
        "",
        "Nguồn: `reports/models/<model_id>/predictions/<dataset>/latest.csv`, ngưỡng **0.5**.",
        "",
        "- **True** = số ảnh **live** được model chấp nhận (pred live → TP trên live).",
        "- **False** = số ảnh **live** bị reject (pred spoof → FN trên live).",
        "- Định dạng ô: **`True / False`** (chỉ đếm mẫu nhãn live; spoof không hiển thị ở đây).",
        "",
        "**Phạm vi dataset** (khớp §7.1–7.2):",
        "- **Face VN** + **drivers_250_fn**, crop SFAS **exp 1.5**.",
        "",
    ]
    lines += list(benchmark_matrix_header())
    for slug, label, drivers in SECTION751_ROWS:
        cells: list[str] = []
        note = ""
        for mid, _ in MODELS:
            try:
                tp, fn, n_live, n_total = _tp_fn_live_at_05(slug, mid)
                cells.append(f"{tp} / {fn}")
                if not note and n_live:
                    note = f"{n_live} live" if n_live == n_total else f"{n_live} live, {n_total} mẫu"
            except (FileNotFoundError, OSError, ValueError, KeyError):
                cells.append("— / —")
        row_label = f"{label} ({note})" if note else label
        lines.append(f"| {row_label} | {' | '.join(cells)} |")
    lines.append("")
    return "\n".join(lines)


def build_section75_matrix() -> str:
    lines = [
        "### 7.5 Ma trận benchmark — tóm tắt @0.5",
        "",
        "Inference + evaluation trên hai dataset §7.1–7.2. Bảng đầy đủ threshold: §7.1–7.2.",
        "",
        "Định dạng mỗi ô: **BPCER (FN/n_live) / APCER (FP/n_spoof)**.",
        "",
    ]
    lines += list(benchmark_matrix_header())
    for slug, label, drivers in BENCHMARK_ROWS:
        cells = _bpcer_apcer_cells(slug, from_predictions=drivers)
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines.append("")
    return "\n".join(lines)


DRIVERS_EXP_FOR_752 = (("drivers_250_fn_exp1.5", "1.5"),)


def build_section752() -> str:
    lines = [
        "#### 7.5.2 Phân phối `y_prob` @0.5 (nhóm True / False)",
        "",
        "`live_score` = **y_prob**; ngưỡng **0.5**. **Drivers** SFAS **exp 1.5** (393 live).",
        "",
        "- **True** = live được chấp nhận (TP); **False** = live bị reject (FN).",
        "- Artifact: `live_score_true_false_0.5.png`, `live_score_histogram_bins_0.5.png` "
        "(`run_evaluation.py`, chỉ `drivers_250_fn*`).",
        "",
    ]
    for slug, exp in DRIVERS_EXP_FOR_752:
        lines += [
            f"**Drivers exp {exp}** (`{slug}`, 393 live):",
            "",
            "Phân phối theo nhóm True / False @0.5 (`live_score_true_false_0.5.png`):",
            "",
            plot_row(slug, "live_score_true_false_0.5.png", alt="True/False"),
            "",
            "Histogram toàn tập live (step **0.1**, `live_score_histogram_bins_0.5.png`):",
            "",
            plot_row(slug, "live_score_histogram_bins_0.5.png", alt="histogram"),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    if not REPORT.is_file():
        raise SystemExit(f"Missing {REPORT}")
    text = REPORT.read_text(encoding="utf-8")
    start = text.index("## 7. Kết quả")
    i10 = text.index("\n## 10.")
    REPORT.write_text(
        text[:start]
        + build_section7()
        + "\n"
        + build_section75_matrix()
        + "\n"
        + build_section751()
        + "\n"
        + build_section752()
        + "\n---\n\n"
        + text[i10:].lstrip("\n"),
        encoding="utf-8",
    )
    print(f"Updated {REPORT}")
    print("§7: face_antispoofing_vn_exp1.5 + drivers_250_fn_exp1.5")


if __name__ == "__main__":
    main()
