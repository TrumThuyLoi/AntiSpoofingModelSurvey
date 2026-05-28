import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

# Label Studio strips only the leading "$" from value="$source_dataset | $split"
# and then requires this literal key in task.data (see label_studio/core/label_config.py).
SOURCE_DATASET_SPLIT_KEY = "source_dataset | $split"


def _dataset_spec(dataset: str) -> Tuple[Path, Path]:
    """
    Return (raw_csv_path, output_tasks_path) for a supported dataset name.
    """
    if dataset not in {"celeba_spoof", "casia_fasd"}:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'celeba_spoof' or 'casia_fasd'.")

    raw_csv = REPO_ROOT / "data" / "raw" / dataset / "annotations" / "raw.csv"
    out_json = REPO_ROOT / "label-studio" / "import" / f"{dataset}_tasks.json"
    return raw_csv, out_json


def _image_url_from_image_path(image_path: str) -> str:
    """
    Convert a CSV image_path (e.g. 'data/raw/celeba_spoof/images/test/000001.jpg')
    to a Label Studio local-files URL (e.g. '/data/local-files/?d=raw/celeba_spoof/images/test/000001.jpg').

    This assumes LOCAL_FILES_DOCUMENT_ROOT points to '<repo_root>/data'.
    """
    # Normalise separators and strip leading './'
    image_path = image_path.lstrip("./")

    # We expect paths to start with 'data/'. Everything after that is relative to DOCUMENT_ROOT=/.../data
    prefix = "data/"
    if image_path.startswith(prefix):
        relative = image_path[len(prefix) :]
    else:
        # Fallback: treat the whole path as relative (still works if DOCUMENT_ROOT matches parent)
        relative = image_path

    return f"/data/local-files/?d={relative}"


def load_raw_annotations(raw_csv_path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    """
    Load rows from a raw.csv file as dictionaries.
    """
    if not raw_csv_path.is_file():
        raise FileNotFoundError(f"raw.csv not found at {raw_csv_path}")

    rows: List[Dict[str, str]] = []
    with raw_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_tasks_from_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert raw.csv rows into Label Studio tasks.

    Expected columns in each row (see AGENT.md / raw.csv):
      - image_path
      - label
      - source_dataset
      - split
      - is_valid
      - note
    """
    required_cols = {"image_path", "label", "source_dataset", "split"}
    if not rows:
        return []

    missing = required_cols - set(rows[0].keys())
    if missing:
        raise KeyError(f"raw.csv is missing required columns: {sorted(missing)}")

    tasks: List[Dict[str, Any]] = []
    for row in rows:
        image_path = row["image_path"]
        source_dataset = row["source_dataset"]
        split = row["split"]
        label_original = row["label"]

        task = {
            "data": {
                "image": _image_url_from_image_path(image_path),
                "image_path": image_path,
                "source_dataset": source_dataset,
                "split": split,
                SOURCE_DATASET_SPLIT_KEY: f"{source_dataset} | {split}",
                "label_original": label_original,
            },
            "meta": {
                "image_path": image_path,
                "source_dataset": source_dataset,
                "split": split,
                "label_original": label_original,
            },
        }
        tasks.append(task)

    return tasks


def write_tasks(tasks: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def generate_tasks_for_dataset(dataset: str, limit: int | None = None) -> Path:
    raw_csv_path, out_json_path = _dataset_spec(dataset)
    rows = load_raw_annotations(raw_csv_path, limit=limit)
    tasks = build_tasks_from_rows(rows)
    write_tasks(tasks, out_json_path)
    return out_json_path


def generate_unknown_tasks_for_dataset(dataset: str, limit: int | None = None) -> Path:
    """
    Generate Label Studio tasks only for rows in raw.csv where label == "unknown".

    Output: label-studio/import/{dataset}_unknown_tasks.json
    """
    raw_csv_path, _ = _dataset_spec(dataset)
    out_json_path = REPO_ROOT / "label-studio" / "import" / f"{dataset}_unknown_tasks.json"

    rows = load_raw_annotations(raw_csv_path)
    unknown_rows = [row for row in rows if row.get("label") == "unknown"]
    if limit is not None:
        unknown_rows = unknown_rows[:limit]

    tasks = build_tasks_from_rows(unknown_rows)
    write_tasks(tasks, out_json_path)
    return out_json_path


def generate_sampled_tasks_for_dataset(dataset: str, sample_ratio: float = 0.1, seed: int = 42) -> Path:
    """
    Generate Label Studio tasks from data/sampled/<dataset>_sample.csv with random sampling.

    - sample_ratio mặc định 10% (0.1)
    - output mỗi dataset: label-studio/import/<dataset>_task.json
    """
    if dataset not in {"celeba_spoof", "casia_fasd"}:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected 'celeba_spoof' or 'casia_fasd'.")
    if not (0 < sample_ratio <= 1):
        raise ValueError("sample_ratio must be in range (0, 1].")

    sampled_csv_path = REPO_ROOT / "data" / "sampled" / f"{dataset}_sample.csv"
    out_json_path = REPO_ROOT / "label-studio" / "import" / f"{dataset}_10pct_task.json"

    rows = load_raw_annotations(sampled_csv_path)
    if not rows:
        write_tasks([], out_json_path)
        return out_json_path

    rng = random.Random(seed)
    sample_size = max(1, int(len(rows) * sample_ratio))
    sampled_rows = rng.sample(rows, k=min(sample_size, len(rows)))

    tasks = build_tasks_from_rows(sampled_rows)
    write_tasks(tasks, out_json_path)
    return out_json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Label Studio tasks from raw.csv and optionally validate them."
    )
    parser.add_argument(
        "--dataset",
        choices=["celeba_spoof", "casia_fasd"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate (and optionally check) tasks for all supported datasets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows/tasks per dataset (for sampling / quick runs).",
    )
    parser.add_argument(
        "--unknown-only",
        action="store_true",
        help='Only generate tasks for rows with label "unknown" (writes {dataset}_unknown_tasks.json).',
    )
    parser.add_argument(
        "--sampled-10pct",
        action="store_true",
        help=(
            "Generate tasks from data/sampled/<dataset>_sample.csv using 10% random sample "
            "(writes {dataset}_task.json)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets: List[str]
    if args.all:
        datasets = ["celeba_spoof", "casia_fasd"]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        raise SystemExit("Please specify --dataset {celeba_spoof,casia_fasd} or --all.")

    for ds in datasets:
        if args.unknown_only and args.sampled_10pct:
            raise SystemExit("Please use only one of --unknown-only or --sampled-10pct.")

        if args.sampled_10pct:
            out_path = generate_sampled_tasks_for_dataset(ds, sample_ratio=0.1, seed=42)
        elif args.unknown_only:
            out_path = generate_unknown_tasks_for_dataset(ds, limit=args.limit)
        else:
            out_path = generate_tasks_for_dataset(ds, limit=args.limit)
        print(f"[INFO] Generated tasks for {ds}: {out_path}")


if __name__ == "__main__":
    main()

