import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


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
        out_path = generate_tasks_for_dataset(ds, limit=args.limit)
        print(f"[INFO] Generated tasks for {ds}: {out_path}")


if __name__ == "__main__":
    main()

