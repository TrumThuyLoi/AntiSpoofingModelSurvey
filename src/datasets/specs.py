from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class HFRawDatasetSpec:
    source_dataset: str          # "celeba-spoof" | "casia-fasd"
    repo_id: str
    dataset_page: str
    default_raw_rel_dir: str     # "data/raw/celeba-spoof"
    hf_split: str = "test"
    image_column: str = "cropped_image"
    label_field: str = "labels"
    label_name_field: str = "labelNames"
    label_aliases: dict[str, str] | None = None  # chung live/spoof
    token_env_var: str | None = None

@dataclass(frozen=True)
class DownloadResult:
    """Kết quả sau khi tải raw dataset."""
    output_dir: Path
    annotation_path: Path
    num_images: int
    label_counts: dict[str, int]
    skipped: bool

CELEBA_SPOOF_SPEC = HFRawDatasetSpec(
    source_dataset="celeba-spoof",
    repo_id="nguyenkhoa/celeba-spoof-for-face-antispoofing-test",
    hf_split="test",
    default_raw_rel_dir="data/raw/celeba-spoof",
    label_field="labels",
    label_name_field="labelNames",
    label_aliases={
        "live": "live",
        "0live": "live",
        "spoof": "spoof",
        "1spoof": "spoof",
    },
)

CASIA_FASD_SPEC = HFRawDatasetSpec(
    source_dataset="casia-fasd",
    repo_id="vu-hong-quang/casia_fasd",
    hf_split="test",
    default_raw_rel_dir="data/raw/casia-fasd",
    label_field="labels",
    label_name_field="labelNames",
    label_aliases={
        "live": "live",
        "0live": "live",
        "spoof": "spoof",
        "1spoof": "spoof",
    },
    token_env_var="HUGGINGFACE_TOKEN",
)