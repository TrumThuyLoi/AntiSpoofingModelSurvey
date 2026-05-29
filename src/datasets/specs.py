from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class HFRawDatasetSpec:
    source_dataset: str          # "celeba_spoof" | "casia_fasd"
    repo_id: str
    dataset_page: str
    default_raw_rel_dir: str     # "data/raw/celeba_spoof"
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
    source_dataset="celeba_spoof",
    repo_id="nguyenkhoa/celeba-spoof-for-face-antispoofing-test",
    dataset_page="https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test",
    hf_split="test",
    default_raw_rel_dir="data/raw/celeba_spoof",
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
    source_dataset="casia_fasd",
    repo_id="vu-hong-quang/casia_fasd",
    dataset_page="https://huggingface.co/datasets/vu-hong-quang/casia_fasd",
    hf_split="test",
    default_raw_rel_dir="data/raw/casia_fasd",
    label_field="labels",
    label_name_field="labelNames",
    label_aliases={
        "live": "live",
        "0live": "live",
        "spoof": "spoof",
        "1spoof": "spoof",
    },
    token_env_var="HUGGINGFACE_PRIVATE_DATASET_TOKEN",
)

FACE_ANTI_SPOOFING_VN_SPEC = HFRawDatasetSpec(
    source_dataset="face_antispoofing_vn",
    repo_id="vu-hong-quang/face_antispoofing_vn",
    dataset_page="https://huggingface.co/datasets/vu-hong-quang/face_antispoofing_vn",
    hf_split="test",
    default_raw_rel_dir="data/raw/face_antispoofing_vn",
    label_field="labels",
    label_name_field="labelNames",
    label_aliases={
        "live": "live",
        "0live": "live",
        "spoof": "spoof",
        "1spoof": "spoof",
        "not_live": "spoof",
        "0": "live",
        "1": "spoof",
    },
    token_env_var="HUGGINGFACE_PRIVATE_DATASET_TOKEN",
)

DATASET_SPECS: dict[str, HFRawDatasetSpec] = {
    "celeba_spoof": CELEBA_SPOOF_SPEC,
    "casia_fasd": CASIA_FASD_SPEC,
    "face_antispoofing_vn": FACE_ANTI_SPOOFING_VN_SPEC,
}