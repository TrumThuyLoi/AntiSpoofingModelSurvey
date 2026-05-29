# Đề bài:
Tìm hiểu các model anti spoofing trước. Theo pattern tìm các model open source SOTA hiện nay, tìm 1 tập dữ liệu test khoảng vài nghìn ảnh, label lại cho chính xác (có thể host label studio trên máy e là công cụ để label), rồi chạy test đánh giá kết quả. Rồi báo cáo lại.

---

## Data loader (Hugging Face → raw)

Tải ảnh và `annotations/raw.csv` từ Hugging Face vào `data/raw/<dataset>/` (module `src/datasets/hf_raw.py`).

**Chuẩn bị**

```bash
cd /path/to/DVX
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- **CelebA-Spoof** (`celeba_spoof`): repo public, không bắt buộc token.
- **CASIA-FASD** (`casia_fasd`) và **Face Anti-Spoofing VN** (`face_antispoofing_vn`): repo private — thêm vào `.env` ở thư mục gốc repo:

```env
HUGGINGFACE_PRIVATE_DATASET_TOKEN=hf_...
```

**Đưa dữ liệu lên Hugging Face (lần đầu)** — chỉ khi bạn là người publish repo private. Cần token **write** trên huggingface.co và repo dataset đã tạo sẵn (private). Script push đọc `HUGGINGFACE_TOKEN` (có thể đặt cùng giá trị với `HUGGINGFACE_PRIVATE_DATASET_TOKEN`).

**CASIA-FASD** → `vu-hong-quang/casia_fasd` (split `test`):

- Ảnh local: `data/raw/kaggle/casia_fasd/test_img/color/` và `train_img/color/` (tên file `*_real.*` = live, `*_fake.*` = spoof).
- Nếu split `test` trên HF đã có ≥ 4063 mẫu, script thoát và không push lại.

```bash
python3 scripts/create_casia_fasd_hf_dataset.py
```

**Face Anti-Spoofing VN** → `vu-hong-quang/face_antispoofing_vn` (split `train` + `test`):

- Ảnh gốc: `data/face_antispoofing_vn/train_photo|test_photo/{live,not_live}/`.
- Submodule detector (RetinaFace): `git submodule update --init third_party/Silent-Face-Anti-Spoofing` và đủ file trong `third_party/Silent-Face-Anti-Spoofing/resources/detection_model/`.
- Một lệnh: crop mặt → ghi `data/face_antispoofing_vn_cropped/` → push HF; in `Crop: images=... fail_detect=...`.

```bash
python3 scripts/create_vietnam_hf_dataset.py
```

Chi tiết schema / checklist: `hf_face_antispoofingvn.md`.

**Chạy tải dữ liệu** (từ thư mục gốc repo, sau khi HF đã có dữ liệu):

```bash
# CelebA-Spoof (~67k ảnh, có thể mất lâu)
python -m src.datasets.hf_raw --dataset celeba_spoof

# CASIA-FASD
python -m src.datasets.hf_raw --dataset casia_fasd

# Face Anti-Spoofing VN (private)
python -m src.datasets.hf_raw --dataset face_antispoofing_vn

# Tải lại dù đã có dữ liệu
python -m src.datasets.hf_raw --dataset casia_fasd --force
python -m src.datasets.hf_raw --dataset face_antispoofing_vn --force
```

Kết quả mẫu:

```text
data/raw/celeba_spoof/   hoặc   data/raw/casia_fasd/   hoặc   data/raw/face_antispoofing_vn/
  images/test/000000.jpg ...
  annotations/raw.csv
  meta/download_manifest.json
```

Gọi từ Python:

```python
from src.datasets.hf_raw import download_raw_dataset
from src.datasets.specs import CELEBA_SPOOF_SPEC, CASIA_FASD_SPEC, FACE_ANTI_SPOOFING_VN_SPEC

download_raw_dataset(CELEBA_SPOOF_SPEC)
download_raw_dataset(CASIA_FASD_SPEC)
download_raw_dataset(FACE_ANTI_SPOOFING_VN_SPEC)
```

## Label Studio

Trong `.env` (path WSL tuyệt đối):

```env
LOCAL_FILES_SERVING_ENABLED=true
LOCAL_FILES_DOCUMENT_ROOT=/mnt/d/ThucTap/DVX/data
```

Project → **Local files** → path `.../data/raw` (phải là thư mục con của `DOCUMENT_ROOT`, không trùng `data/`).

```bash
./label-studio/start.sh          # http://127.0.0.1:8080
```

Tạo tasks → import JSON trong `label-studio/import/`:

```bash
# Từ raw.csv (toàn bộ hoặc giới hạn)
python3 scripts/create_label_studio_task.py --dataset celeba_spoof   # → celeba_spoof_tasks.json
python3 scripts/create_label_studio_task.py --all --unknown-only   # → *_unknown_tasks.json
python3 scripts/create_label_studio_task.py --all --limit 100

# Verify 10% sample từ data/sampled/*_sample.csv (khuyến nghị)
python3 scripts/create_label_studio_task.py --dataset celeba_spoof --sampled-10pct
python3 scripts/create_label_studio_task.py --all --sampled-10pct
# → celeba_spoof_10pct_task.json, casia_fasd_10pct_task.json
```

**Labeling Interface** (Settings → Labeling Interface): copy từ `label-studio/labeling_config.xml`. Mỗi trường cần tag riêng (`$source_dataset`, `$split`, …). **Không** gộp trong một dòng kiểu `value="$source_dataset | $split"` — Label Studio coi đó là một key tên `source_dataset | $split` và import JSON sẽ báo *Validation error* dù `source_dataset` / `split` đã có trong task. Chọn nhãn mới qua `Choices name="label"` (live/spoof/unknown).

**Sau khi label:** Save trong UI → Export project → đặt file vào `data/labeled/` để merge vào CSV test.

## Pretrained weights

```bash
python3 scripts/download_pretrained_weights.py
```

Tải 3 file `.pth` vào `models/` (theo `configs/model_*.yaml`). File đã có thì bỏ qua.

## Sample cho inference (`data/sampled/`)

`configs/dataset_celeba_spoof.yaml`, `configs/dataset_casia_fasd.yaml` và `configs/dataset_face_antispoofing_vn.yaml` trỏ tới `data/sampled/*_sample.csv`. Tạo các file này **sau khi** đã có `data/raw/<dataset>/annotations/raw.csv` (bước HF ở trên):

```bash
python3 scripts/create_test_sample_annotation.py
```

Kết quả:

- `data/sampled/celeba_spoof_sample.csv` — 2000 live + 2000 spoof (random, seed 42)
- `data/sampled/casia_fasd_sample.csv` — toàn bộ dòng `is_valid=true` từ raw CASIA
- `data/sampled/face_antispoofing_vn_sample.csv` — sample từ raw VN (sau khi script sample hỗ trợ dataset này)

## Batch inference

Chạy từ thư mục gốc repo. Cần có weights (`download_pretrained_weights.py`), submodule (`git submodule update --init --recursive`), và `data/sampled/*_sample.csv`.

**Tham số CLI**

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--dataset-config` | `configs/dataset.yaml` | Dataset + `source_dataset` (đặt thư mục predictions) |
| `--model-config` | `configs/model_minifasnet.yaml` | Model + weight + `model_id` |
| `--inference-config` | `configs/inference.yaml` | `batch_size`, `save_raw_output`, … |
| `--annotation-csv` | *(từ dataset config)* | Ghi đè file CSV ảnh cần chạy; metadata vẫn theo `--dataset-config` |

**Dataset** (`--dataset-config`)

| File | `source_dataset` | Annotation mặc định |
|------|------------------|---------------------|
| `configs/dataset_celeba_spoof.yaml` | `celeba_spoof` | `data/sampled/celeba_spoof_sample.csv` |
| `configs/dataset_casia_fasd.yaml` | `casia_fasd` | `data/sampled/casia_fasd_sample.csv` |
| `configs/dataset_face_antispoofing_vn.yaml` | `face_antispoofing_vn` | `data/sampled/face_antispoofing_vn_sample.csv` |

**Model** (`--model-config`)

| File | `model_id` | Weight |
|------|------------|--------|
| `configs/model_minifasnet.yaml` | `minifasnet_v2_2p7` | `models/2.7_80x80_MiniFASNetV2.pth` |
| `configs/model_vit_fas.yaml` | `vitfas_vitb16_224` | `models/vitfas_vitb16_224x224.pth` |
| `configs/model_face_antispoof_onnx.yaml` | `face_antispoof_onnx_9820` | `models/face_antispoof_onnx_best_9820.pth` |

**Ví dụ** (ghép tùy ý dataset + model):

```bash
# MiniFASNet — CelebA (mặc định cả dataset lẫn model)
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_celeba_spoof.yaml

# MiniFASNet — CASIA
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_casia_fasd.yaml

# MiniFASNet — Face Anti-Spoofing VN
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_face_antispoofing_vn.yaml

# ViT-FAS — CelebA
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_celeba_spoof.yaml \
  --model-config configs/model_vit_fas.yaml

# ViT-FAS — CASIA
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_casia_fasd.yaml \
  --model-config configs/model_vit_fas.yaml

# Face antispoof ONNX — CelebA
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_celeba_spoof.yaml \
  --model-config configs/model_face_antispoof_onnx.yaml

# Face antispoof ONNX — CASIA
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_casia_fasd.yaml \
  --model-config configs/model_face_antispoof_onnx.yaml

# Ghi đè CSV ảnh (metadata vẫn theo dataset-config)
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_celeba_spoof.yaml \
  --model-config configs/model_minifasnet.yaml \
  --annotation-csv data/sampled/celeba_spoof_sample.csv
```

**Output** (theo `model_id` + `source_dataset`):

- `reports/models/<model_id>/predictions/<source_dataset>/run_<source_dataset>_YYYYMMDD_HHMMSS.csv`
- `reports/models/<model_id>/predictions/<source_dataset>/latest.csv` (ghi đè khi chạy lại **cùng model** + cùng dataset)
- `reports/models/<model_id>/model_manifest.json`

Mỗi model một thư mục `<model_id>` riêng, không đè predictions của model khác.

## Evaluation

Chạy sau **batch inference**. Mặc định đọc `latest.csv` của model trong `configs/evaluation.yaml` (`model_config` + `dataset`).

```bash
# Sau inference CelebA — cần dataset trong evaluation.yaml hoặc --dataset
python3 scripts/run_evaluation.py --dataset celeba_spoof

python3 scripts/run_evaluation.py \
  --predictions reports/models/minifasnet_v2_2p7/predictions/casia_fasd/latest.csv

python3 scripts/run_evaluation.py \
  --predictions reports/models/vitfas_vitb16_224/predictions/celeba_spoof/latest.csv
```

`model_id`: khai báo trong `configs/model_minifasnet.yaml` (A) hoặc tự sinh từ `name` + tên weight (B).

**Output:** `reports/models/<model_id>/metrics/<dataset>/` (`metrics_summary.csv`, `metrics_threshold_*.json`, `confusion_matrix_*.png`).

## Chạy test

Từ thư mục gốc repo (đã activate `.venv`):

```bash
# Tất cả test trong tests/
python3 -m unittest discover -s tests -v; rm -f test_log_*.log

# Từng module
python3 -m unittest tests.test_create_vietnam_hf_dataset -v
python3 -m unittest tests.test_hf_raw -v
python3 -m unittest tests.test_label_studio_tasks -v
python3 -m unittest tests.test_run_evaluation -v
python3 -m unittest tests.test_reports_layout -v
python3 -m unittest tests.test_minifasnet_preprocess -v
python3 -m unittest tests.test_minifasnet_model -v
python3 -m unittest tests.test_vit_fas_model -v; rm -f test_log_*.log
python3 -m unittest tests.test_face_antispoof_onnx_model -v
```