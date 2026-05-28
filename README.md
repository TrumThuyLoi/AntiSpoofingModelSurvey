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
- **CASIA-FASD** (`casia_fasd`): repo private — thêm vào file `.env` ở thư mục gốc repo:

```env
HUGGINGFACE_CASIA_FASD_TOKEN=hf_...
```

**Chạy tải dữ liệu** (từ thư mục gốc DVX):

```bash
# CelebA-Spoof (~67k ảnh, có thể mất lâu)
python -m src.datasets.hf_raw --dataset celeba_spoof

# CASIA-FASD
python -m src.datasets.hf_raw --dataset casia_fasd

# Tải lại dù đã có dữ liệu
python -m src.datasets.hf_raw --dataset casia_fasd --force
```

Kết quả mẫu:

```text
data/raw/celeba_spoof/   hoặc   data/raw/casia_fasd/
  images/test/000000.jpg ...
  annotations/raw.csv
  meta/download_manifest.json
```

Gọi từ Python:

```python
from src.datasets.hf_raw import download_raw_dataset
from src.datasets.specs import CELEBA_SPOOF_SPEC, CASIA_FASD_SPEC

download_raw_dataset(CELEBA_SPOOF_SPEC)
download_raw_dataset(CASIA_FASD_SPEC)
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

## Batch inference

```bash
# CelebA-Spoof
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_celeba_spoof.yaml

# CASIA-FASD
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_casia_fasd.yaml

# Override nhanh annotation CSV (metadata vẫn theo dataset-config đã chọn)
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_casia_fasd.yaml \
  --annotation-csv data/sampled/casia_fasd_sample.csv
```

Kết quả (theo `model_id` trong `configs/model_minifasnet.yaml`, mặc định `minifasnet_v2_2p7`):

- `reports/models/<model_id>/predictions/<dataset>/run_<dataset>_YYYYMMDD_HHMMSS.csv`
- `reports/models/<model_id>/predictions/<dataset>/latest.csv` (chỉ ghi đè khi chạy lại **cùng model**)
- `reports/models/<model_id>/model_manifest.json`

Model khác → thư mục `<model_id>` khác, không đè lẫn nhau.

## Evaluation

Chạy sau **batch inference**. Mặc định đọc `latest.csv` của model trong `configs/evaluation.yaml` (`model_config` + `dataset`).

```bash
# Sau inference CelebA — cần dataset trong evaluation.yaml hoặc --dataset
python3 scripts/run_evaluation.py --dataset celeba_spoof

python3 scripts/run_evaluation.py \
  --predictions reports/models/minifasnet_v2_2p7/predictions/casia_fasd/latest.csv

python3 scripts/run_evaluation.py \
  --predictions reports/models/vitfas_vitb16_224/predictions/celeba_spoof/latest.c
sv
```

`model_id`: khai báo trong `configs/model_minifasnet.yaml` (A) hoặc tự sinh từ `name` + tên weight (B).

**Output:** `reports/models/<model_id>/metrics/<dataset>/` (`metrics_summary.csv`, `metrics_threshold_*.json`, `confusion_matrix_*.png`).

## Chạy test

```bash
python3 -m unittest tests.test_hf_raw -v
python3 -m unittest tests.test_label_studio_tasks -v
python3 -m unittest tests.test_run_evaluation -v
python3 -m unittest tests.test_reports_layout -v
python3 -m unittest discover -s tests -v
```