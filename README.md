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

## SFAS crop (bbox expansion)

Hằng số: `scripts/sfas_bbox_expansions.py` → `SFAS_BBOX_EXPANSIONS = (1.6, 2.7)` (crop/benchmark tự động). Baseline SFAS exp **1.0** (`drivers_250_fn`, `face_antispoofing_vn`) dùng config/dữ liệu riêng, không nằm trong tuple này.

**Một lệnh crop cả hai tập** (drivers + face VN):

```bash
git submodule update --init third_party/Silent-Face-Anti-Spoofing
python3 scripts/download_pretrained_weights.py
python3 scripts/crop_sfas_expansions.py
```

### Drivers 250 FN

Ảnh gốc: `data/drivers_250_FN/` (291 live). Metric: **BPCER**. Chi tiết: `drivers_250_fn_sfas_crop_and_evaluation.md`.

```bash
python3 scripts/create_test_sample_annotation.py --dataset drivers_250_fn
python3 scripts/create_drivers_250_fn_dataset_configs.py
python3 scripts/run_drivers_250_fn_expansion_benchmark.py --model-config configs/model_minifasnet.yaml
```

### Face Anti-Spoofing VN (expansion ablation)

Ảnh gốc: `data/face_antispoofing_vn/` (cùng layout `train_photo|test_photo/{live,not_live}/`). Chi tiết: `face_antispoofing_vn_sfas_crop_and_evaluation.md`.

```bash
python3 scripts/create_test_sample_annotation.py --dataset face_antispoofing_vn
python3 scripts/create_face_antispoofing_vn_expansion_configs.py
python3 scripts/run_face_vn_expansion_benchmark.py --model-config configs/model_minifasnet.yaml
```

*(Chạy `crop_sfas_expansions.py` trước khi tạo sample, giống drivers.)*

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

`configs/dataset_*.yaml` trỏ tới `data/sampled/*_sample.csv`. HF: tạo sau khi có `data/raw/<dataset>/annotations/raw.csv`. **Drivers 250 FN:** xem mục [Drivers 250 FN](#drivers-250-fn--crop-sfas) (crop trước, rồi `create_test_sample_annotation.py --dataset drivers_250_fn`).

```bash
python3 scripts/create_test_sample_annotation.py   # celeba + casia + face_vn (+ drivers nếu đã crop)
```

- `celeba_spoof_sample.csv` — 2000 live + 2000 spoof
- `casia_fasd_sample.csv`, `face_antispoofing_vn_sample.csv` — toàn bộ `is_valid=true`
- `drivers_250_fn*_sample.csv` — 4 file (một expansion một CSV), nhãn `live`

## Inference + evaluation

### Một model, một dataset

Hai lệnh nối tiếp (`run_inference.py` in đường dẫn `latest.csv` ra stdout):

```bash
PRED=$(python3 scripts/run_inference.py \
  --dataset-config configs/dataset_casia_fasd.yaml \
  --model-config configs/model_minifasnet.yaml)
python3 scripts/run_evaluation.py \
  --predictions "$PRED" \
  --dataset casia_fasd \
  --model-id minifasnet_v2_2p7
```

- `--dataset` = `source_dataset` trong dataset config; `--model-id` = `model_id` trong model config.

### Ma trận toàn repo (mọi `model_*.yaml` × mọi `dataset_*.yaml`)

Sau khi có `data/sampled/*_sample.csv` (và đã crop drivers nếu dùng tập FN):

```bash
python3 scripts/run_inference.py --all
python3 scripts/run_evaluation.py --all
```

- Inference: quét `configs/model_*.yaml` × `configs/dataset_*.yaml` (hiện **3 × 7 = 21** cặp).
- Evaluation: cùng ma trận; **bỏ qua** cặp chưa có `reports/models/<model_id>/predictions/<source_dataset>/latest.csv` (chạy inference trước hoặc sau từng cặp thiếu).

**Output:** `reports/models/<model_id>/predictions/<source_dataset>/latest.csv` và `metrics/<source_dataset>/` (`metrics_summary.csv`, `apcer_bpcer_vs_threshold.png`, `confusion_matrix_*.png`, …).

**Drivers (chỉ 4 expansion, một model):** vòng `for` ở mục [Drivers 250 FN](#drivers-250-fn--crop-sfas) hoặc `run_drivers_250_fn_expansion_benchmark.py` (inference + evaluation, không gồm CelebA/CASIA/VN).

## Batch inference

Chỉ inference. Luồng đủ + ma trận `--all`: mục [Inference + evaluation](#inference--evaluation).

Cần weights, submodule, `data/sampled/*_sample.csv`.

**Tham số CLI**

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--dataset-config` | `configs/dataset.yaml` | Một dataset (`source_dataset` → thư mục predictions) |
| `--model-config` | `configs/model_minifasnet.yaml` | Model + `model_id` |
| `--inference-config` | `configs/inference.yaml` | `batch_size`, `save_raw_output`, … |
| `--annotation-csv` | *(từ dataset config)* | Ghi đè CSV ảnh |
| `--all` | — | Mọi `configs/model_*.yaml` × `configs/dataset_*.yaml` |

**Dataset** (`--dataset-config`)

| File | `source_dataset` | Annotation mặc định |
|------|------------------|---------------------|
| `configs/dataset_celeba_spoof.yaml` | `celeba_spoof` | `data/sampled/celeba_spoof_sample.csv` |
| `configs/dataset_casia_fasd.yaml` | `casia_fasd` | `data/sampled/casia_fasd_sample.csv` |
| `configs/dataset_face_antispoofing_vn.yaml` | `face_antispoofing_vn` | `data/sampled/face_antispoofing_vn_sample.csv` (exp 1.0 crop) |
| `configs/dataset_face_antispoofing_vn_exp*.yaml` | `face_antispoofing_vn_exp*` | `data/sampled/face_antispoofing_vn_exp*_sample.csv` |
| `configs/dataset_drivers_250_fn.yaml` | `drivers_250_fn` | `data/sampled/drivers_250_fn_sample.csv` |
| `configs/dataset_drivers_250_fn_exp*.yaml` | `drivers_250_fn_exp*` | `data/sampled/drivers_250_fn_exp*_sample.csv` |

*(Cần `crop_sfas_expansions.py` + script tạo config tương ứng; xem [SFAS crop](#sfas-crop-bbox-expansion).)*

**Model** (`--model-config`)

| File | `model_id` | Weight |
|------|------------|--------|
| `configs/model_minifasnet.yaml` | `minifasnet_v2_2p7` | `models/2.7_80x80_MiniFASNetV2.pth` |
| `configs/model_vit_fas.yaml` | `vitfas_vitb16_224` | `models/vitfas_vitb16_224x224.pth` |
| `configs/model_face_antispoof_onnx.yaml` | `face_antispoof_onnx_9820` | `models/face_antispoof_onnx_best_9820.pth` |

**Ví dụ** (một cặp; ma trận đủ: `--all`):

```bash
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_celeba_spoof.yaml \
  --model-config configs/model_vit_fas.yaml
```

**Output** (theo `model_id` + `source_dataset`):

- `reports/models/<model_id>/predictions/<source_dataset>/run_<source_dataset>_YYYYMMDD_HHMMSS.csv`
- `reports/models/<model_id>/predictions/<source_dataset>/latest.csv` (ghi đè khi chạy lại **cùng model** + cùng dataset)
- `reports/models/<model_id>/model_manifest.json`

Mỗi model một thư mục `<model_id>` riêng, không đè predictions của model khác.

## Evaluation

Chạy sau inference — xem [Inference + evaluation](#inference--evaluation). Ngưỡng: `configs/evaluation.yaml` (`thresholds`).

| Tham số | Ý nghĩa |
|---------|---------|
| `--predictions` | Đường dẫn `latest.csv` (khuyến nghị kèm `--dataset`, `--model-id`) |
| `--dataset` | `source_dataset` (thư mục metrics) |
| `--model-id` | Namespace `reports/models/<model_id>/` |
| `--all` | Ma trận model×dataset; skip cặp thiếu predictions |

```bash
python3 scripts/run_evaluation.py \
  --predictions reports/models/minifasnet_v2_2p7/predictions/casia_fasd/latest.csv \
  --dataset casia_fasd \
  --model-id minifasnet_v2_2p7
```

Hoặc `python3 scripts/run_evaluation.py --dataset celeba_spoof` khi `evaluation.yaml` đã khớp model + dataset.

## Chạy test

Từ thư mục gốc repo (đã activate `.venv`):

```bash
# Tất cả test trong tests/
python3 -m unittest discover -s tests -v; rm -f test_log_*.log

# Từng module
python3 -m unittest tests.test_create_vietnam_hf_dataset -v
python3 -m unittest tests.test_sfas_bbox_expansions -v
python3 -m unittest tests.test_crop_sfas_expansions -v
python3 -m unittest tests.test_create_face_antispoofing_vn_expansion_configs -v
python3 -m unittest tests.test_run_face_vn_expansion_benchmark -v
python3 -m unittest tests.test_create_test_sample_annotation_drivers -v
python3 -m unittest tests.test_create_drivers_250_fn_dataset_configs -v
python3 -m unittest tests.test_run_drivers_250_fn_expansion_benchmark -v
python3 -m unittest tests.test_hf_raw -v
python3 -m unittest tests.test_label_studio_tasks -v
python3 -m unittest tests.test_run_evaluation tests.test_run_inference -v
python3 -m unittest tests.test_reports_layout -v
python3 -m unittest tests.test_minifasnet_preprocess -v
python3 -m unittest tests.test_minifasnet_model -v
python3 -m unittest tests.test_vit_fas_model -v; rm -f test_log_*.log
python3 -m unittest tests.test_face_antispoof_onnx_model -v

# SFAS expansion (drivers + face VN)
python3 -m unittest \
  tests.test_sfas_bbox_expansions \
  tests.test_crop_sfas_expansions \
  tests.test_create_test_sample_annotation_drivers \
  tests.test_create_drivers_250_fn_dataset_configs \
  tests.test_create_face_antispoofing_vn_expansion_configs \
  tests.test_run_drivers_250_fn_expansion_benchmark \
  tests.test_run_face_vn_expansion_benchmark \
  tests.test_create_vietnam_hf_dataset.TestCropBgrFaceSfasExpanded \
  tests.test_create_vietnam_hf_dataset.TestDetectAndCropRgb \
  -v
```