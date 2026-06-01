# Báo cáo đánh giá Anti-Spoofing (MiniFASNet vs ViT-FAS vs FaceAntispoof-ONNX)

> Mẫu báo cáo — điền nội dung vào từng mục. Cập nhật ngày / phiên bản khi hoàn thiện.

| | |
|---|---|
| **Tác giả** |Vũ Hồng Quang|
| **Ngày** |28/05/2026|
| **Repo / branch** |main|
| **Môi trường** | Python, OS, GPU/CPU |

---

## 1. Tóm tắt (Executive summary)

- Mục tiêu: đánh giá stage anti-spoofing cho bài toán verify ảnh tài xế trước khi tích hợp service.
- Model đã đánh giá: **MiniFASNetV2** (`minifasnet_v2_2p7`), **ViT-FAS** (`vitfas_vitb16_224`) và **FaceAntispoof-ONNX** (`face_antispoof_onnx_9820`).
- Dataset đã dùng:
  - CelebA-Spoof sample (4000 ảnh): chủ yếu là ảnh người phương tây.
  - CASIA-FASD sample (4063 ảnh): chủ yếu là ảnh người Trung Quốc, phù hợp hơn với bài toán này.
  - Face Anti-Spoofing VN (`face_antispoofing_vn`, 2118 ảnh test): dữ liệu nội bộ Việt Nam, crop mặt RetinaFace (SFAS).
  - Drivers 250 FN (`drivers_250_fn`, 291 ảnh live đã crop): dữ liệu thực tế các ca false-negative trong vận hành.
- Kết luận chính: MiniFASNet vẫn là model tốt nhất trên CASIA-FASD. FaceAntispoof-ONNX cho kết quả cạnh tranh trên CelebA-Spoof (ACER thấp nhất @0.5) và xếp thứ hai trên CASIA-FASD. Trên **face_antispoofing_vn**, FaceAntispoof-ONNX tốt nhất về accuracy/APCER @0.5 nhưng BPCER cao. Trên **drivers_250_fn** (tập live thực tế), MiniFASNet cho BPCER thấp nhất, FaceAntispoof-ONNX cho BPCER cao nhất.
- Đề xuất tích hợp service: ưu tiên metric vận hành theo **BPCER trên live thực tế**; giữ MiniFASNet làm baseline; dùng ONNX khi cần ưu tiên chống spoof và có calibrate threshold theo production.

---

## 2. Mục tiêu và phạm vi

### 2.1 Mục tiêu

- 
- 

### 2.2 Phạm vi / ngoài phạm vi

- Trong phạm vi:
- Không làm trong đợt này:

## 3. Khảo sát model anti-spoofing

### 3.1 Danh sách model đã xem xét

| Model / repo | Nguồn | Ghi chú |
|--------------|-------|---------|
| MiniFASNet (Silent-Face-Anti-Spoofing) | | |
| Vision Transformers (ViT)| | |
| Face Anti-Spoof ONNX | | |

### 3.2 Tiêu chí so sánh

- Độ chính xác / ACER trên tập test
- Tốc độ inference, yêu cầu GPU
- Dễ deploy (Docker, weight, license)
- Khả năng mở rộng (fine-tune, API)

### 3.3 Model được chọn cho thí nghiệm đầu tiên

- **Tên model:** MiniFASNetV2 (Silent-Face-Anti-Spoofing).
- **Lý do chọn:** phổ biến, nhẹ, dễ tích hợp, có sẵn wrapper/pipeline ổn định trong repo.
- **Weight / config:** `configs/model_minifasnet.yaml`, `models/`.

---

## 4. Dataset

### 4.1 Dataset sử dụng

| Dataset | Nguồn (HF / local) | Ghi chú |
|---------|-------------------|---------|
| CelebA-Spoof | |Chủ yêu là hình ảnh người phương Tây|
| CASIA-FASD | |Chủ yếu là hình ảnh người Trung Quốc, nên phù hợp hơn với bài toán này|
| Face Anti-Spoofing VN | `vu-hong-quang/face_antispoofing_vn` (HF private) | Ảnh tài xế VN, crop SFAS, split `test` |
| Drivers 250 FN | `data/drivers_250_fn_cropped` (local) | Ảnh thực tế false-negative, toàn bộ nhãn live, split `all` |

### 4.2 Số lượng ảnh (sample / test)

| Dataset | Tổng | Live | Spoof | Unknown / lỗi |
|---------|------|------|-------|----------------|
| celeba_spoof | 4000 | 2000 | 2000 | 0 |
| casia_fasd | 4063 | 995 | 3068 | 0 |
| face_antispoofing_vn | 2118 | 782 | 1336 | 0 |
| drivers_250_fn | 291 | 291 | 0 | 0 |

*(Đường dẫn CSV: `data/sampled/`, `data/raw/.../annotations/raw.csv`.)*

### 4.3 Đặc điểm tiền xử lý dữ liệu đầu vào

- CelebA-Spoof (`cropped_image`): không cần tiền xử lý gì thêm
- CASIA-FASD (`cropped_image` 256×256, v.v.): không cần tiền xử lý gì thêm
- Face Anti-Spoofing VN: ảnh đã crop mặt trên HF; inference resize theo từng model (80×80 / 128×128 / 224×224)
- Crop offline CASIA (`data/processed/casia_fasd/`): có / không — ghi chú

---

## 6. Thiết lập thí nghiệm

### 6.1 Cấu hình

| Thành phần | File / giá trị |
|------------|----------------|
| Model | `configs/model_minifasnet.yaml` |
| Inference | `configs/inference.yaml` |
| Dataset CelebA | `configs/dataset_celeba_spoof.yaml` |
| Dataset CASIA (raw / cropped) | `configs/dataset_casia_fasd.yaml` / `dataset_casia_fasd_cropped.yaml` |
| Dataset Face VN | `configs/dataset_face_antispoofing_vn.yaml` |
| Dataset Drivers FN | `configs/dataset_drivers_250_fn.yaml` |
| Evaluation | `configs/evaluation.yaml` |

### 6.2 Preprocessing

- Input size:
- Threshold mặc định:
- Crop detect + scale (CASIA):

### 6.3 Lệnh chạy (tham chiếu)

```bash
# Inference MiniFASNet
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_celeba_spoof.yaml \
  --model-config configs/model_minifasnet.yaml \
  --inference-config configs/inference.yaml

python3 scripts/run_inference.py \
  --dataset-config configs/dataset_casia_fasd.yaml \
  --model-config configs/model_minifasnet.yaml \
  --inference-config configs/inference.yaml

# Inference ViT-FAS
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_celeba_spoof.yaml \
  --model-config configs/model_vit_fas.yaml \
  --inference-config configs/inference.yaml

python3 scripts/run_inference.py \
  --dataset-config configs/dataset_casia_fasd.yaml \
  --model-config configs/model_vit_fas.yaml \
  --inference-config configs/inference.yaml

# Evaluation (sau khi có latest.csv theo từng model)
python3 scripts/run_evaluation.py --dataset celeba_spoof
python3 scripts/run_evaluation.py --dataset casia_fasd

# Inference / evaluation — Face Anti-Spoofing VN
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_face_antispoofing_vn.yaml \
  --model-config configs/model_minifasnet.yaml

python3 scripts/run_evaluation.py --dataset face_antispoofing_vn

# Inference / evaluation — Drivers 250 FN (dữ liệu thực tế)
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_drivers_250_fn.yaml \
  --model-config configs/model_minifasnet.yaml
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_drivers_250_fn.yaml \
  --model-config configs/model_vit_fas.yaml
python3 scripts/run_inference.py \
  --dataset-config configs/dataset_drivers_250_fn.yaml \
  --model-config configs/model_face_antispoof_onnx.yaml

python3 scripts/run_evaluation.py \
  --predictions reports/models/minifasnet_v2_2p7/predictions/drivers_250_fn/latest.csv \
  --dataset drivers_250_fn
python3 scripts/run_evaluation.py \
  --predictions reports/models/vitfas_vitb16_224/predictions/drivers_250_fn/latest.csv \
  --dataset drivers_250_fn
python3 scripts/run_evaluation.py \
  --predictions reports/models/face_antispoof_onnx_9820/predictions/drivers_250_fn/latest.csv \
  --dataset drivers_250_fn
```

### 6.4 Metric báo cáo

- Accuracy, Precision, Recall, F1 (live / spoof)
- **APCER**, **BPCER**, **ACER**
- Confusion matrix (theo threshold)
- Ngưỡng thử: `thresholds: [...]`

---

## 7. Kết quả

### 7.1 CelebA-Spoof

- File predictions:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/predictions/celeba_spoof/latest.csv`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/predictions/celeba_spoof/latest.csv`
  - FaceAntispoof-ONNX: `reports/models/face_antispoof_onnx_9820/predictions/celeba_spoof/latest.csv`
- File metrics:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/metrics/celeba_spoof/`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/metrics/celeba_spoof/`
  - FaceAntispoof-ONNX: `reports/models/face_antispoof_onnx_9820/metrics/celeba_spoof/`

| Model | Threshold | Accuracy | APCER | BPCER | ACER | Ghi chú |
|-------|-----------|----------|-------|-------|------|---------|
| MiniFASNet | 0.3 | 0.7060 | 0.3405 | 0.2475 | 0.2940 | Cân bằng khá tốt |
| MiniFASNet | 0.5 | 0.6883 | 0.2685 | 0.3550 | 0.3118 | Điểm vận hành trung tính |
| MiniFASNet | 0.7 | 0.6670 | 0.1835 | 0.4825 | 0.3330 | Giảm APCER, tăng BPCER |
| ViT-FAS | 0.3 | 0.6685 | 0.2255 | 0.4375 | 0.3315 | BPCER cao |
| ViT-FAS | 0.5 | 0.6443 | 0.1830 | 0.5285 | 0.3558 | Kém MiniFASNet |
| ViT-FAS | 0.7 | 0.6295 | 0.1340 | 0.6070 | 0.3705 | BPCER rất cao |
| FaceAntispoof-ONNX | 0.3 | 0.8365 | 0.2490 | 0.0780 | 0.1635 | Recall live cao, reject live thấp |
| FaceAntispoof-ONNX | 0.5 | 0.8415 | 0.1910 | 0.1260 | 0.1585 | ACER tốt nhất trên CelebA |
| FaceAntispoof-ONNX | 0.7 | 0.8415 | 0.1380 | 0.1790 | 0.1585 | Cân bằng APCER/BPCER tốt |

- Confusion matrix @ threshold 0.5 (dạng bảng):

| Model | TP (live→live) | FN (live→spoof) | FP (spoof→live) | TN (spoof→spoof) |
|-------|-----------------|-----------------|-----------------|------------------|
| MiniFASNet | 1290 | 710 | 537 | 1463 |
| ViT-FAS | 943 | 1057 | 366 | 1634 |
| FaceAntispoof-ONNX | 1748 | 252 | 382 | 1618 |

- Nhận xét ngắn: FaceAntispoof-ONNX cho kết quả vượt trội trên CelebA-Spoof so với hai model còn lại; ViT-FAS vẫn có xu hướng reject nhiều live khi threshold tăng.

### 7.2 CASIA-FASD

- Điều kiện chạy: sample CASIA-FASD theo annotation hiện có.
- File predictions:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/predictions/casia_fasd/latest.csv`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/predictions/casia_fasd/latest.csv`
  - FaceAntispoof-ONNX: `reports/models/face_antispoof_onnx_9820/predictions/casia_fasd/latest.csv`
- File metrics:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/metrics/casia_fasd/`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/metrics/casia_fasd/`
  - FaceAntispoof-ONNX: `reports/models/face_antispoof_onnx_9820/metrics/casia_fasd/`

| Model | Threshold | Accuracy | APCER | BPCER | ACER | Ghi chú |
|-------|-----------|----------|-------|-------|------|---------|
| MiniFASNet | 0.3 | 0.9488 | 0.0290 | 0.1196 | 0.0743 | Rất tốt |
| MiniFASNet | 0.5 | 0.9527 | 0.0130 | 0.1528 | 0.0829 | Điểm vận hành đề xuất |
| MiniFASNet | 0.7 | 0.9488 | 0.0052 | 0.1930 | 0.0991 | Ưu tiên giảm APCER |
| ViT-FAS | 0.3 | 0.5843 | 0.4977 | 0.1628 | 0.3303 | APCER cao |
| ViT-FAS | 0.5 | 0.6215 | 0.4234 | 0.2402 | 0.3318 | Vẫn kém xa |
| ViT-FAS | 0.7 | 0.6520 | 0.3504 | 0.3407 | 0.3455 | Không đạt mức deploy |
| FaceAntispoof-ONNX | 0.3 | 0.6638 | 0.4195 | 0.0794 | 0.2494 | Ưu tiên giữ live, lọt spoof cao |
| FaceAntispoof-ONNX | 0.5 | 0.7061 | 0.3556 | 0.1035 | 0.2296 | Điểm cân bằng hiện tại |
| FaceAntispoof-ONNX | 0.7 | 0.7413 | 0.2963 | 0.1427 | 0.2195 | Tốt nhất trong 3 ngưỡng đã thử |

- Confusion matrix @ threshold 0.5 (dạng bảng):

| Model | TP (live→live) | FN (live→spoof) | FP (spoof→live) | TN (spoof→spoof) |
|-------|-----------------|-----------------|-----------------|------------------|
| MiniFASNet | 843 | 152 | 40 | 3028 |
| ViT-FAS | 756 | 239 | 1299 | 1769 |
| FaceAntispoof-ONNX | 892 | 103 | 1091 | 1977 |

- Nhận xét ngắn: trên CASIA-FASD, MiniFASNet vẫn vượt trội rõ rệt. FaceAntispoof-ONNX xếp thứ hai, tốt hơn ViT-FAS nhưng APCER còn cao.

### 7.3 Face Anti-Spoofing VN (`face_antispoofing_vn`)

- Điều kiện chạy: toàn bộ split `test` sau `hf_raw` (2118 mẫu: 782 live, 1336 spoof); sample CSV `data/sampled/face_antispoofing_vn_sample.csv`.
- File predictions:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/predictions/face_antispoofing_vn/latest.csv`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/predictions/face_antispoofing_vn/latest.csv`
  - FaceAntispoof-ONNX: `reports/models/face_antispoof_onnx_9820/predictions/face_antispoofing_vn/latest.csv`
- File metrics:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn/`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn/`
  - FaceAntispoof-ONNX: `reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn/`

| Model | Threshold | Accuracy | APCER | BPCER | ACER | Ghi chú |
|-------|-----------|----------|-------|-------|------|---------|
| MiniFASNet | 0.3 | 0.7436 | 0.2141 | 0.3286 | 0.2714 | Cân bằng trung bình |
| MiniFASNet | 0.5 | 0.7502 | 0.1587 | 0.4054 | 0.2820 | BPCER cao (nhiều live → spoof) |
| MiniFASNet | 0.7 | 0.7545 | 0.1003 | 0.4936 | 0.2970 | APCER thấp, BPCER rất cao |
| ViT-FAS | 0.3 | 0.5415 | 0.3945 | 0.5678 | 0.4811 | Chưa đạt mức deploy |
| ViT-FAS | 0.5 | 0.5680 | 0.3076 | 0.6445 | 0.4761 | ACER cao nhất trong 3 model |
| ViT-FAS | 0.7 | 0.5897 | 0.2290 | 0.7199 | 0.4745 | Cải thiện nhẹ accuracy, vẫn kém |
| FaceAntispoof-ONNX | 0.3 | 0.7828 | 0.0906 | 0.4335 | 0.2620 | ACER thấp nhất @0.3 |
| FaceAntispoof-ONNX | 0.5 | 0.7691 | 0.0681 | 0.5090 | 0.2885 | Accuracy cao nhất; APCER thấp nhất @0.5 |
| FaceAntispoof-ONNX | 0.7 | 0.7573 | 0.0419 | 0.5857 | 0.3138 | Ưu tiên bảo mật (APCER rất thấp) |

- Confusion matrix @ threshold 0.5 (dạng bảng):

| Model | TP (live→live) | FN (live→spoof) | FP (spoof→live) | TN (spoof→spoof) |
|-------|-----------------|-----------------|-----------------|------------------|
| MiniFASNet | 465 | 317 | 212 | 1124 |
| ViT-FAS | 278 | 504 | 411 | 925 |
| FaceAntispoof-ONNX | 384 | 398 | 91 | 1245 |

- Nhận xét ngắn: trên dữ liệu người Việt, **FaceAntispoof-ONNX** cho accuracy và APCER tốt nhất @0.5 (ít spoof lọt thành live), nhưng **BPCER cao** (nhiều live bị reject). **MiniFASNet** đứng thứ hai, cân bằng hơn BPCER @0.5 nhưng APCER cao hơn ONNX. **ViT-FAS** kém rõ trên domain VN (ACER ~0.48 @0.5) — khác với xu hướng trên CASIA/CelebA không đồng nhất, cần xem lại checkpoint/domain.

### 7.4 Drivers 250 FN (dữ liệu thực tế, đã crop)

- Điều kiện chạy: `data/drivers_250_fn_cropped/` + `data/sampled/drivers_250_fn_sample.csv` (291 mẫu live, split `all`).
- File predictions:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/predictions/drivers_250_fn/latest.csv`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/predictions/drivers_250_fn/latest.csv`
  - FaceAntispoof-ONNX: `reports/models/face_antispoof_onnx_9820/predictions/drivers_250_fn/latest.csv`
- File metrics:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn/`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/metrics/drivers_250_fn/`
  - FaceAntispoof-ONNX: `reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn/`

| Model | Threshold | Accuracy | APCER | BPCER | ACER | Ghi chú |
|-------|-----------|----------|-------|-------|------|---------|
| MiniFASNet | 0.3 | 0.6735 | 0.0000 | 0.3265 | 0.1632 | BPCER thấp nhất trong 3 model |
| MiniFASNet | 0.5 | 0.5636 | 0.0000 | 0.4364 | 0.2182 | Điểm vận hành hiện tại |
| MiniFASNet | 0.7 | 0.4502 | 0.0000 | 0.5498 | 0.2749 | BPCER tăng mạnh khi tăng threshold |
| ViT-FAS | 0.3 | 0.4983 | 0.0000 | 0.5017 | 0.2509 | Chất lượng thấp trên dữ liệu thực tế |
| ViT-FAS | 0.5 | 0.4227 | 0.0000 | 0.5773 | 0.2887 | Kém MiniFASNet |
| ViT-FAS | 0.7 | 0.3333 | 0.0000 | 0.6667 | 0.3333 | BPCER rất cao |
| FaceAntispoof-ONNX | 0.3 | 0.4330 | 0.0000 | 0.5670 | 0.2835 | Ưu tiên reject (an toàn) |
| FaceAntispoof-ONNX | 0.5 | 0.3643 | 0.0000 | 0.6357 | 0.3179 | BPCER cao nhất @0.5 |
| FaceAntispoof-ONNX | 0.7 | 0.3058 | 0.0000 | 0.6942 | 0.3471 | Không phù hợp UX hiện tại |

- Confusion matrix @ threshold 0.5 (dạng bảng):

| Model | TP (live→live) | FN (live→spoof) | FP (spoof→live) | TN (spoof→spoof) |
|-------|-----------------|-----------------|-----------------|------------------|
| MiniFASNet | 164 | 127 | 0 | 0 |
| ViT-FAS | 123 | 168 | 0 | 0 |
| FaceAntispoof-ONNX | 106 | 185 | 0 | 0 |

- Nhận xét ngắn: với dữ liệu production false-negative, thứ hạng đảo chiều so với kết luận theo accuracy trên tập VN: **MiniFASNet tốt nhất theo BPCER**, ONNX tệ nhất theo BPCER.

### 7.5 So sánh giữa các dataset

| Tiêu chí | CelebA-Spoof | CASIA-FASD | Face Anti-Spoofing VN | Drivers 250 FN |
|----------|--------------|------------|------------------------|----------------|
| MiniFASNet ACER @ 0.5 | 0.3118 | 0.0829 | 0.2820 | 0.2182 |
| ViT-FAS ACER @ 0.5 | 0.3558 | 0.3318 | 0.4761 | 0.2887 |
| FaceAntispoof-ONNX ACER @ 0.5 | 0.1585 | 0.2296 | 0.2885 | 0.3179 |
| MiniFASNet BPCER @ 0.5 | 0.3550 | 0.1528 | 0.4054 | 0.4364 |
| ViT-FAS BPCER @ 0.5 | 0.5285 | 0.2402 | 0.6445 | 0.5773 |
| FaceAntispoof-ONNX BPCER @ 0.5 | 0.1260 | 0.1035 | 0.5090 | 0.6357 |
| Số mẫu (test/sample) | 4000 | 4063 | 2118 (782 live / 1336 spoof) | 291 (291 live / 0 spoof) |
| Độ khó / domain | CelebA cân bằng, phân tách khó | CASIA khớp train MiniFASNet | VN OOD; ONNX hơn MiniFASNet theo accuracy | Production hard cases (false-negative) |
| Preprocess đầu vào | resize + normalize theo wrapper | resize + normalize theo wrapper | crop SFAS + resize theo wrapper | crop SFAS local + resize theo wrapper |

### 7.6 Artifact minh họa (không dùng ảnh confusion matrix)

- Trước / sau crop (tham khảo): `reports/casia_crop_before_after_10.png`
- Failure cases: `reports/failure_cases/`

---

## 8. Phân tích failure cases

### 8.1 False accept (spoof → live) — rủi ro bảo mật

- Số lượng / tỷ lệ:
- Ví dụ điển hình (đường dẫn ảnh):
- Pattern (màn hình HD, in ảnh, ánh sáng, …):

### 8.2 False reject (live → spoof) — ảnh hưởng UX

- Số lượng / tỷ lệ:
- Ví dụ:
- Pattern (mờ, nghiêng, tối, …):

### 8.3 Lỗi kỹ thuật (detect / inference)

- Không detect mặt:
- Lỗi đọc ảnh / preprocess:

---

## 9. Thảo luận

### 9.1 Domain và độ khớp dữ liệu với pretrained

- 

### 9.2 Ảnh đã crop sẵn (HF) vs crop offline

- 

### 9.3 Ưu tiên metric cho production (APCER vs Accuracy)

- 

### 9.4 Hạn chế thí nghiệm

- Chỉ một model / một weight
- Sample size, chưa fine-tune
- Khác:

---

## 10. Kết luận và đề xuất

### 10.1 Kết luận

- Với dữ liệu thực tế `drivers_250_fn` (291 mẫu live), tiêu chí quyết định là **BPCER** (tỷ lệ live bị reject).
- Xếp hạng phù hợp production hiện tại trên `drivers_250_fn` (@0.5): **MiniFASNet (BPCER 0.4364) > ViT-FAS (0.5773) > FaceAntispoof-ONNX (0.6357)**.
- Kết luận triển khai: **MiniFASNet** là lựa chọn phù hợp nhất ở thời điểm hiện tại cho luồng verify tài xế thực tế; ONNX chỉ phù hợp nếu ưu tiên rất cao việc chặn spoof và chấp nhận reject live nhiều.

### 10.2 Phân tích từng mô hình trên `drivers_250_fn`

| Model | Điểm mạnh | Điểm yếu | Nhận định phù hợp |
|------|-----------|----------|-------------------|
| MiniFASNet (`minifasnet_v2_2p7`) | BPCER thấp nhất trong 3 model trên tập thực tế; ổn định hơn khi thay đổi threshold; cho tỷ lệ pass live cao nhất | APCER trên các tập cân bằng không phải tốt nhất; vẫn reject nhiều live ở ngưỡng cao | **Phù hợp nhất hiện tại** cho production nếu mục tiêu chính là giảm false reject tài xế thật |
| ViT-FAS (`vitfas_vitb16_224`) | Có thể chạy end-to-end ổn định trong pipeline hiện tại | BPCER cao trên `drivers_250_fn`; ACER cao trên VN/CASIA; chất lượng chưa đạt mức deploy | Chưa phù hợp để làm model chính; chỉ nên giữ cho mục đích benchmark nội bộ |
| FaceAntispoof-ONNX (`face_antispoof_onnx_9820`) | APCER rất tốt trên VN/CelebA; mạnh ở mục tiêu giảm spoof lọt | BPCER cao nhất trên `drivers_250_fn` (reject live nhiều); độ phù hợp production thấp nếu KPI ưu tiên UX | Phù hợp cho kịch bản ưu tiên bảo mật cao; cần calibrate threshold/fine-tune trước khi dùng làm model chính |

### 10.3 Đề xuất tích hợp service

| Hạng mục | Đề xuất |
|----------|---------|
| Model anti-spoofing | Chọn **MiniFASNetV2** làm model chính cho production hiện tại; ONNX để phương án phụ/ensemble sau calibrate |
| Ngưỡng operating point | Không cố định `0.5`; tune theo KPI production với ràng buộc BPCER mục tiêu trên tập live thực tế |
| Preprocessing (detect + crop) | VN: giữ crop SFAS nhất quán với bước build HF; resize theo từng model |
| Bước tiếp theo (fine-tune, data nội bộ) | Ưu tiên giảm BPCER trên `drivers_250_fn`: calibrate threshold theo production, mở rộng tập live khó, đánh giá lại ONNX sau fine-tune |

### 10.4 Công việc tiếp theo

- [ ] Kiểm tra sâu nhánh load/checkpoint ViT-FAS và thống nhất class order live/spoof.
- [ ] Calibrate threshold theo KPI production (ưu tiên BPCER trên live thực tế `drivers_250_fn`).
- [ ] Chạy lại ONNX với sweep threshold + báo cáo trade-off APCER/BPCER theo ngưỡng.
- [ ] Chạy benchmark lặp lại nhiều seed để đo độ ổn định metric.
- [ ] Bổ sung tập validation nội bộ để chọn threshold theo KPI service.

---

## Phụ lục

### A. Cây thư mục artifact chính

```
reports/
  predictions/
  metrics/<dataset>/
  failure_cases/
  REPORT.md
data/
  raw/
  sampled/
  processed/
  labeled/
```

### B. Lệnh pipeline đầy đủ (copy từ README)

*(Điền hoặc link README.md.)*

### C. Tài liệu tham khảo

- Silent-Face-Anti-Spoofing / MiniFASNet
- CelebA-Spoof
- CASIA-FASD
- ISO/IEC 30107 (APCER, BPCER, ACER)

### D. Changelog báo cáo

| Ngày | Phiên bản | Thay đổi |
|------|-----------|----------|
| 28/05/2026 | v0.1 | Khởi tạo mẫu |
| 29/05/2026 | v0.2 | Bổ sung kết quả `face_antispoofing_vn` (3 model) |
| 01/06/2026 | v0.3 | Bổ sung đánh giá `drivers_250_fn`, thêm metric BPCER trọng tâm cho production, thay confusion matrix ảnh bằng bảng markdown |
| 01/06/2026 | v0.4 | Cập nhật mục 10: phân tích model phù hợp nhất cho `drivers_250_fn`, nêu điểm mạnh/yếu từng model theo dữ liệu thực tế |
