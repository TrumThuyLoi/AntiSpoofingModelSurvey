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
- Kết luận chính: MiniFASNet vẫn là model tốt nhất trên CASIA-FASD. FaceAntispoof-ONNX cho kết quả cạnh tranh trên CelebA-Spoof (ACER thấp nhất @0.5) và xếp thứ hai trên CASIA-FASD. Trên **face_antispoofing_vn**, FaceAntispoof-ONNX tốt nhất về accuracy/APCER @0.5; MiniFASNet xếp thứ hai; ViT-FAS chưa đạt mức deploy.
- Đề xuất tích hợp service: giữ MiniFASNet làm baseline trên domain châu Á (CASIA); cân nhắc FaceAntispoof-ONNX cho tập người Việt hoặc CelebA; calibrate threshold riêng theo từng dataset.

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

### 4.2 Số lượng ảnh (sample / test)

| Dataset | Tổng | Live | Spoof | Unknown / lỗi |
|---------|------|------|-------|----------------|
| celeba_spoof | 4000 | 2000 | 2000 | 0 |
| casia_fasd | 4063 | 995 | 3068 | 0 |
| face_antispoofing_vn | 2118 | 782 | 1336 | 0 |

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

- Confusion matrix: `confusion_matrix_*.png`
- MiniFASNet (threshold 0.5):
![MiniFASNet CelebA Confusion Matrix](models/minifasnet_v2_2p7/metrics/celeba_spoof/confusion_matrix_0.5.png)

- ViT-FAS (threshold 0.5):
![ViT-FAS CelebA Confusion Matrix](models/vitfas_vitb16_224/metrics/celeba_spoof/confusion_matrix_0.5.png)

- FaceAntispoof-ONNX (threshold 0.5):
![FaceAntispoof-ONNX CelebA Confusion Matrix](models/face_antispoof_onnx_9820/metrics/celeba_spoof/confusion_matrix_0.5.png)

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

- Confusion matrix:
- MiniFASNet (threshold 0.5):
![MiniFASNet CASIA Confusion Matrix](models/minifasnet_v2_2p7/metrics/casia_fasd/confusion_matrix_0.5.png)

- ViT-FAS (threshold 0.5):
![ViT-FAS CASIA Confusion Matrix](models/vitfas_vitb16_224/metrics/casia_fasd/confusion_matrix_0.5.png)

- FaceAntispoof-ONNX (threshold 0.5):
![FaceAntispoof-ONNX CASIA Confusion Matrix](models/face_antispoof_onnx_9820/metrics/casia_fasd/confusion_matrix_0.5.png)

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

- Confusion matrix: chưa export PNG cho dataset này (chỉ `metrics_summary.csv`, `metrics_threshold_*.json`).

- Nhận xét ngắn: trên dữ liệu người Việt, **FaceAntispoof-ONNX** cho accuracy và APCER tốt nhất @0.5 (ít spoof lọt thành live), nhưng **BPCER cao** (nhiều live bị reject). **MiniFASNet** đứng thứ hai, cân bằng hơn BPCER @0.5 nhưng APCER cao hơn ONNX. **ViT-FAS** kém rõ trên domain VN (ACER ~0.48 @0.5) — khác với xu hướng trên CASIA/CelebA không đồng nhất, cần xem lại checkpoint/domain.

### 7.4 So sánh giữa các dataset

| Tiêu chí | CelebA-Spoof | CASIA-FASD | Face Anti-Spoofing VN |
|----------|--------------|------------|------------------------|
| MiniFASNet ACER @ 0.5 | 0.3118 | 0.0829 | 0.2820 |
| ViT-FAS ACER @ 0.5 | 0.3558 | 0.3318 | 0.4761 |
| FaceAntispoof-ONNX ACER @ 0.5 | 0.1585 | 0.2296 | 0.2885 |
| MiniFASNet APCER @ 0.5 | 0.2685 | 0.0130 | 0.1587 |
| ViT-FAS APCER @ 0.5 | 0.1830 | 0.4234 | 0.3076 |
| FaceAntispoof-ONNX APCER @ 0.5 | 0.1910 | 0.3556 | 0.0681 |
| Số mẫu (test/sample) | 4000 | 4063 | 2118 (782 live / 1336 spoof) |
| Độ khó / domain | CelebA cân bằng, phân tách khó | CASIA khớp train MiniFASNet | VN OOD; ONNX hơn MiniFASNet |
| Preprocess đầu vào | resize + normalize theo wrapper | resize + normalize theo wrapper | crop SFAS + resize theo wrapper |

### 7.5 Ảnh minh họa (tuỳ chọn)

- Trước / sau crop: `reports/casia_crop_before_after_10.png`
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

- MiniFASNet (`minifasnet_v2_2p7`) hiện là mô hình tốt nhất trên **CASIA-FASD**; trên **face_antispoofing_vn** xếp thứ hai (ACER @0.5 ≈ 0.28).
- ViT-FAS đã chạy end-to-end thành công nhưng quality còn thấp trên mọi tập; trên VN đặc biệt kém (ACER @0.5 ≈ 0.48) — cần kiểm chứng checkpoint/domain trước khi dùng production.
- FaceAntispoof-ONNX (`face_antispoof_onnx_9820`) rất tốt trên CelebA-Spoof; thứ hai trên CASIA-FASD; **tốt nhất trên face_antispoofing_vn** @0.5 (accuracy 0.769, APCER 0.068) nhưng BPCER cao (~0.51).

### 10.2 Đề xuất tích hợp service

| Hạng mục | Đề xuất |
|----------|---------|
| Model anti-spoofing | CASIA/domain châu Á: MiniFASNetV2; tập người VN: cân nhắc FaceAntispoof-ONNX hoặc ensemble |
| Ngưỡng operating point | Bắt đầu với `0.5`, tune riêng theo dataset (VN: cân nhắc giảm APCER với ONNX @0.7 hoặc tăng threshold) |
| Preprocessing (detect + crop) | VN: giữ crop SFAS nhất quán với bước build HF; resize theo từng model |
| Bước tiếp theo (fine-tune, data nội bộ) | Fine-tune trên `face_antispoofing_vn` train; calibrate threshold trên VN; giảm BPCER ONNX |

### 10.3 Công việc tiếp theo

- [ ] Kiểm tra sâu nhánh load/checkpoint ViT-FAS và thống nhất class order live/spoof.
- [ ] Calibrate threshold cho FaceAntispoof-ONNX để giảm APCER trên CASIA-FASD.
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
