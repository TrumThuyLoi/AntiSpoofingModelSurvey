# Báo cáo đánh giá Anti-Spoofing (MiniFASNet vs ViT-FAS)

> Mẫu báo cáo — điền nội dung vào từng mục. Cập nhật ngày / phiên bản khi hoàn thiện.

| | |
|---|---|
| **Tác giả** |Vũ Hông Quang|
| **Ngày** |28/05/2026|
| **Repo / branch** |main|
| **Môi trường** | Python, OS, GPU/CPU |

---

## 1. Tóm tắt (Executive summary)

- Mục tiêu: đánh giá stage anti-spoofing cho bài toán verify ảnh tài xế trước khi tích hợp service.
- Model đã đánh giá: **MiniFASNetV2** (`minifasnet_v2_2p7`) và **ViT-FAS** (`vitfas_vitb16_224`).
- Dataset đã dùng: 
  - CelebA-Spoof sample (4000 ảnh): chủ yếu là ảnh người phương tây.
  - CASIA-FASD sample (4063 ảnh): chủ yếu là ảnh người Trung Quốc, phù hợp hơn với bài toán này.
- Kết luận chính: MiniFASNet cho kết quả ổn định và vượt trội rõ rệt trên cả 2 dataset; ViT-FAS hiện tại thấp hơn đáng kể trên cùng pipeline đánh giá.
- Đề xuất tích hợp service: ưu tiên MiniFASNet làm baseline production, tiếp tục kiểm chứng/căn chỉnh ViT-FAS trước khi đưa vào luồng chính.

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
| | | |
| | | |

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

### 4.2 Số lượng ảnh (sample / test)

| Dataset | Tổng | Live | Spoof | Unknown / lỗi |
|---------|------|------|-------|----------------|
| celeba_spoof | 4000 | 2000 | 2000 | 0 |
| casia_fasd | 4063 | 995 | 3068 | 0 |

*(Đường dẫn CSV: `data/sampled/`, `data/raw/.../annotations/raw.csv`.)*

### 4.3 Đặc điểm tiền xử lý dữ liệu đầu vào

- CelebA-Spoof (`cropped_image`): không cần tiền xử lý gì thêm
- CASIA-FASD (`cropped_image` 256×256, v.v.): không cần tiền xử lý gì thêm
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
- File metrics:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/metrics/celeba_spoof/`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/metrics/celeba_spoof/`

| Model | Threshold | Accuracy | APCER | BPCER | ACER | Ghi chú |
|-------|-----------|----------|-------|-------|------|---------|
| MiniFASNet | 0.3 | 0.7060 | 0.3405 | 0.2475 | 0.2940 | Cân bằng khá tốt |
| MiniFASNet | 0.5 | 0.6883 | 0.2685 | 0.3550 | 0.3118 | Điểm vận hành trung tính |
| MiniFASNet | 0.7 | 0.6670 | 0.1835 | 0.4825 | 0.3330 | Giảm APCER, tăng BPCER |
| ViT-FAS | 0.3 | 0.6685 | 0.2255 | 0.4375 | 0.3315 | BPCER cao |
| ViT-FAS | 0.5 | 0.6443 | 0.1830 | 0.5285 | 0.3558 | Kém MiniFASNet |
| ViT-FAS | 0.7 | 0.6295 | 0.1340 | 0.6070 | 0.3705 | BPCER rất cao |

- Confusion matrix: `confusion_matrix_*.png`
- MiniFASNet (threshold 0.5):
![MiniFASNet CelebA Confusion Matrix](models/minifasnet_v2_2p7/metrics/celeba_spoof/confusion_matrix_0.5.png)

- ViT-FAS (threshold 0.5):
![ViT-FAS CelebA Confusion Matrix](models/vitfas_vitb16_224/metrics/celeba_spoof/confusion_matrix_0.5.png)

- Nhận xét ngắn: MiniFASNet ổn định hơn trên CelebA-Spoof; ViT-FAS có xu hướng reject nhiều live khi tăng threshold.

### 7.2 CASIA-FASD

- Điều kiện chạy: sample CASIA-FASD theo annotation hiện có.
- File predictions:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/predictions/casia_fasd/latest.csv`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/predictions/casia_fasd/latest.csv`
- File metrics:
  - MiniFASNet: `reports/models/minifasnet_v2_2p7/metrics/casia_fasd/`
  - ViT-FAS: `reports/models/vitfas_vitb16_224/metrics/casia_fasd/`

| Model | Threshold | Accuracy | APCER | BPCER | ACER | Ghi chú |
|-------|-----------|----------|-------|-------|------|---------|
| MiniFASNet | 0.3 | 0.9488 | 0.0290 | 0.1196 | 0.0743 | Rất tốt |
| MiniFASNet | 0.5 | 0.9527 | 0.0130 | 0.1528 | 0.0829 | Điểm vận hành đề xuất |
| MiniFASNet | 0.7 | 0.9488 | 0.0052 | 0.1930 | 0.0991 | Ưu tiên giảm APCER |
| ViT-FAS | 0.3 | 0.5843 | 0.4977 | 0.1628 | 0.3303 | APCER cao |
| ViT-FAS | 0.5 | 0.6215 | 0.4234 | 0.2402 | 0.3318 | Vẫn kém xa |
| ViT-FAS | 0.7 | 0.6520 | 0.3504 | 0.3407 | 0.3455 | Không đạt mức deploy |

- Confusion matrix:
- MiniFASNet (threshold 0.5):
![MiniFASNet CASIA Confusion Matrix](models/minifasnet_v2_2p7/metrics/casia_fasd/confusion_matrix_0.5.png)

- ViT-FAS (threshold 0.5):
![ViT-FAS CASIA Confusion Matrix](models/vitfas_vitb16_224/metrics/casia_fasd/confusion_matrix_0.5.png)

- Nhận xét ngắn: trên CASIA-FASD, MiniFASNet vượt trội toàn diện; ViT-FAS có tỷ lệ spoof lọt (APCER) còn cao.

### 7.3 So sánh giữa hai dataset

| Tiêu chí | CelebA-Spoof | CASIA-FASD |
|----------|--------------|------------|
| MiniFASNet ACER @ 0.5 | 0.3118 | 0.0829 |
| ViT-FAS ACER @ 0.5 | 0.3558 | 0.3318 |
| MiniFASNet APCER @ 0.5 | 0.2685 | 0.0130 |
| ViT-FAS APCER @ 0.5 | 0.1830 | 0.4234 |
| Độ khó / domain | CelebA cân bằng sample, phân tách khó hơn | CASIA sample cho MiniFASNet phân tách tốt |
| Preprocess đầu vào | resize + normalize theo wrapper | resize + normalize theo wrapper |

### 7.4 Ảnh minh họa (tuỳ chọn)

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

- MiniFASNet (`minifasnet_v2_2p7`) hiện là mô hình tốt nhất trong pipeline hiện tại trên cả CelebA-Spoof và CASIA-FASD.
- ViT-FAS đã chạy end-to-end thành công nhưng quality còn thấp, cần tiếp tục kiểm chứng tương thích kiến trúc/checkpoint và tối ưu thêm trước khi dùng production.

### 10.2 Đề xuất tích hợp service

| Hạng mục | Đề xuất |
|----------|---------|
| Model anti-spoofing | MiniFASNetV2 làm baseline tích hợp |
| Ngưỡng operating point | Bắt đầu với `0.5`, tune theo mục tiêu APCER/BPCER thực tế |
| Preprocessing (detect + crop) | Giữ pipeline crop/resize hiện tại, chuẩn hóa đầu vào nhất quán train-test |
| Bước tiếp theo (fine-tune, data nội bộ) | Thu thập thêm data nội bộ + calibrate threshold theo rủi ro bảo mật |

### 10.3 Công việc tiếp theo

- [ ] Kiểm tra sâu nhánh load/checkpoint ViT-FAS và thống nhất class order live/spoof.
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
| | v0.1 | Khởi tạo mẫu |
