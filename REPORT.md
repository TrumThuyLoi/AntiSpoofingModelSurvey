# Báo cáo đánh giá Anti-Spoofing (MiniFASNet vs ViT-FAS vs FaceAntispoof-ONNX)

> Mẫu báo cáo — điền nội dung vào từng mục. Cập nhật ngày / phiên bản khi hoàn thiện.

| | |
|---|---|
| **Tác giả** |Vũ Hồng Quang|
| **Ngày** |01/06/2026|
| **Repo / branch** |main|
| **Môi trường** | Python, OS, GPU/CPU |

---

## 1. Tóm tắt (Executive summary)

- Mục tiêu: đánh giá stage anti-spoofing cho bài toán verify ảnh tài xế trước khi tích hợp service.
- Model đã đánh giá: **MiniFASNetV2** (`minifasnet_v2_2p7`), **ViT-FAS** (`vitfas_vitb16_224`) và **FaceAntispoof-ONNX** (`face_antispoof_onnx_9820`).
- Dataset đã dùng: ma trận **3 model × 10 dataset** (CelebA, CASIA, Face VN, drivers × 7 SFAS expansion); benchmark `run_inference.py --all` + `run_evaluation.py --all`.
- Kết luận chính: MiniFASNet vẫn là model tốt nhất trên CASIA-FASD (BPCER/APCER thấp @0.5). FaceAntispoof-ONNX cạnh tranh trên CelebA-Spoof (BPCER thấp) và xếp thứ hai trên CASIA-FASD. Trên **face_antispoofing_vn** (11376 mẫu), trade-off rõ khi tăng SFAS expansion (BPCER giảm, APCER tăng). Ablation drivers (393 live, SFAS **1.6 / 2.7**): **MiniFASNet @ exp 1.6** tốt nhất — **BPCER 0.059 (23/393)** @0.5; exp 2.7: 0.061 (24/393). Dưới gate BPCER ≤ 0.10; cần validation spoof trước rollout.
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
| MiniFASNet (Silent-Face-Anti-Spoofing) | [minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) · weight: [2.7_80x80_MiniFASNetV2.pth](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth) | Submodule `third_party/Silent-Face-Anti-Spoofing`; wrapper `configs/model_minifasnet.yaml`, input 80×80 |
| ViT-FAS (vit-spoof-detection-pda) | [ArchitRastogi20/vit-spoof-detection-pda](https://github.com/ArchitRastogi20/vit-spoof-detection-pda) · weight: [HF `best_model_run_eif1jakb.pth`](https://huggingface.co/ArchitRastogi/vit-spoof-detection-pda) | Submodule `third_party/vit-spoof-detection-pda`; backbone ViT-B/16; `configs/model_vit_fas.yaml`, input 224×224 |
| Face Anti-Spoof ONNX | [facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) · weight: [best_model.pth (98.20%)](https://github.com/facenox/face-antispoof-onnx/blob/main/models/best/98.20/best_model.pth) | Submodule `third_party/face-antispoof-onnx`; MiniFASNetV2SE; `configs/model_face_antispoof_onnx.yaml`, input 128×128 |

### 3.2 Tiêu chí so sánh

- Độ chính xác; **APCER** / **BPCER** trên tập test
- Tốc độ inference, yêu cầu GPU
- Dễ deploy (Docker, weight, license)
- Khả năng mở rộng (fine-tune, API)

### 3.3 Model được chọn cho benchmark

| Model | `model_id` | Weight (local) | Config |
|-------|------------|----------------|--------|
| MiniFASNetV2 | `minifasnet_v2_2p7` | `models/2.7_80x80_MiniFASNetV2.pth` | `configs/model_minifasnet.yaml` |
| ViT-FAS | `vitfas_vitb16_224` | `models/vitfas_vitb16_224x224.pth` | `configs/model_vit_fas.yaml` |
| FaceAntispoof-ONNX | `face_antispoof_onnx_9820` | `models/face_antispoof_onnx_best_9820.pth` | `configs/model_face_antispoof_onnx.yaml` |

Tải weight: `python3 scripts/download_pretrained_weights.py`. Submodule: `git submodule update --init third_party/Silent-Face-Anti-Spoofing third_party/vit-spoof-detection-pda third_party/face-antispoof-onnx`.

---

## 4. Dataset

### 4.1 Dataset sử dụng

| Dataset | Nguồn (HF / local) | Ghi chú |
|---------|-------------------|---------|
| CelebA-Spoof | [nguyenkhoa/celeba-spoof-for-face-antispoofing-test](https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test) | Chủ yếu người phương Tây; benchmark dùng sample 4k (`celeba_spoof_sample.csv`), cột `cropped_image` |
| CASIA-FASD | [kaggle/casia_fasd](https://www.kaggle.com/datasets/minhnh2107/casiafasd) | Chủ yếu người Trung Quốc; split `test`; publish HF: `scripts/create_casia_fasd_hf_dataset.py` |
| Face Anti-Spoofing VN | [kaggle/face_antispoofing_vn](https://www.kaggle.com/datasets/maihongtng/face-anti-spoofing?resource=download&select=Image) | Ảnh tài xế VN; crop SFAS offline; ablation báo cáo exp **1.6**, **2.7** |
| Drivers 250 FN | Local: `data/drivers_250_FN/` → crop (`drivers_250_fn_cropped*`) | Tập FN nội bộ; 393 live; pipeline crop **1.6, 2.7** + baseline **1.0** riêng; `drivers_250_fn_sfas_crop_and_evaluation.md` |

### 4.2 Số lượng ảnh (sample / test)

| Dataset | Tổng | Live | Spoof | Unknown / lỗi |
|---------|------|------|-------|----------------|
| celeba_spoof | 4000 | 2000 | 2000 | 0 |
| casia_fasd | 4063 | 995 | 3068 | 0 |
| face_antispoofing_vn | 11376 | 7541 | 3835 | 0 |
| drivers_250_fn | 393 | 393 | 0 | 0 |

*(CSV: `data/sampled/*_sample.csv`; HF raw: `data/raw/<dataset>/annotations/raw.csv`; spec repo HF: `src/datasets/specs.py`.)*

### 4.3 Đặc điểm tiền xử lý dữ liệu đầu vào

- CelebA-Spoof (`cropped_image`): không cần tiền xử lý gì thêm
- CASIA-FASD (`cropped_image` 256×256, v.v.): không cần tiền xử lý gì thêm
- Face Anti-Spoofing VN: crop SFAS từ `data/face_antispoofing_vn/` (giống drivers); inference resize theo model (80×80 / 128×128 / 224×224)
- Crop offline CASIA (`data/processed/casia_fasd/`): có / không — ghi chú

---

## 7. Kết quả

Ngưỡng báo cáo: **0.3, 0.5, 0.7** (`configs/evaluation.yaml`). Face VN / CelebA / CASIA: `metrics_summary.csv`; drivers: `predictions/latest.csv` (393 live / expansion).

### 7.0 Chú thích chỉ số đo lường

Quy ước trong repo (`scripts/run_evaluation.py`): ảnh có nhãn thật **live** hoặc **spoof**; model xuất **`live_score`** ∈ [0, 1] (càng cao càng nghiêng về live). Với ngưỡng **Threshold** *t*: dự đoán **live** nếu `live_score ≥ t`, ngược lại **spoof**.

| Chỉ số / thuật ngữ | Ý nghĩa | Công thức (theo confusion matrix) | Ghi chú vận hành |
|-------------------|---------|-----------------------------------|------------------|
| **Threshold** | Ngưỡng cắt `live_score` | — | Báo cáo dùng **0.3, 0.5, 0.7**; đường cong APCER/BPCER vẽ dày hơn (0.00–1.00). |
| **live_score** | Điểm “độ live” do model | — | Biểu đồ phân phối: tách theo nhãn thật live vs spoof. |
| **Accuracy** | Tỷ lệ dự đoán đúng (cả hai lớp) | (TP + TN) / tổng mẫu | Dễ lệch nếu mất cân bằng lớp; không thay BPCER/APCER trên production. |
| **APCER** | Attack Presentation Classification Error Rate | FP / (FP + TN) = spoof bị gán **live** | **Spoof lọt** (false accept). Càng **thấp** càng chặt spoof. |
| **BPCER** | Bona Fide Presentation Classification Error Rate | FN / (TP + FN) = live bị gán **spoof** | **Live bị reject** (false reject). Metric **ưu tiên** trên `drivers_250_fn` (toàn live). |
| **TP** | True Positive (live→live) | Live đúng, pass | Trên drivers: số live được chấp nhận. |
| **FN** | False Negative (live→spoof) | Live sai, reject | = BPCER × số live; ví dụ BPCER 0.059, 393 live → ~23 FN. |
| **FP** | False Positive (spoof→live) | Spoof sai, pass | = APCER × số spoof; trên drivers (0 spoof) → APCER = 0. |
| **TN** | True Negative (spoof→spoof) | Spoof đúng, reject | |

**Đọc biểu đồ:**
- `apcer_bpcer_vs_threshold.png`: đổi **Threshold** → trade-off APCER ↔ BPCER.
- `live_score_distribution.png`: live thật nên dồn **cao**; spoof thật nên dồn **thấp** (tách càng rõ càng tốt).
- `confusion_matrix_0.5.png`: ma trận 2×2 @ ngưỡng **0.5** (trục: true live/spoof × pred live/spoof).

**Định dạng bảng §7.1–7.4:** mỗi ô Accuracy / APCER / BPCER = **`tỷ lệ (số_lỗi/tổng_lớp)`** (vd. BPCER `0.059 (23/393)` = 23 live bị reject trên 393 live).

**Production (verify tài xế):** gate đề xuất **BPCER ≤ 0.10** trên tập live thực tế; kiểm tra thêm APCER khi có tập spoof.

### 7.1 CelebA-Spoof

- Predictions / metrics: `reports/models/<model_id>/predictions|metrics/celeba_spoof/`
- @0.5: ONNX APCER/BPCER cân bằng; ViT BPCER cao `0.5285 (1057/2000)`.

| Model | Threshold | Accuracy | APCER | BPCER |
|-------|-----------|----------|-------|-------|
| MiniFASNet | 0.3 | 0.7060 (2824/4000) | 0.3405 (681/2000) | 0.2475 (495/2000) |
| MiniFASNet | 0.5 | 0.6883 (2753/4000) | 0.2685 (537/2000) | 0.3550 (710/2000) |
| MiniFASNet | 0.7 | 0.6667 (2667/4000) | 0.1835 (367/2000) | 0.4830 (966/2000) |
| ViT-FAS | 0.3 | 0.6685 (2674/4000) | 0.2255 (451/2000) | 0.4375 (875/2000) |
| ViT-FAS | 0.5 | 0.6442 (2577/4000) | 0.1830 (366/2000) | 0.5285 (1057/2000) |
| ViT-FAS | 0.7 | 0.6295 (2518/4000) | 0.1340 (268/2000) | 0.6070 (1214/2000) |
| ONNX | 0.3 | 0.8365 (3346/4000) | 0.2490 (498/2000) | 0.0780 (156/2000) |
| ONNX | 0.5 | 0.8415 (3366/4000) | 0.1910 (382/2000) | 0.1260 (252/2000) |
| ONNX | 0.7 | 0.8415 (3366/4000) | 0.1380 (276/2000) | 0.1790 (358/2000) |

Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet APCER/BPCER](./reports/models/minifasnet_v2_2p7/metrics/celeba_spoof/apcer_bpcer_vs_threshold.png) | ![ViT-FAS APCER/BPCER](./reports/models/vitfas_vitb16_224/metrics/celeba_spoof/apcer_bpcer_vs_threshold.png) | ![ONNX APCER/BPCER](./reports/models/face_antispoof_onnx_9820/metrics/celeba_spoof/apcer_bpcer_vs_threshold.png) |

Phân phối `live_score` (`live_score_distribution.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet live_score](./reports/models/minifasnet_v2_2p7/metrics/celeba_spoof/live_score_distribution.png) | ![ViT-FAS live_score](./reports/models/vitfas_vitb16_224/metrics/celeba_spoof/live_score_distribution.png) | ![ONNX live_score](./reports/models/face_antispoof_onnx_9820/metrics/celeba_spoof/live_score_distribution.png) |

Confusion matrix @0.5 (`confusion_matrix_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet CM@0.5](./reports/models/minifasnet_v2_2p7/metrics/celeba_spoof/confusion_matrix_0.5.png) | ![ViT-FAS CM@0.5](./reports/models/vitfas_vitb16_224/metrics/celeba_spoof/confusion_matrix_0.5.png) | ![ONNX CM@0.5](./reports/models/face_antispoof_onnx_9820/metrics/celeba_spoof/confusion_matrix_0.5.png) |

### 7.2 CASIA-FASD

- Predictions / metrics: `reports/models/<model_id>/predictions|casia_fasd/`
- @0.5: MiniFASNet BPCER thấp `0.1528 (152/995)`; APCER rất thấp.

| Model | Threshold | Accuracy | APCER | BPCER |
|-------|-----------|----------|-------|-------|
| MiniFASNet | 0.3 | 0.9488 (3855/4063) | 0.0290 (89/3068) | 0.1196 (119/995) |
| MiniFASNet | 0.5 | 0.9527 (3871/4063) | 0.0130 (40/3068) | 0.1528 (152/995) |
| MiniFASNet | 0.7 | 0.9488 (3855/4063) | 0.0052 (16/3068) | 0.1930 (192/995) |
| ViT-FAS | 0.3 | 0.5843 (2374/4063) | 0.4977 (1527/3068) | 0.1628 (162/995) |
| ViT-FAS | 0.5 | 0.6215 (2525/4063) | 0.4234 (1299/3068) | 0.2402 (239/995) |
| ViT-FAS | 0.7 | 0.6520 (2649/4063) | 0.3504 (1075/3068) | 0.3407 (339/995) |
| ONNX | 0.3 | 0.6638 (2697/4063) | 0.4195 (1287/3068) | 0.0794 (79/995) |
| ONNX | 0.5 | 0.7061 (2869/4063) | 0.3556 (1091/3068) | 0.1035 (103/995) |
| ONNX | 0.7 | 0.7413 (3012/4063) | 0.2963 (909/3068) | 0.1427 (142/995) |

Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet APCER/BPCER](./reports/models/minifasnet_v2_2p7/metrics/casia_fasd/apcer_bpcer_vs_threshold.png) | ![ViT-FAS APCER/BPCER](./reports/models/vitfas_vitb16_224/metrics/casia_fasd/apcer_bpcer_vs_threshold.png) | ![ONNX APCER/BPCER](./reports/models/face_antispoof_onnx_9820/metrics/casia_fasd/apcer_bpcer_vs_threshold.png) |

Phân phối `live_score` (`live_score_distribution.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet live_score](./reports/models/minifasnet_v2_2p7/metrics/casia_fasd/live_score_distribution.png) | ![ViT-FAS live_score](./reports/models/vitfas_vitb16_224/metrics/casia_fasd/live_score_distribution.png) | ![ONNX live_score](./reports/models/face_antispoof_onnx_9820/metrics/casia_fasd/live_score_distribution.png) |

Confusion matrix @0.5 (`confusion_matrix_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet CM@0.5](./reports/models/minifasnet_v2_2p7/metrics/casia_fasd/confusion_matrix_0.5.png) | ![ViT-FAS CM@0.5](./reports/models/vitfas_vitb16_224/metrics/casia_fasd/confusion_matrix_0.5.png) | ![ONNX CM@0.5](./reports/models/face_antispoof_onnx_9820/metrics/casia_fasd/confusion_matrix_0.5.png) |

### 7.3 Face Anti-Spoofing VN (SFAS expansion)

- **11376** mẫu (7541 live, 3835 spoof). Ablation crop SFAS: exp **1.6**, **2.7**.

#### SFAS exp 1.6 (`face_antispoofing_vn_exp1.6`)

| Model | Threshold | Accuracy | APCER | BPCER |
|-------|-----------|----------|-------|-------|
| MiniFASNet | 0.3 | 0.7772 (8841/11376) | 0.6138 (2354/3835) | 0.0240 (181/7541) |
| MiniFASNet | 0.5 | 0.7948 (9042/11376) | 0.5419 (2078/3835) | 0.0339 (256/7541) |
| MiniFASNet | 0.7 | 0.8080 (9192/11376) | 0.4754 (1823/3835) | 0.0479 (361/7541) |
| ViT-FAS | 0.3 | 0.6784 (7718/11376) | 0.6847 (2626/3835) | 0.1369 (1032/7541) |
| ViT-FAS | 0.5 | 0.6729 (7655/11376) | 0.6136 (2353/3835) | 0.1814 (1368/7541) |
| ViT-FAS | 0.7 | 0.6612 (7522/11376) | 0.5387 (2066/3835) | 0.2371 (1788/7541) |
| ONNX | 0.3 | 0.7890 (8976/11376) | 0.4579 (1756/3835) | 0.0854 (644/7541) |
| ONNX | 0.5 | 0.7900 (8987/11376) | 0.3948 (1514/3835) | 0.1160 (875/7541) |
| ONNX | 0.7 | 0.7885 (8970/11376) | 0.3218 (1234/3835) | 0.1554 (1172/7541) |

Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet APCER/BPCER](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.6/apcer_bpcer_vs_threshold.png) | ![ViT-FAS APCER/BPCER](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp1.6/apcer_bpcer_vs_threshold.png) | ![ONNX APCER/BPCER](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.6/apcer_bpcer_vs_threshold.png) |

Phân phối `live_score` (`live_score_distribution.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet live_score](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.6/live_score_distribution.png) | ![ViT-FAS live_score](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp1.6/live_score_distribution.png) | ![ONNX live_score](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.6/live_score_distribution.png) |

Confusion matrix @0.5 (`confusion_matrix_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet CM@0.5](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.6/confusion_matrix_0.5.png) | ![ViT-FAS CM@0.5](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp1.6/confusion_matrix_0.5.png) | ![ONNX CM@0.5](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.6/confusion_matrix_0.5.png) |

Âm tính giả / FN @0.5 (`false_negative_montage_0.5.png`, 6×4):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet FN](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.6/false_negative_montage_0.5.png) | ![ViT-FAS FN](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp1.6/false_negative_montage_0.5.png) | ![ONNX FN](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.6/false_negative_montage_0.5.png) |

Dương tính giả / FP @0.5 (`false_positive_montage_0.5.png`, 6×4):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet FP](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.6/false_positive_montage_0.5.png) | ![ViT-FAS FP](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp1.6/false_positive_montage_0.5.png) | ![ONNX FP](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.6/false_positive_montage_0.5.png) |

#### SFAS exp 2.7 (`face_antispoofing_vn_exp2.7`)

| Model | Threshold | Accuracy | APCER | BPCER |
|-------|-----------|----------|-------|-------|
| MiniFASNet | 0.3 | 0.7867 (8949/11376) | 0.5784 (2218/3835) | 0.0277 (209/7541) |
| MiniFASNet | 0.5 | 0.8008 (9110/11376) | 0.5126 (1966/3835) | 0.0398 (300/7541) |
| MiniFASNet | 0.7 | 0.8163 (9286/11376) | 0.4396 (1686/3835) | 0.0536 (404/7541) |
| ViT-FAS | 0.3 | 0.6729 (7655/11376) | 0.6668 (2557/3835) | 0.1544 (1164/7541) |
| ViT-FAS | 0.5 | 0.6676 (7595/11376) | 0.5914 (2268/3835) | 0.2006 (1513/7541) |
| ViT-FAS | 0.7 | 0.6499 (7393/11376) | 0.5186 (1989/3835) | 0.2644 (1994/7541) |
| ONNX | 0.3 | 0.7397 (8415/11376) | 0.4274 (1639/3835) | 0.1753 (1322/7541) |
| ONNX | 0.5 | 0.7319 (8326/11376) | 0.3622 (1389/3835) | 0.2203 (1661/7541) |
| ONNX | 0.7 | 0.7213 (8205/11376) | 0.3040 (1166/3835) | 0.2659 (2005/7541) |

Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet APCER/BPCER](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp2.7/apcer_bpcer_vs_threshold.png) | ![ViT-FAS APCER/BPCER](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp2.7/apcer_bpcer_vs_threshold.png) | ![ONNX APCER/BPCER](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp2.7/apcer_bpcer_vs_threshold.png) |

Phân phối `live_score` (`live_score_distribution.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet live_score](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp2.7/live_score_distribution.png) | ![ViT-FAS live_score](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp2.7/live_score_distribution.png) | ![ONNX live_score](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp2.7/live_score_distribution.png) |

Confusion matrix @0.5 (`confusion_matrix_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet CM@0.5](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp2.7/confusion_matrix_0.5.png) | ![ViT-FAS CM@0.5](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp2.7/confusion_matrix_0.5.png) | ![ONNX CM@0.5](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp2.7/confusion_matrix_0.5.png) |

Âm tính giả / FN @0.5 (`false_negative_montage_0.5.png`, 6×4):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet FN](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp2.7/false_negative_montage_0.5.png) | ![ViT-FAS FN](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp2.7/false_negative_montage_0.5.png) | ![ONNX FN](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp2.7/false_negative_montage_0.5.png) |

Dương tính giả / FP @0.5 (`false_positive_montage_0.5.png`, 6×4):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet FP](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp2.7/false_positive_montage_0.5.png) | ![ViT-FAS FP](./reports/models/vitfas_vitb16_224/metrics/face_antispoofing_vn_exp2.7/false_positive_montage_0.5.png) | ![ONNX FP](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp2.7/false_positive_montage_0.5.png) |

- Nhận xét: exp 1.6/2.7 — BPCER giảm mạnh, APCER tăng (trade-off spoof vs live).

### 7.4 Drivers 250 FN (SFAS expansion)

- **393** live / expansion; **APCER = 0**. Pipeline crop SFAS: **1.6**, **2.7**.

#### SFAS exp 1.6 (`drivers_250_fn_exp1.6`)

| Model | Threshold | Accuracy | APCER | BPCER |
|-------|-----------|----------|-------|-------|
| MiniFASNet | 0.3 | 0.9593 (377/393) | 0.0000 (—) | 0.0407 (16/393) |
| MiniFASNet | 0.5 | 0.9415 (370/393) | 0.0000 (—) | 0.0585 (23/393) |
| MiniFASNet | 0.7 | 0.9288 (365/393) | 0.0000 (—) | 0.0712 (28/393) |
| ViT-FAS | 0.3 | 0.8702 (342/393) | 0.0000 (—) | 0.1298 (51/393) |
| ViT-FAS | 0.5 | 0.8244 (324/393) | 0.0000 (—) | 0.1756 (69/393) |
| ViT-FAS | 0.7 | 0.7710 (303/393) | 0.0000 (—) | 0.2290 (90/393) |
| ONNX | 0.3 | 0.8550 (336/393) | 0.0000 (—) | 0.1450 (57/393) |
| ONNX | 0.5 | 0.8041 (316/393) | 0.0000 (—) | 0.1959 (77/393) |
| ONNX | 0.7 | 0.7659 (301/393) | 0.0000 (—) | 0.2341 (92/393) |

Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet APCER/BPCER](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.6/apcer_bpcer_vs_threshold.png) | ![ViT-FAS APCER/BPCER](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp1.6/apcer_bpcer_vs_threshold.png) | ![ONNX APCER/BPCER](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.6/apcer_bpcer_vs_threshold.png) |

Phân phối `live_score` (`live_score_distribution.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet live_score](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.6/live_score_distribution.png) | ![ViT-FAS live_score](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp1.6/live_score_distribution.png) | ![ONNX live_score](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.6/live_score_distribution.png) |

Confusion matrix @0.5 (`confusion_matrix_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet CM@0.5](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.6/confusion_matrix_0.5.png) | ![ViT-FAS CM@0.5](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp1.6/confusion_matrix_0.5.png) | ![ONNX CM@0.5](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.6/confusion_matrix_0.5.png) |

Âm tính giả / FN @0.5 (`false_negative_montage_0.5.png`, 6×4):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet FN](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.6/false_negative_montage_0.5.png) | ![ViT-FAS FN](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp1.6/false_negative_montage_0.5.png) | ![ONNX FN](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.6/false_negative_montage_0.5.png) |

Dương tính giả / FP @0.5 (`false_positive_montage_0.5.png`, 6×4):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| *(MiniFASNet: chưa có `false_positive_montage_0.5.png`)* | *(ViT-FAS: chưa có `false_positive_montage_0.5.png`)* | *(ONNX: chưa có `false_positive_montage_0.5.png`)* |

#### SFAS exp 2.7 (`drivers_250_fn_exp2.7`)

| Model | Threshold | Accuracy | APCER | BPCER |
|-------|-----------|----------|-------|-------|
| MiniFASNet | 0.3 | 0.9644 (379/393) | 0.0000 (—) | 0.0356 (14/393) |
| MiniFASNet | 0.5 | 0.9389 (369/393) | 0.0000 (—) | 0.0611 (24/393) |
| MiniFASNet | 0.7 | 0.9288 (365/393) | 0.0000 (—) | 0.0712 (28/393) |
| ViT-FAS | 0.3 | 0.8626 (339/393) | 0.0000 (—) | 0.1374 (54/393) |
| ViT-FAS | 0.5 | 0.8244 (324/393) | 0.0000 (—) | 0.1756 (69/393) |
| ViT-FAS | 0.7 | 0.7659 (301/393) | 0.0000 (—) | 0.2341 (92/393) |
| ONNX | 0.3 | 0.8092 (318/393) | 0.0000 (—) | 0.1908 (75/393) |
| ONNX | 0.5 | 0.7506 (295/393) | 0.0000 (—) | 0.2494 (98/393) |
| ONNX | 0.7 | 0.6947 (273/393) | 0.0000 (—) | 0.3053 (120/393) |

Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet APCER/BPCER](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp2.7/apcer_bpcer_vs_threshold.png) | ![ViT-FAS APCER/BPCER](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp2.7/apcer_bpcer_vs_threshold.png) | ![ONNX APCER/BPCER](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp2.7/apcer_bpcer_vs_threshold.png) |

Phân phối `live_score` (`live_score_distribution.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet live_score](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp2.7/live_score_distribution.png) | ![ViT-FAS live_score](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp2.7/live_score_distribution.png) | ![ONNX live_score](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp2.7/live_score_distribution.png) |

Confusion matrix @0.5 (`confusion_matrix_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet CM@0.5](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp2.7/confusion_matrix_0.5.png) | ![ViT-FAS CM@0.5](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp2.7/confusion_matrix_0.5.png) | ![ONNX CM@0.5](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp2.7/confusion_matrix_0.5.png) |

Âm tính giả / FN @0.5 (`false_negative_montage_0.5.png`, 6×4):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet FN](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp2.7/false_negative_montage_0.5.png) | ![ViT-FAS FN](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp2.7/false_negative_montage_0.5.png) | ![ONNX FN](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp2.7/false_negative_montage_0.5.png) |

Dương tính giả / FP @0.5 (`false_positive_montage_0.5.png`, 6×4):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| *(MiniFASNet: chưa có `false_positive_montage_0.5.png`)* | *(ViT-FAS: chưa có `false_positive_montage_0.5.png`)* | *(ONNX: chưa có `false_positive_montage_0.5.png`)* |

- **Xu hướng (MiniFASNet @0.5):** BPCER **0.0585 (23/393)** @ exp 1.6; 0.0611 (24/393) @ exp 2.7.

### 7.5 Ma trận benchmark — tóm tắt @0.5

`run_inference.py --all` + `run_evaluation.py --all`. Bảng đầy đủ threshold: §7.1–7.4.

Định dạng mỗi ô: **BPCER (FN/n_live) / APCER (FP/n_spoof)**.

| Dataset | MiniFASNet BPCER / APCER | ViT-FAS | ONNX |
|---------|-------------------------|---------|------|
| celeba_spoof | 0.3550 (710/2000) / 0.2685 (537/2000) | 0.5285 (1057/2000) / 0.1830 (366/2000) | 0.1260 (252/2000) / 0.1910 (382/2000) |
| casia_fasd | 0.1528 (152/995) / 0.0130 (40/3068) | 0.2402 (239/995) / 0.4234 (1299/3068) | 0.1035 (103/995) / 0.3556 (1091/3068) |
| face_antispoofing_vn_exp1.6 | 0.0339 (256/7541) / 0.5419 (2078/3835) | 0.1814 (1368/7541) / 0.6136 (2353/3835) | 0.1160 (875/7541) / 0.3948 (1514/3835) |
| face_antispoofing_vn_exp2.7 | 0.0398 (300/7541) / 0.5126 (1966/3835) | 0.2006 (1513/7541) / 0.5914 (2268/3835) | 0.2203 (1661/7541) / 0.3622 (1389/3835) |
| drivers_250_fn_exp1.6 | 0.0585 (23/393) / 0.0000 (—) | 0.1756 (69/393) / 0.0000 (—) | 0.1959 (77/393) / 0.0000 (—) |
| drivers_250_fn_exp2.7 | 0.0611 (24/393) / 0.0000 (—) | 0.1756 (69/393) / 0.0000 (—) | 0.2494 (98/393) / 0.0000 (—) |

#### 7.5.1 Dataset × dự đoán @0.5 (`True` / `False`)

Nguồn: `reports/models/<model_id>/predictions/<dataset>/latest.csv`, ngưỡng **0.5**.

- **True** = số ảnh **live** được model chấp nhận (pred live → TP trên live).
- **False** = số ảnh **live** bị reject (pred spoof → FN trên live).
- Định dạng ô: **`True / False`** (chỉ đếm mẫu nhãn live; spoof không hiển thị ở đây).

**Phạm vi dataset** (khớp §7.5 và §7.1–7.4):
- **CelebA / CASIA:** benchmark chuẩn (không SFAS expansion).
- **Face VN / Drivers:** chỉ crop SFAS **exp 1.6** và **2.7** (không exp 1.0 trong báo cáo).

| Dataset | MiniFASNet | ViT-FAS | ONNX |
|---------|------------|---------|------|
| celeba_spoof (2000 live, 4000 mẫu) | 1290 / 710 | 943 / 1057 | 1748 / 252 |
| casia_fasd (995 live, 4063 mẫu) | 843 / 152 | 756 / 239 | 892 / 103 |
| face_antispoofing_vn_exp1.6 (7541 live, 11376 mẫu) | 7285 / 256 | 6173 / 1368 | 6666 / 875 |
| face_antispoofing_vn_exp2.7 (7541 live, 11376 mẫu) | 7241 / 300 | 6028 / 1513 | 5880 / 1661 |
| drivers_250_fn_exp1.6 (393 live) | 370 / 23 | 324 / 69 | 316 / 77 |
| drivers_250_fn_exp2.7 (393 live) | 369 / 24 | 324 / 69 | 295 / 98 |

#### 7.5.2 Phân phối `y_prob` @0.5 (nhóm True / False)

`live_score` = **y_prob**; ngưỡng **0.5**. Chỉ **drivers** SFAS **exp 1.6** và **2.7** (393 live).

- **True** = live được chấp nhận (TP); **False** = live bị reject (FN).
- Artifact: `live_score_true_false_0.5.png`, `live_score_histogram_bins_0.5.png` (`run_evaluation.py`, chỉ `drivers_250_fn*`).

**Drivers exp 1.6** (`drivers_250_fn_exp1.6`, 393 live):

Phân phối theo nhóm True / False @0.5 (`live_score_true_false_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet True/False](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.6/live_score_true_false_0.5.png) | ![ViT-FAS True/False](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp1.6/live_score_true_false_0.5.png) | ![ONNX True/False](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.6/live_score_true_false_0.5.png) |

Histogram toàn tập live (step **0.1**, `live_score_histogram_bins_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet histogram](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.6/live_score_histogram_bins_0.5.png) | ![ViT-FAS histogram](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp1.6/live_score_histogram_bins_0.5.png) | ![ONNX histogram](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.6/live_score_histogram_bins_0.5.png) |

**Drivers exp 2.7** (`drivers_250_fn_exp2.7`, 393 live):

Phân phối theo nhóm True / False @0.5 (`live_score_true_false_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet True/False](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp2.7/live_score_true_false_0.5.png) | ![ViT-FAS True/False](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp2.7/live_score_true_false_0.5.png) | ![ONNX True/False](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp2.7/live_score_true_false_0.5.png) |

Histogram toàn tập live (step **0.1**, `live_score_histogram_bins_0.5.png`):

| MiniFASNet | ViT-FAS | ONNX |
|:---:|:---:|:---:|
| ![MiniFASNet histogram](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp2.7/live_score_histogram_bins_0.5.png) | ![ViT-FAS histogram](./reports/models/vitfas_vitb16_224/metrics/drivers_250_fn_exp2.7/live_score_histogram_bins_0.5.png) | ![ONNX histogram](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp2.7/live_score_histogram_bins_0.5.png) |

---

## 10. Kết luận và đề xuất

*Thuật ngữ dùng trong mục này (chi tiết §7.0):*

| Thuật ngữ | Giải thích ngắn |
|-----------|-----------------|
| **live** | Ảnh người thật (tài xế thật). |
| **spoof** | Ảnh giả / tấn công (màn hình, in ảnh, …). |
| **live_score** | Điểm 0→1 do model trả về; càng cao model càng “tin” là live. |
| **Ngưỡng (threshold)** | Cắt `live_score`: từ ngưỡng trở lên → chấp nhận live; dưới ngưỡng → coi là spoof (từ chối). Báo cáo hay dùng **0.5**; có thể hạ **0.3** để chấp nhận thêm live. |
| **BPCER** | Tỷ lệ **live bị từ chối nhầm** (false reject). *Càng thấp càng tốt cho trải nghiệm tài xế.* |
| **APCER** | Tỷ lệ **spoof bị chấp nhận nhầm** (false accept). *Càng thấp càng an toàn.* Tập `drivers_250_fn` chỉ có live → không đo APCER trên tập này. |
| **SFAS / expansion 1.6, 2.7** | Cách **phóng khung mặt** sau detect (Silent-Face-Anti-Spoofing). Số **1.6** / **2.7** = hệ số mở rộng bbox; ảnh hưởng trực tiếp đến BPCER trên drivers. |
| **Gate BPCER ≤ 0.10** | Tiêu chí nội bộ: tối đa **10%** ảnh live thật bị reject trên tập kiểm thử production. |

### 10.1 Kết luận

**Bài toán production:** verify ảnh check-in tài xế — ưu tiên **không từ chối nhầm người thật** (BPCER thấp). Benchmark thêm trên CelebA, CASIA, Face VN để so sánh model; **quyết định triển khai** dựa chủ yếu vào **`drivers_250_fn`** (393 ảnh live thực tế, crop SFAS exp **1.6** và **2.7**).

**Kết quả trên drivers @ ngưỡng 0.5 (§7.4, §7.5):**

| Cấu hình | MiniFASNet | ViT-FAS | ONNX |
|----------|------------|---------|------|
| SFAS **exp 1.6** — BPCER (live bị reject) | **5.9%** (23/393) | 17.6% (69/393) | 19.6% (77/393) |
| SFAS **exp 2.7** | 6.1% (24/393) | 17.6% (69/393) | 25.0% (98/393) |

- **Crop SFAS exp 1.6 + MiniFASNet** cho BPCER thấp nhất và **đạt gate ≤ 10%** trên 393 mẫu drivers.
- **ViT-FAS** và **ONNX** trên cùng crop: BPCER ~18–25% → khoảng **1/5–1/4** ảnh live thật bị reject → **chưa phù hợp** làm model chính nếu KPI giống drivers.
- Trên benchmark có spoof (CelebA, CASIA, Face VN): ONNX/ViT đôi khi **APCER** tốt hơn (ít spoof lọt), nhưng trade-off **BPCER** trên VN vẫn cao hơn MiniFASNet (xem §7.1–7.3).

**Kết luận triển khai:** có thể **pilot** (chạy thử có kiểm soát) **MiniFASNet** với **bbox expansion 1.6** và ngưỡng **0.5**, sau khi: (1) xác nhận **APCER** trên tập có spoof (Face VN / dữ liệu nội bộ); (2) đồng bộ pipeline detect + crop với benchmark.

### 10.2 So sánh ba model (góc nhìn drivers)

| Model | Trên drivers (live thật) | Trên benchmark có spoof | Gợi ý vai trò |
|-------|---------------------------|-------------------------|---------------|
| **MiniFASNet** | BPCER thấp nhất; exp 1.6 đạt gate 10% | APCER không phải tốt nhất trên mọi tập | **Ứng viên chính** cho verify tài xế + SFAS **1.6** |
| **ViT-FAS** | BPCER cao (~18%) | Kém hơn trên CelebA / VN | Giữ để **so sánh nội bộ**, chưa deploy |
| **FaceAntispoof-ONNX** | BPCER cao nhất trên drivers (~20–25%) | Mạnh chống spoof lọt (APCER tốt trên VN/CelebA) | Chỉ cân nhắc nếu ưu tiên **bảo mật** và chấp nhận reject live nhiều hơn; cần chỉnh ngưỡng / fine-tune |

### 10.3 Đề xuất tích hợp service

| Hạng mục | Đề xuất |
|----------|---------|
| **Trạng thái** | **Pilot có điều kiện** — chưa rollout toàn bộ cho đến khi có thêm kiểm thử spoof + sign-off vận hành. |
| **Model** | **MiniFASNetV2** (`minifasnet_v2_2p7`). |
| **Tiền xử lý ảnh** | Detect mặt + crop SFAS với **bbox_expansion = 1.6** (cùng tham số benchmark §7.4). |
| **Ngưỡng `live_score`** | Mặc định **0.5** → BPCER **5.9%** (23/393) trên drivers exp 1.6. Hạ **0.3** → **4.1%** (16/393) nếu muốn giảm thêm false reject; cần đo lại **APCER** trên tập spoof. |
| **Tiêu chí chấp nhận** | Trên tập live production mở rộng: **BPCER ≤ 10%**; song song theo dõi **APCER** khi có ảnh spoof. |
| **Việc cần làm trước GO** | Kiểm tra APCER trên Face VN / spoof nội bộ; mở rộng tập `drivers_250_fn`; so sánh exp **1.6** vs **2.7** (latency, chất lượng ảnh crop). |

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

- [Silent-Face-Anti-Spoofing / MiniFASNet](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
- [ViT spoof detection (PDA)](https://github.com/ArchitRastogi20/vit-spoof-detection-pda) · [HF weights](https://huggingface.co/ArchitRastogi/vit-spoof-detection-pda)
- [face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx)
- [CelebA-Spoof](https://github.com/Davidzhangyuanhan/CelebA-Spoof) · [HF sample pack](https://huggingface.co/datasets/nguyenkhoa/celeba-spoof-for-face-antispoofing-test)
- [CASIA Face Anti-Spoofing Database](https://www.cbsr.ia.ac.cn/english/Face%20Anti-spoofing%20Database%20Published.html) · HF mirror `vu-hong-quang/casia_fasd`
- [Face Anti-Spoofing VN](https://huggingface.co/datasets/vu-hong-quang/face_antispoofing_vn) (private)
- ISO/IEC 30107-3 (APCER, BPCER)

### D. Changelog báo cáo

| Ngày | Phiên bản | Thay đổi |
|------|-----------|----------|
| 28/05/2026 | v0.1 | Khởi tạo mẫu |
| 29/05/2026 | v0.2 | Bổ sung kết quả `face_antispoofing_vn` (3 model) |
| 01/06/2026 | v0.3 | Bổ sung đánh giá `drivers_250_fn`, thêm metric BPCER trọng tâm cho production, thay confusion matrix ảnh bằng bảng markdown |
| 01/06/2026 | v0.4 | Cập nhật mục 10: phân tích model phù hợp nhất cho `drivers_250_fn`, nêu điểm mạnh/yếu từng model theo dữ liệu thực tế |
| 01/06/2026 | v0.5 | Chốt kết luận NO-GO cho production trên `drivers_250_fn`; bổ sung acceptance gate theo BPCER |
| 01/06/2026 | v0.6 | Ablation SFAS bbox expansion (1.0–1.6); bảng dataset×pred True/False; phân phối `live_score`; đường cong BPCER vs threshold (mermaid); cập nhật kết luận MiniFASNet @ exp 1.6 |
| 01/06/2026 | v0.7 | Ma trận 3×7 (`--all`); bảng True/False + phân phối y_prob; nhúng PNG `apcer_bpcer_vs_threshold` (thay mermaid) |
| 01/06/2026 | v0.8 | Metric đầy đủ threshold 0.1–0.9 (HF + drivers); biểu đồ APCER/BPCER nhúng trong §7.1–7.4 |
| 01/06/2026 | v0.9 | §7.5.2: phân phối `y_prob` @0.5 cho cả 4 SFAS expansion (1.0–1.6), tính từ `latest.csv` |
| 01/06/2026 | v1.0 | Expansion 1.5, 2.7, 4.0; cập nhật §7.4–7.5, §10 từ evaluation mới |
| 01/06/2026 | v1.3 | §7.1–7.4 chỉ ngưỡng 0.3/0.5/0.7; đủ expansion face (1.0, 1.6, 2.7) và drivers (1.0–2.7); drivers 393 live; bỏ exp 4 |
| 01/06/2026 | v1.4 | Điền nguồn repo model (§3.1) và dataset (§4.1); cập nhật số liệu face VN 11376 |
| 01/06/2026 | v1.5 | `SFAS_BBOX_EXPANSIONS = (1.6, 2.7)`; baseline exp 1.0 tách config riêng |
| 01/06/2026 | v1.6 | §7.0 chú thích APCER/BPCER, CM, biểu đồ; nhúng PNG evaluation |
| 01/06/2026 | v1.7 | Bỏ ACER; bảng metric dạng `tỷ lệ (số_lỗi/tổng_lớp)` |
| 01/06/2026 | v1.8 | Bỏ SFAS exp 1.0 khỏi §7; bỏ bảng CM markdown (chỉ ảnh CM) |
