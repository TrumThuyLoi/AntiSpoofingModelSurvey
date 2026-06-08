# Báo cáo đánh giá Anti-Spoofing (MiniFASNet, FaceAntispoof-ONNX, Hairymax ONNX)

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
- Model đã đánh giá: **MiniFASNetV2** (`minifasnet_v2_2p7`), **FaceAntispoof-ONNX** (`face_antispoof_onnx_9820`), **Hairymax ONNX** pretrained/retrain (`hairymax_bin_1.5_*`). *ViT-FAS loại khỏi báo cáo.*
- Dataset đã dùng: **face_antispoofing_vn** (11376 mẫu) và **drivers_250_fn** (393 live), crop SFAS **exp 1.5**; `run_inference.py` / `run_evaluation.py` + `patch_report_section7.py`.
- Kết luận chính: trên **drivers @ exp 1.5** @0.5 — **MiniFASNet BPCER 8.4% (33/393)**; **Hairymax retrain 12.7% (50/393)**; ONNX 20.4% (80/393); Hairymax pretrained 41.0% (161/393). Trên **Face VN exp 1.5**: trade-off APCER/BPCER giữa MiniFASNet và ONNX (§7.1). Quyết định triển khai dựa trên hai tập này.
- Đề xuất tích hợp service: ưu tiên **BPCER trên live thực tế**; pilot MiniFASNet với **bbox expansion 1.5**; calibrate ngưỡng theo production.

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
| FaceAntispoof-ONNX | `face_antispoof_onnx_9820` | `models/face_antispoof_onnx_best_9820.pth` | `configs/model_face_antispoof_onnx.yaml` |

Tải weight: `python3 scripts/download_pretrained_weights.py`. Submodule: `git submodule update --init third_party/Silent-Face-Anti-Spoofing third_party/face-antispoof-onnx`. Hairymax: `tich_hop_onnx_hairymax.md`.

---

## 4. Dataset

### 4.1 Dataset sử dụng

| Dataset | Nguồn (HF / local) | Ghi chú |
|---------|-------------------|---------|
| Face Anti-Spoofing VN | [kaggle/face_antispoofing_vn](https://www.kaggle.com/datasets/maihongtng/face-anti-spoofing?resource=download&select=Image) | Ảnh tài xế VN; crop SFAS offline **exp 1.5** (`face_antispoofing_vn_exp1.5`) |
| Drivers 250 FN | Local: `data/drivers_250_FN/` → crop (`drivers_250_fn_cropped*`) | Tập FN nội bộ; 393 live; crop SFAS **exp 1.5** (`drivers_250_fn_exp1.5`); `tich_hop_onnx_hairymax.md` |

### 4.2 Số lượng ảnh (sample / test)

| Dataset | Tổng | Live | Spoof | Unknown / lỗi |
|---------|------|------|-------|----------------|
| face_antispoofing_vn_exp1.5 | 11376 | 7541 | 3835 | 0 |
| drivers_250_fn_exp1.5 | 393 | 393 | 0 | 0 |

*(CSV: `data/sampled/*_sample.csv`; HF raw: `data/raw/<dataset>/annotations/raw.csv`; spec repo HF: `src/datasets/specs.py`.)*

### 4.3 Đặc điểm tiền xử lý dữ liệu đầu vào

- Face Anti-Spoofing VN: crop SFAS **bbox_expansion = 1.5** từ `data/face_antispoofing_vn/`; inference resize theo model (80×80 / 128×128)
- Drivers 250 FN: crop SFAS **bbox_expansion = 1.5** từ `data/drivers_250_FN/` (393 live)

---

## 7. Kết quả

Phạm vi: **Face Anti-Spoofing VN** + **drivers_250_fn** (ảnh live thật), crop SFAS **exp 1.5**.
Ngưỡng: **0.3, 0.5, 0.7** (`configs/evaluation.yaml`). Metric từ `predictions/latest.csv` (Face VN: live+spoof; drivers: 393 live, APCER = 0).

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

**Định dạng bảng §7.1–7.2:** mỗi ô Accuracy / APCER / BPCER = **`tỷ lệ (số_lỗi/tổng_lớp)`** (vd. BPCER `0.059 (23/393)` = 23 live bị reject trên 393 live).

**Production (verify tài xế):** gate đề xuất **BPCER ≤ 0.10** trên tập live thực tế; kiểm tra thêm APCER khi có tập spoof.

### 7.1 Face Anti-Spoofing VN (SFAS exp 1.5)

- **11376** mẫu (7541 live, 3835 spoof). `source_dataset`: `face_antispoofing_vn_exp1.5`.

#### SFAS exp 1.5 (`face_antispoofing_vn_exp1.5`)

| Model | Threshold | Accuracy | APCER | BPCER |
|-------|-----------|----------|-------|-------|
| MiniFASNet | 0.3 | 0.7764 (8832/11376) | 0.6052 (2321/3835) | 0.0296 (223/7541) |
| MiniFASNet | 0.5 | 0.7937 (9029/11376) | 0.5338 (2047/3835) | 0.0398 (300/7541) |
| MiniFASNet | 0.7 | 0.8098 (9212/11376) | 0.4587 (1759/3835) | 0.0537 (405/7541) |
| ONNX | 0.3 | 0.7877 (8961/11376) | 0.4532 (1738/3835) | 0.0898 (677/7541) |
| ONNX | 0.5 | 0.7870 (8953/11376) | 0.3898 (1495/3835) | 0.1231 (928/7541) |
| ONNX | 0.7 | 0.7847 (8927/11376) | 0.3241 (1243/3835) | 0.1599 (1206/7541) |

Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet APCER/BPCER](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.5/apcer_bpcer_vs_threshold.png) | ![ONNX APCER/BPCER](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.5/apcer_bpcer_vs_threshold.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| *(Hairymax pre: chưa có `apcer_bpcer_vs_threshold.png`)* | *(Hairymax retrain: chưa có `apcer_bpcer_vs_threshold.png`)* |

Phân phối `live_score` (`live_score_distribution.png`):

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet live_score](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.5/live_score_distribution.png) | ![ONNX live_score](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.5/live_score_distribution.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| *(Hairymax pre: chưa có `live_score_distribution.png`)* | *(Hairymax retrain: chưa có `live_score_distribution.png`)* |

Confusion matrix @0.5 (`confusion_matrix_0.5.png`):

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet CM@0.5](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.5/confusion_matrix_0.5.png) | ![ONNX CM@0.5](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.5/confusion_matrix_0.5.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| *(Hairymax pre: chưa có `confusion_matrix_0.5.png`)* | *(Hairymax retrain: chưa có `confusion_matrix_0.5.png`)* |

**FN (âm tính giả)** — nhãn **live**, model **reject** (`live_score` < 0.5). Ảnh là **người thật** (không phải spoof). `false_negative_montage_0.5.png`:

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet FN](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.5/false_negative_montage_0.5.png) | ![ONNX FN](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.5/false_negative_montage_0.5.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| *(Hairymax pre: chưa có `false_negative_montage_0.5.png`)* | *(Hairymax retrain: chưa có `false_negative_montage_0.5.png`)* |

**FP (dương tính giả)** — nhãn **spoof**, model **chấp nhận** (`live_score` ≥ 0.5). Ảnh là **spoof** (thư mục `not_live` / spoof trên Face VN). `false_positive_montage_0.5.png`:

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet FP](./reports/models/minifasnet_v2_2p7/metrics/face_antispoofing_vn_exp1.5/false_positive_montage_0.5.png) | ![ONNX FP](./reports/models/face_antispoof_onnx_9820/metrics/face_antispoofing_vn_exp1.5/false_positive_montage_0.5.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| *(Hairymax pre: chưa có `false_positive_montage_0.5.png`)* | *(Hairymax retrain: chưa có `false_positive_montage_0.5.png`)* |

### 7.2 Drivers 250 FN (SFAS exp 1.5)

- **393** live. **APCER = 0**. `source_dataset`: `drivers_250_fn_exp1.5`.
- Hairymax ONNX (pretrained / retrain) benchmark trên tập này.

#### SFAS exp 1.5 (`drivers_250_fn_exp1.5`)

| Model | Threshold | Accuracy | APCER | BPCER |
|-------|-----------|----------|-------|-------|
| MiniFASNet | 0.3 | 0.9415 (370/393) | 0.0000 (—) | 0.0585 (23/393) |
| MiniFASNet | 0.5 | 0.9160 (360/393) | 0.0000 (—) | 0.0840 (33/393) |
| MiniFASNet | 0.7 | 0.8906 (350/393) | 0.0000 (—) | 0.1094 (43/393) |
| ONNX | 0.3 | 0.8448 (332/393) | 0.0000 (—) | 0.1552 (61/393) |
| ONNX | 0.5 | 0.7964 (313/393) | 0.0000 (—) | 0.2036 (80/393) |
| ONNX | 0.7 | 0.7481 (294/393) | 0.0000 (—) | 0.2519 (99/393) |
| Hairymax pre | 0.3 | 0.6438 (253/393) | 0.0000 (—) | 0.3562 (140/393) |
| Hairymax pre | 0.5 | 0.5903 (232/393) | 0.0000 (—) | 0.4097 (161/393) |
| Hairymax pre | 0.7 | 0.5420 (213/393) | 0.0000 (—) | 0.4580 (180/393) |
| Hairymax retrain | 0.3 | 0.8906 (350/393) | 0.0000 (—) | 0.1094 (43/393) |
| Hairymax retrain | 0.5 | 0.8728 (343/393) | 0.0000 (—) | 0.1272 (50/393) |
| Hairymax retrain | 0.7 | 0.8473 (333/393) | 0.0000 (—) | 0.1527 (60/393) |

Đường cong APCER / BPCER (`apcer_bpcer_vs_threshold.png`):

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet APCER/BPCER](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.5/apcer_bpcer_vs_threshold.png) | ![ONNX APCER/BPCER](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.5/apcer_bpcer_vs_threshold.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| ![Hairymax pre APCER/BPCER](./reports/models/hairymax_bin_1p5_pretrained/metrics/drivers_250_fn_exp1.5/apcer_bpcer_vs_threshold.png) | ![Hairymax retrain APCER/BPCER](./reports/models/hairymax_bin_1p5_retrain/metrics/drivers_250_fn_exp1.5/apcer_bpcer_vs_threshold.png) |

Phân phối `live_score` (`live_score_distribution.png`):

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet live_score](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.5/live_score_distribution.png) | ![ONNX live_score](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.5/live_score_distribution.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| ![Hairymax pre live_score](./reports/models/hairymax_bin_1p5_pretrained/metrics/drivers_250_fn_exp1.5/live_score_distribution.png) | ![Hairymax retrain live_score](./reports/models/hairymax_bin_1p5_retrain/metrics/drivers_250_fn_exp1.5/live_score_distribution.png) |

Confusion matrix @0.5 (`confusion_matrix_0.5.png`):

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet CM@0.5](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.5/confusion_matrix_0.5.png) | ![ONNX CM@0.5](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.5/confusion_matrix_0.5.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| ![Hairymax pre CM@0.5](./reports/models/hairymax_bin_1p5_pretrained/metrics/drivers_250_fn_exp1.5/confusion_matrix_0.5.png) | ![Hairymax retrain CM@0.5](./reports/models/hairymax_bin_1p5_retrain/metrics/drivers_250_fn_exp1.5/confusion_matrix_0.5.png) |

**FN (âm tính giả)** — nhãn **live**, model **reject** (`live_score` < 0.5). Ảnh là **người thật** (không phải spoof). `false_negative_montage_0.5.png`:

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet FN](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.5/false_negative_montage_0.5.png) | ![ONNX FN](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.5/false_negative_montage_0.5.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| ![Hairymax pre FN](./reports/models/hairymax_bin_1p5_pretrained/metrics/drivers_250_fn_exp1.5/false_negative_montage_0.5.png) | ![Hairymax retrain FN](./reports/models/hairymax_bin_1p5_retrain/metrics/drivers_250_fn_exp1.5/false_negative_montage_0.5.png) |

### 7.5 Ma trận benchmark — tóm tắt @0.5

Inference + evaluation trên hai dataset §7.1–7.2. Bảng đầy đủ threshold: §7.1–7.2.

Định dạng mỗi ô: **BPCER (FN/n_live) / APCER (FP/n_spoof)**.

| Dataset | MiniFASNet | ONNX | Hairymax pre | Hairymax retrain |
|---------|:---:|:---:|:---:|:---:|
| face_antispoofing_vn_exp1.5 | 0.0398 (300/7541) / 0.5338 (2047/3835) | 0.1231 (928/7541) / 0.3898 (1495/3835) | — / — | — / — |
| drivers_250_fn_exp1.5 | 0.0840 (33/393) / 0.0000 (—) | 0.2036 (80/393) / 0.0000 (—) | 0.4097 (161/393) / 0.0000 (—) | 0.1272 (50/393) / 0.0000 (—) |

#### 7.5.1 Dataset × dự đoán @0.5 (`True` / `False`)

Nguồn: `reports/models/<model_id>/predictions/<dataset>/latest.csv`, ngưỡng **0.5**.

- **True** = số ảnh **live** được model chấp nhận (pred live → TP trên live).
- **False** = số ảnh **live** bị reject (pred spoof → FN trên live).
- Định dạng ô: **`True / False`** (chỉ đếm mẫu nhãn live; spoof không hiển thị ở đây).

**Phạm vi dataset** (khớp §7.1–7.2):
- **Face VN** + **drivers_250_fn**, crop SFAS **exp 1.5**.

| Dataset | MiniFASNet | ONNX | Hairymax pre | Hairymax retrain |
|---------|:---:|:---:|:---:|:---:|
| face_antispoofing_vn_exp1.5 (7541 live, 11376 mẫu) | 7241 / 300 | 6613 / 928 | — / — | — / — |
| drivers_250_fn_exp1.5 (393 live) | 360 / 33 | 313 / 80 | 232 / 161 | 343 / 50 |

#### 7.5.2 Phân phối `y_prob` @0.5 (nhóm True / False)

`live_score` = **y_prob**; ngưỡng **0.5**. **Drivers** SFAS **exp 1.5** (393 live).

- **True** = live được chấp nhận (TP); **False** = live bị reject (FN).
- Artifact: `live_score_true_false_0.5.png`, `live_score_histogram_bins_0.5.png` (`run_evaluation.py`, chỉ `drivers_250_fn*`).

**Drivers exp 1.5** (`drivers_250_fn_exp1.5`, 393 live):

Phân phối theo nhóm True / False @0.5 (`live_score_true_false_0.5.png`):

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet True/False](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.5/live_score_true_false_0.5.png) | ![ONNX True/False](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.5/live_score_true_false_0.5.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| ![Hairymax pre True/False](./reports/models/hairymax_bin_1p5_pretrained/metrics/drivers_250_fn_exp1.5/live_score_true_false_0.5.png) | ![Hairymax retrain True/False](./reports/models/hairymax_bin_1p5_retrain/metrics/drivers_250_fn_exp1.5/live_score_true_false_0.5.png) |

Histogram toàn tập live (step **0.1**, `live_score_histogram_bins_0.5.png`):

| MiniFASNet | ONNX |
| :---: | :---: |
| ![MiniFASNet histogram](./reports/models/minifasnet_v2_2p7/metrics/drivers_250_fn_exp1.5/live_score_histogram_bins_0.5.png) | ![ONNX histogram](./reports/models/face_antispoof_onnx_9820/metrics/drivers_250_fn_exp1.5/live_score_histogram_bins_0.5.png) |
| Hairymax pre | Hairymax retrain |
| :---: | :---: |
| ![Hairymax pre histogram](./reports/models/hairymax_bin_1p5_pretrained/metrics/drivers_250_fn_exp1.5/live_score_histogram_bins_0.5.png) | ![Hairymax retrain histogram](./reports/models/hairymax_bin_1p5_retrain/metrics/drivers_250_fn_exp1.5/live_score_histogram_bins_0.5.png) |

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
| **SFAS / expansion 1.5** | Cách **phóng khung mặt** sau detect (Silent-Face-Anti-Spoofing). Hệ số **1.5** = mở rộng bbox; dùng thống nhất cho Face VN và drivers trong báo cáo này. |
| **Gate BPCER ≤ 0.10** | Tiêu chí nội bộ: tối đa **10%** ảnh live thật bị reject trên tập kiểm thử production. |

### 10.1 Kết luận

**Bài toán production:** verify ảnh check-in tài xế — ưu tiên **không từ chối nhầm người thật** (BPCER thấp). Báo cáo chỉ gồm **face_antispoofing_vn** (có spoof) và **drivers_250_fn** (393 live), crop SFAS **exp 1.5**.

**Kết quả @ ngưỡng 0.5 (§7.1–7.2, §7.5):**

| Tập | MiniFASNet | ONNX | Hairymax pre | Hairymax retrain |
|-----|------------|------|--------------|------------------|
| **drivers** — BPCER @0.5 | **8.4%** (33/393) | 20.4% (80/393) | 41.0% (161/393) | **12.7%** (50/393) |
| **Face VN** — BPCER / APCER @0.5 | **4.0%** / 53.4% | 12.3% / **39.0%** | — | — |

- **MiniFASNet @ exp 1.5** cho BPCER thấp nhất trên drivers; **đạt gate ≤ 10%** (§7.2).
- **Hairymax retrain** cạnh tranh (12.7% BPCER) nhưng vẫn cao hơn MiniFASNet; pretrained ~41%.
- **ONNX**: BPCER ~20% trên drivers → cần chỉnh ngưỡng hoặc fine-tune nếu làm model chính.
- Trên **Face VN exp 1.5**: so sánh APCER/BPCER §7.1 trước khi rollout.

**Kết luận triển khai:** có thể **pilot** **MiniFASNet** với **bbox expansion 1.5** và ngưỡng **0.5**, sau khi xác nhận APCER trên Face VN và đồng bộ pipeline detect + crop với benchmark.

### 10.2 So sánh model (góc nhìn drivers)

| Model | Trên drivers (live thật) | Trên benchmark có spoof | Gợi ý vai trò |
|-------|---------------------------|-------------------------|---------------|
| **MiniFASNet** | BPCER thấp nhất @ exp 1.5 (8.4% drivers) | Trade-off APCER trên Face VN §7.1 | **Ứng viên chính** + SFAS **1.5** |
| **FaceAntispoof-ONNX** | BPCER ~20% trên drivers exp 1.5 | APCER thường tốt hơn trên Face VN | Cân nhắc nếu ưu tiên **bảo mật** |
| **Hairymax ONNX** | retrain **12.7%**; pretrained ~41% @ exp 1.5 | Chưa có metric Face VN | Thử nghiệm ONNX 128×128; crop **1.5** |

### 10.3 Đề xuất tích hợp service

| Hạng mục | Đề xuất |
|----------|---------|
| **Trạng thái** | **Pilot có điều kiện** — chưa rollout toàn bộ cho đến khi có thêm kiểm thử spoof + sign-off vận hành. |
| **Model** | **MiniFASNetV2** (`minifasnet_v2_2p7`). |
| **Tiền xử lý ảnh** | Detect mặt + crop SFAS với **bbox_expansion = 1.5** (cùng tham số benchmark §7.2). |
| **Ngưỡng `live_score`** | Mặc định **0.5** → BPCER **8.4%** (33/393) trên drivers exp 1.5. Hạ **0.3** → **5.9%** (23/393); cần đo lại **APCER** trên Face VN. |
| **Tiêu chí chấp nhận** | Trên tập live production mở rộng: **BPCER ≤ 10%**; song song theo dõi **APCER** trên Face VN. |
| **Việc cần làm trước GO** | Kiểm tra APCER trên `face_antispoofing_vn_exp1.5`; mở rộng tập `drivers_250_fn`; benchmark Hairymax trên Face VN nếu cần. |

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
- [face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx)
- [hairymax/Face-AntiSpoofing](https://github.com/hairymax/Face-AntiSpoofing) (ONNX 128×128)
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
| 03/06/2026 | v1.9 | Bỏ ViT-FAS khỏi báo cáo; §7: MiniFASNet + ONNX + Hairymax; drivers exp 1.5 |
| 03/06/2026 | v2.0 | Thu hẹp phạm vi: chỉ **face_antispoofing_vn** + **drivers_250_fn**, SFAS **exp 1.5**; bỏ CelebA/CASIA và expansion 1.6/2.7 |
