# Project: MiniFASNet Anti-Spoofing Survey

## Overview
MiniFASNet Anti-Spoofing Survey là dự án nhỏ dùng để khảo sát, chạy thử và đánh giá mô hình anti-spoofing cho bài toán xác minh khuôn mặt.

Mục tiêu ban đầu của dự án là đánh giá mô hình MiniFASNet trên một phần nhỏ của tập dữ liệu CelebA-Spoof, sử dụng khoảng 2000 ảnh live và 2000 ảnh spoof.

Dự án không tập trung vào việc train model mới ngay từ đầu, mà tập trung vào:
- hiểu pipeline anti-spoofing
- chuẩn bị tập test sạch
- chạy inference bằng model có sẵn
- tính metric đánh giá
- phân tích failure case
- viết báo cáo kỹ thuật

Dự án có thể mở rộng để khảo sát thêm các mô hình anti-spoofing khác và các tập dữ liệu khác.

## Architecture

### Directory Structure
- `/data/raw` - Dữ liệu gốc tải về từ CelebA-Spoof hoặc các dataset khác
- `/data/sampled` - Tập dữ liệu đã sample, ví dụ 2000 ảnh live và 2000 ảnh spoof
- `/data/labeled` - Dữ liệu đã kiểm tra lại nhãn hoặc export từ Label Studio
- `/data/processed` - Dữ liệu đã crop mặt, resize, normalize hoặc chuẩn hóa format
- `/models` - Chứa pretrained weights của MiniFASNet và các model khác
- `/notebooks` - Notebook dùng để khám phá dữ liệu, kiểm tra ảnh mẫu, phân tích kết quả
- `/src/datasets` - Code đọc dataset, parse annotation, split dữ liệu
- `/src/models` - Wrapper cho MiniFASNet và các model anti-spoofing khác
- `/src/inference` - Code chạy inference trên từng ảnh hoặc batch ảnh
- `/src/evaluation` - Code tính metric như accuracy, APCER, BPCER, ACER, confusion matrix
- `/src/utils` - Hàm tiện ích: đọc ảnh, lưu JSON/CSV, logging, xử lý path
- `/scripts` - Script chạy các tác vụ chính như sample data, run inference, evaluate
- `/reports` - Báo cáo kết quả khảo sát, bảng metric, hình ảnh minh họa failure case (xem dưới)
- `/configs` - File cấu hình cho dataset, model, threshold, batch size
- `/docker` - Dockerfile, docker-compose và cấu hình service phụ nếu cần
- `/label-studio` - Cấu hình hoặc file export/import phục vụ kiểm tra nhãn bằng Label Studio

### Key Patterns
1. Không sửa trực tiếp dữ liệu gốc trong `/data/raw`
2. Mọi dữ liệu dùng để benchmark phải có file annotation (chú thích) rõ ràng
3. Mỗi lần chạy inference phải lưu lại prediction, score và metadata
4. Metric phải được tính từ file prediction đã lưu, không tính thủ công
5. Mỗi model phải có wrapper riêng trong `/src/models`
6. Mỗi dataset phải có loader riêng trong `/src/datasets`
7. Các tham số như threshold, input size, model path phải đặt trong `/configs`
8. Kết quả benchmark phải có khả năng chạy lại bằng script
9. Báo cáo phải ghi rõ dataset, số lượng ảnh, model, weight, threshold và metric
10. Code phục vụ khảo sát phải ưu tiên rõ ràng, dễ kiểm chứng hơn là tối ưu quá sớm

## Coding Conventions

### Python
- Sử dụng Python 3.10 hoặc mới hơn
- Ưu tiên type hint cho function quan trọng
- Không hard-code absolute path
- Không viết toàn bộ logic trong notebook
- Notebook chỉ dùng để khám phá và phân tích, logic chính phải nằm trong `/src`
- Function nên nhỏ, rõ nhiệm vụ
- Tên biến dùng tiếng Anh, comment có thể dùng tiếng Việt nếu cần giải thích
- Không dùng wildcard import
- Không silent exception nếu lỗi ảnh, lỗi path hoặc lỗi model
- Khi skip ảnh lỗi, phải log lại lý do

### Dataset
- Dataset gốc phải giữ nguyên trong `/data/raw`
- File annotation chuẩn nên có các trường:
  - `image_path`
  - `label`
  - `source_dataset`
  - `split`
  - `is_valid`
  - `note`
- Label chuẩn hóa:
  - `live` cho ảnh thật
  - `spoof` cho ảnh giả
  - `unknown` cho ảnh bị lỗi hoặc không nhận biết được
- Không dùng lẫn nhiều format label trong cùng một pipeline
- Nếu ảnh bị lỗi, không detect được mặt hoặc nhãn mơ hồ (unkown), đánh dấu `is_valid = false`
- Không xóa ảnh lỗi khỏi dataset nếu chưa có lý do rõ ràng

### Model
- Mỗi model phải được bọc bằng một class hoặc interface thống nhất
- Model wrapper nên có các method chính:
  - `load()`
  - `preprocess(image)`
  - `predict(image)`
  - `predict_batch(images)`
- Output của model phải được chuẩn hóa về format chung:
  - `label_pred`
  - `live_score`
  - `spoof_score`
  - `raw_output`
- Không để logic tính metric nằm trong model wrapper
- Không thay đổi weight pretrained nếu chỉ đang chạy benchmark inference

### Evaluation
- Luôn lưu prediction ra file trước khi tính metric
- Metric tối thiểu cần có:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
  - APCER
  - BPCER
  - ACER
- Cần đánh giá theo nhiều threshold nếu model trả về score
- Không chỉ báo cáo accuracy vì anti-spoofing cần quan tâm lỗi để lọt spoof
- Phải tách rõ false accept và false reject
- Nên lưu failure cases để phân tích thủ công

### Reports
- Báo cáo phải viết bằng tiếng Việt hoặc tiếng Anh nhất quán
- Mỗi báo cáo nên có:
  - mục tiêu thử nghiệm
  - dataset sử dụng
  - số lượng ảnh live/spoof
  - model và weight sử dụng
  - preprocessing
  - metric
  - nhận xét kết quả
  - failure cases
  - đề xuất bước tiếp theo
- Không tuyên bố model là SOTA nếu chưa có bằng chứng benchmark hoặc paper rõ ràng
- Phân biệt rõ “paper SOTA”, “open-source usable” và “production-friendly”

## DO NOT
- Không train model mới khi chưa chạy được benchmark inference cơ bản
- Không dùng toàn bộ CelebA-Spoof ngay từ đầu nếu chưa kiểm soát được pipeline
- Không sửa, đổi tên hoặc di chuyển dữ liệu gốc trong `/data/raw`
- Không đánh giá model chỉ bằng vài ảnh demo
- Không chỉ nhìn accuracy rồi kết luận model tốt
- Không bỏ qua ảnh lỗi mà không log lại
- Không hard-code path theo máy cá nhân
- Không commit dataset lớn, model weight lớn hoặc file nhạy cảm lên Git
- Không lưu secret, token hoặc credential trong code
- Không trộn logic dataset, model và evaluation vào cùng một file lớn
- Không dùng kết quả từ paper để thay thế benchmark thực tế trên tập test của dự án
- Không gọi một model là phù hợp production nếu chưa kiểm tra latency, memory và khả năng dockerize

## Common Tasks

### Khởi tạo dự án
1. Tạo cấu trúc thư mục chuẩn
2. Tạo virtual environment hoặc Docker environment
3. Cài các thư viện cần thiết
4. Tạo file config mẫu trong `/configs`
5. Tạo script kiểm tra môi trường chạy

### Thêm dataset CelebA-Spoof
1. Tải sub dataset của celeba-spoof về `/data/raw/celeba-spoof` chứ không tải toàn bộ dataset gốc
2. Đọc annotation gốc của dataset
3. Viết loader trong `/src/datasets/celeba_spoof.py`
4. Chuẩn hóa label về `live` và `spoof`
5. Kiểm tra số lượng ảnh live/spoof
6. Sample khoảng 2000 live và 2000 spoof
7. Lưu danh sách sample vào `/data/sampled/celeba_spoof_sample.csv`

### Kiểm tra lại nhãn bằng Label Studio
1. Chạy Label Studio bằng Docker hoặc local environment
2. Import danh sách ảnh sample
3. Tạo labeling interface với hai nhãn `live` và `spoof`
4. Kiểm tra các ảnh bị nghi ngờ, ảnh lỗi hoặc ảnh khó phân biệt
5. Export kết quả label
6. Lưu file export vào `/data/labeled`
7. Merge label đã kiểm tra vào annotation chuẩn

### Thêm model MiniFASNet
1. Clone hoặc tham khảo source code MiniFASNet chính thức
2. Tải pretrained weight vào `/models/minifasnet`
3. Viết wrapper trong `/src/models/minifasnet.py`
4. Implement preprocessing đúng theo yêu cầu của model
5. Chạy thử inference trên vài ảnh live/spoof
6. Kiểm tra output score và label
7. Viết script batch inference

### Chạy inference
1. Đọc config model và dataset
2. Load annotation từ `/data/sampled` hoặc `/data/labeled`
3. Load model weight
4. Chạy predict trên từng ảnh hoặc batch ảnh
5. Lưu kết quả theo **model_id** vào `reports/models/<model_id>/predictions/<dataset>/`  
   - File theo thời gian: `run_<dataset>_YYYYMMDD_HHMMSS.csv`  
   - File mới nhất: `latest.csv` (ghi đè **chỉ trong cùng `<model_id>`**)
6. Log ảnh lỗi hoặc ảnh không chạy được
7. Đảm bảo script có thể chạy lại mà không phụ thuộc notebook

### Tính metric
1. Đọc file prediction đã lưu
2. So sánh `label_true` và `label_pred`
3. Tính accuracy, precision, recall, F1-score
4. Tính confusion matrix
5. Tính APCER, BPCER và ACER
6. Thử nhiều threshold nếu có score
7. Lưu bảng metric vào `reports/models/<model_id>/metrics/<dataset>/`  
   - `metrics_summary.csv`  
   - `metrics_threshold_<t>.json`  
   - `confusion_matrix_<t>.png`
8. Lưu biểu đồ hoặc hình minh họa nếu cần

### Phân tích failure cases
1. Lọc false accept: spoof nhưng model dự đoán live
2. Lọc false reject: live nhưng model dự đoán spoof
3. Copy hoặc tạo danh sách ảnh lỗi vào `reports/models/<model_id>/failure_cases/<dataset>/`
4. Quan sát các pattern lỗi thường gặp
5. Ghi nhận nguyên nhân có thể:
   - ảnh mờ
   - ánh sáng yếu
   - mặt nghiêng
   - ảnh bị crop lỗi
   - spoof chất lượng cao
   - màn hình phản chiếu ít
6. Đưa nhận xét vào báo cáo

### Thêm model anti-spoofing mới
1. Tạo file wrapper mới trong `/src/models`
2. Chuẩn hóa input/output giống MiniFASNet wrapper
3. Thêm config model trong `/configs`
4. Chạy inference trên cùng tập test
5. Tính metric bằng cùng script evaluation
6. So sánh công bằng với MiniFASNet
7. Ghi rõ khác biệt về accuracy, ACER, tốc độ và độ dễ deploy

### Thêm dataset mới
1. Tải dataset vào `/data/raw`
2. Viết dataset loader riêng
3. Chuẩn hóa label về `live` và `spoof`
4. Tạo sample test nếu dataset quá lớn
5. Chạy cùng pipeline inference và evaluation
6. So sánh khả năng generalization của model trên nhiều dataset

## Suggested First Milestone

Mốc đầu tiên của dự án là chạy được benchmark tối giản cho MiniFASNet trên một tập nhỏ của CelebA-Spoof.

Kết quả cần đạt:
1. Có khoảng 2000 ảnh live và 2000 ảnh spoof đã được sample
2. Có file annotation chuẩn dạng CSV hoặc JSON
3. Chạy được MiniFASNet inference trên toàn bộ tập sample
4. Lưu được prediction cho từng ảnh
5. Tính được các metric cơ bản
6. Có báo cáo ngắn về kết quả và lỗi thường gặp

## Future Extensions
- Khảo sát thêm CDCN, CDCN++ hoặc các model anti-spoofing mới hơn
- Benchmark trên nhiều dataset như OULU-NPU, CASIA-FASD, Replay-Attack, SiW
- Thêm face detection stage trước anti-spoofing
- Thử crop mặt bằng RetinaFace, MTCNN hoặc YuNet
- Tối ưu inference bằng ONNX hoặc TensorRT
- Dockerize toàn bộ benchmark pipeline
- Xây dựng API service đơn giản cho anti-spoofing
- Kết nối với PostgreSQL để log request và kết quả inference
- Tích hợp vào pipeline verify hoàn chỉnh: detect face → anti-spoofing → face embedding → compare
