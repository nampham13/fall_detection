# Fall detection: YOLO26 + RTMPose + tracking + ST-GCN

Pipeline này là baseline kỹ thuật có cơ chế fail-safe, chưa phải thiết bị y tế hay hệ
thống cảnh báo production đã được chứng nhận.

## Quyết định kiến trúc

Luồng xử lý thực tế:

```text
video
  -> YOLO26s (class 0: person)
  -> ByteTrack (bbox ID)
  -> RTMPose-s top-down (COCO-17)
  -> pose-aware ID repair
  -> temporal buffer được resample theo thời gian
  -> ST-GCN (normal / falling / lying)
  -> state machine: abrupt fall + prolonged lying
  -> suspected / confirmed event
```

ID được tạo từ bbox trước rồi gắn skeleton vào ID. Đây là chủ ý: keypoint thường mất
ở tư thế nằm, bị che hoặc ra khỏi khung; tracking skeleton thuần vì vậy dễ đổi ID
đúng lúc hệ thống cần ổn định nhất. Một lớp pose-aware sau ByteTrack nối lại các ID
bị đứt ngắn bằng IoU, khoảng cách tâm và độ tương đồng khớp.

State machine không còn báo ngay từ một frame đổi hướng. Chuyển động đột ngột chỉ
mở một cửa sổ `watch`; muốn lên `suspected` cần tiếp tục quan sát tư thế nằm ổn
định đủ thời gian/số frame, hoặc nằm quá lâu. Cảnh báo được giữ tối thiểu một khoảng
thời gian để tránh dao động. Một body-axis gate phân biệt người nằm với người chỉ
cúi sâu nhưng chân còn thẳng đứng.

Pipeline cũng phát hiện hard scene cut. Khi video đổi cảnh, ByteTrack, pose history
và rule evidence được reset để chuyển động ở hai cảnh khác nhau không bị nối thành
một sự kiện ngã. Scene cut được ghi riêng trong JSONL audit log.

RTMPose-s 256x192 được chọn cho GTX 1650 4 GB. Trong môi trường hiện tại,
ONNX Runtime không thấy `CUDAExecutionProvider`, nên RTMPose tự chạy CPU trong khi
YOLO26s và ST-GCN chạy CUDA. Chi phí pose tăng gần tuyến tính theo số người.

Smoke benchmark ngày 23-06-2026 trên máy hiện tại, ảnh 1080x810 có 5 người:
YOLO26s + ByteTrack khoảng 19.3 FPS; toàn pipeline detector + RTMPose khoảng
7.55 FPS. Đây chỉ là số đo định hướng, không phải SLA. Muốn đạt 20-30 FPS nhiều
người cần profiling rồi export TensorRT/ONNX CUDA tương thích, hoặc chấp nhận
analytics sampling rate 10-15 FPS.

Sau khi thêm persistence, scene-cut reset và full-body lying gate, video test hiện
tại dài 137,4 giây tạo 4 lần chuyển sang `suspected`, tương ứng 3 cụm thời gian.
Một false positive do người hỗ trợ cúi sâu tại 52,75 giây đã được loại bỏ trong khi
ba cảnh ngã đại diện vẫn được giữ. Đây là kiểm tra hồi quy thủ công, chưa thay thế
test set có annotation.

## Safety gate bắt buộc

- Không có checkpoint `models/stgcn_fall.pt`: hệ thống chỉ xuất `suspected`, không
  bao giờ xuất `confirmed`.
- `confirmed` cần đồng thuận giữa ST-GCN và bằng chứng động học/thời gian.
- Không dùng accuracy theo frame làm tiêu chí release. Tối thiểu phải báo cáo:
  event sensitivity, event precision, false alarms/camera-hour, missed falls,
  detection delay p50/p95 và kết quả theo từng camera/ánh sáng/nhóm đối tượng.
- Train/validation/test phải tách theo subject, video và camera. Cắt các cửa sổ từ
  cùng một video rồi random split là data leakage.
- Các bộ fall công khai chủ yếu là hành động dàn dựng bởi người khỏe, ít phản ánh
  ngất thật, ngã bị che, người già, xe lăn, giường bệnh hoặc camera production.
  Cần dữ liệu pilot đúng domain và hard negatives: nằm ngủ, tập thể dục, nhặt đồ,
  ngồi nhanh, quỳ, bò, nhân viên hỗ trợ người bệnh.
- Cảnh báo phải có human-in-the-loop và health monitoring cho camera/model. Không
  dùng output này làm căn cứ duy nhất để đưa quyết định y khoa hoặc an toàn.

## Legal gate

Ultralytics YOLO26 được cung cấp theo AGPL-3.0 hoặc Enterprise License. Nếu sản phẩm
closed-source hoặc thương mại, team pháp lý phải xác nhận Enterprise License trước
khi tích hợp sâu. RTMPose/MMPose và phần ST-GCN tham chiếu OpenMMLab dùng
Apache-2.0; vẫn cần lập SBOM và kiểm tra license của toàn bộ dependency/model/data.

## Cài đặt

Môi trường khuyến nghị: Python 3.11 hoặc 3.12. Workspace hiện tại dùng Python 3.13
và đã có Torch, Ultralytics, RTMLib, OpenCV.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Lần chạy đầu sẽ tải `yolo26s.pt` và RTMPose-s. RTMPose được cache dưới
`models/rtmlib`.

## Chạy inference

Video:

```powershell
.\.venv\Scripts\python.exe -m fall_detection `
  --source path\to\video.mp4 `
  --output-video outputs\result.mp4 `
  --event-log outputs\events.jsonl
```

Camera:

```powershell
.\.venv\Scripts\python.exe -m fall_detection --source 0 --show
```

Xem trực tiếp khi xử lý video:

```powershell
.\.venv\Scripts\python.exe -m fall_detection --source data\video.mp4 --show
```

Nhấn `q` hoặc `Esc` để dừng. `--display` vẫn được giữ làm alias tương thích.

Khi ST-GCN chưa được train, CLI sẽ in cảnh báo rõ ràng. Các ngưỡng trong
[`configs/default.yaml`](configs/default.yaml) chỉ là giá trị khởi tạo, không phải
ngưỡng production đã calibration.

## Dữ liệu ST-GCN

Mỗi sample là một file `.npz`:

```python
np.savez_compressed(
    "sample.npz",
    x=features,  # float32 [7, T, 17, 1]
    y=label,     # 0=normal, 1=falling, 2=lying
)
```

7 channel được tạo bởi `SkeletonHistory.model_input()`:

1. x tương đối với pelvis / chiều cao bbox
2. y tương đối với pelvis / chiều cao bbox
3. keypoint confidence
4. vận tốc x tương đối
5. vận tốc y tương đối
6. pelvis y / chiều cao frame
7. bbox width / height

Hai channel cuối giữ chuyển động toàn cục và tư thế nằm; nếu chỉ center/scale pose
từng frame, tín hiệu người đang hạ xuống sẽ bị toán học loại bỏ.

Tạo feature bằng manifest đã chia split:

```powershell
.\.venv\Scripts\python.exe -m fall_detection.dataset `
  --manifest data\manifest.csv `
  --output data\processed
```

Schema nằm tại [`data/manifest.example.csv`](data/manifest.example.csv).
`target_x,target_y` là tọa độ chuẩn hóa của người cần theo dõi ở đầu clip; nếu bỏ
trống tool chọn người có bbox lớn nhất. Với cảnh nhiều người, nên luôn annotation
target để tránh label sai người.

Trainer bắt buộc nhận hai thư mục tách biệt:

```powershell
.\.venv\Scripts\python.exe -m fall_detection.training `
  --train-data data\processed\train `
  --validation-data data\processed\validation `
  --output models\stgcn_fall.pt
```

Không release model chỉ dựa trên validation này. Cần một test set khóa, độc lập về
subject/camera/site, và calibration threshold trên validation riêng.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Test hiện có kiểm tra graph/model shape, resampling thời gian, nối ID sau ID switch,
fall đột ngột và nằm kéo dài.

## Việc cần làm trước production

1. Chốt license YOLO26 và quyền sử dụng từng dataset.
2. Thu dữ liệu pilot đúng camera/site, có consent và retention policy.
3. Xây annotation theo event: onset, impact/ground contact, lying, recovery.
4. Train + subject/camera-disjoint evaluation; đo false alarms theo camera-hour.
5. Export YOLO/RTMPose sang TensorRT nếu profiling chứng minh cần thiết.
6. Thêm camera health, audit log, alert deduplication, human acknowledgement và
   cơ chế fail-open/fail-closed được legal/safety phê duyệt.
