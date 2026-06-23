# Data and evaluation protocol

## Label contract

Khuyến nghị giữ ba phase thay vì nhãn fall/no-fall cho mọi frame:

- `normal`: đứng, đi, ngồi, cúi, quỳ và các hoạt động không phải sự cố.
- `falling`: đoạn chuyển trạng thái mất thăng bằng đến chạm sàn/bề mặt.
- `lying`: cơ thể đã nằm sau chuyển động; bao gồm cả hard negative nằm chủ động
  trong tập dữ liệu với metadata `intentional=true`.

Event-level ground truth cần `event_id`, `subject_id`, `camera_id`, `site_id`,
`onset_time`, `impact_time`, `lying_start`, `recovery_time`, occlusion và quality.

## Split

Ưu tiên:

1. Site-disjoint test nếu có nhiều cơ sở.
2. Camera-disjoint test trong cùng cơ sở.
3. Subject-disjoint ở mọi split.
4. Không cho các cửa sổ chồng lấn từ cùng event xuất hiện ở hai split.

## Release metrics

- Event sensitivity/recall.
- Event precision.
- False alarms per camera-hour, kèm khoảng tin cậy.
- Missed falls theo loại ngã và mức occlusion.
- Detection delay từ onset và từ impact: median, p90, p95.
- Performance theo camera, khoảng cách, ánh sáng, số người và subgroup hợp pháp.
- Tracking: IDF1/HOTA hoặc tối thiểu ID switches/event trên tập pilot.
- Pose quality/coverage trong các frame trước, trong và sau fall.

Frame accuracy, clip accuracy và AUROC có thể dùng trong phát triển nhưng không đủ
để quyết định release.

## Public data strategy

Dùng public datasets để pretrain/ablation, không dùng làm bằng chứng duy nhất cho
production. Gộp nhiều nguồn chỉ sau khi chuẩn hóa joint convention, FPS, phase label
và license. Fine-tune cuối cùng bằng dữ liệu pilot đúng domain; giữ test set khóa.

Các augmentation hữu ích:

- temporal speed 0.7x-1.3x;
- keypoint dropout theo limb và occlusion block;
- bbox/keypoint jitter phù hợp sai số RTMPose;
- horizontal flip với hoán đổi left/right;
- camera crop/scale và frame dropping;
- không dùng rotation/warp phi thực tế làm thay đổi hướng trọng lực.

