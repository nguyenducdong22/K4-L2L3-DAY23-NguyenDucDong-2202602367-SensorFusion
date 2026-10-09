# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Đức Đông
- MSSV: 2202602367
- Email: dong160805@gmail.com
- Link repo (fork): https://github.com/nguyenducdong22/K4-L2L3-DAY23-NguyenDucDong-2202602367-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): commit `CP6` mới nhất trên nhánh `main` (hash 40 ký tự nộp trên LMS)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]` (199 frame), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `seed = 0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision 0.9701, recall 0.7004, tp/fp/fn = 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.344, 0, 239, 2.523
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.267, 0, 239, 2.523
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  - Hai mode có **cùng** detection (519/16/222), cùng `matches = 502`, `ghost_track_frames = 0`, `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.523`. Tập track confirmed và các cặp ghép với GT giống hệt nhau, vì vòng đời track (init/score/confirm/delete) chỉ do LiDAR quyết định; camera không tạo, không xoá, không đổi score.
  - Do số cặp như nhau nên so RMSE là công bằng: fused giảm `sum_sq_err` từ 11.344 xuống 9.267 (−18%), RMSE từ 0.1503 xuống 0.1359 m (−0.0145 m). Camera chỉ tinh chỉnh trạng thái EKF, chủ yếu theo phương ngang/đứng của ảnh (vuông góc trục nhìn), nên sai số vị trí giảm. `rmse_fused − rmse_lidar = −0.0145 m ≤ 0.05 m`, tức camera không làm tracking xấu đi.
  - `precision_track = 502/(502+0) = 1.00` (≥ 0.75), `coverage = 502/519 = 0.967` (≥ 0.70). 239 frame-GT bị miss phần lớn do chính detector bỏ sót (fn = 222). 17 detection TP còn lại chưa thành track confirmed, chủ yếu là những frame đầu khi track mới khởi tạo chưa đủ score (cần > 0.8, tức ≥ 5 hit LiDAR).
  - Giới hạn: đo camera là tâm hộp 2D **ground-truth** FRONT cộng nhiễu seeded (σ = 0.5 px khi sinh, R dùng σ = 5 px). Mức cải thiện RMSE vì vậy là cận trên, chưa phản ánh một camera detector thật. Camera FRONT chỉ phủ một góc nhìn, nên track ngoài FOV không được update camera.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   LiDAR: `z = (x, y, z)` (m) là tâm hộp 3D trong hệ cảm biến. `h(x) = R_vs·p + t` tuyến tính nên `H` là ma trận 3×6 hằng (khối xoay, cột vận tốc = 0). `R = diag(0.1², 0.1², 0.1²)` m². Camera: `z = (u, v)` (pixel), `h(x)` là phép chiếu pinhole phi tuyến `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (`camera_fusion.camera_measurement_prediction`). `H` là Jacobian 2×6 tính lại tại trạng thái dự đoán (EKF). `R = diag(5², 5²)` px² (`build_camera_measurement`). Camera không đo được độ sâu: một đo 2D chỉ ràng buộc hướng nhìn. Vì vậy camera chỉ có thể tinh chỉnh track đã có từ LiDAR, không đủ để khởi tạo track 3D.
2. Vì sao cần gating Mahalanobis trước khi gán?

   `d² = γᵀS⁻¹γ` chuẩn hoá residual theo độ bất định thật của cặp track–đo (`S = HPHᵀ + R`, gồm cả P của track lẫn nhiễu cảm biến). Dưới giả thiết Gauss, d² ~ χ² với bậc tự do = `dim_meas`, nên ngưỡng `chi2.ppf(0.995, 3)` ≈ 12.84 cho LiDAR và `chi2.ppf(0.995, 2)` ≈ 10.60 cho camera giữ 99.5% đo đúng mà loại các cặp phi lý. Không có gate, bước greedy vẫn sẽ ghép "cặp ít tệ nhất" (ví dụ một FP với track ở xa) và update làm kéo lệch EKF. Đo không qua gate được giữ lại làm unassigned, nếu là LiDAR thì có thể sinh track mới. Trong `association_cost_matrix`, mình kiểm tra `in_fov` **trước** khi tính Mahalanobis, để không chiếu điểm sau camera/độ sâu ≤ 0.
3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   **Track-then-fuse.** Chỉ có một `TrackManager`. Mỗi frame trong `run_lab.run`: `KF.predict` cho mọi track → `associate_and_update(..., lidar_sensor)` (AssocL: update + score/init/delete) → nếu mode `fused` thì `associate_and_update(..., camera_sensor)` (AssocC: chỉ EKF update) → ghi một record vào `grade_run_<mode>.log`. Dấu vết trên log: `grade_run.log` có 398 record = 199 frame × 2 mode; ở **mọi** frame hai mode có cùng `det_tp/det_fp/det_fn` và cùng `confirmed/matches/ghosts/misses`. Tổng trong `metrics.json`: `matches = 502`, `mean_confirmed_tracks = 2.523` như nhau, chỉ `sum_sq_err` khác (11.344 so với 9.267). Nếu là fuse-then-track (gộp đo trước rồi mới track), tập track và số ghép sẽ thay đổi theo camera.
4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Innovation camera `γ = z − h(x)` sẽ **có bias hệ thống** (trung bình khác 0, cùng một hướng pixel, ví dụ lệch ngang khi sai yaw extrinsic) thay vì nhiễu trắng quanh 0. NIS `γᵀS⁻¹γ` trung bình vượt `dim_meas = 2`, tỉ lệ cặp bị gate loại tăng. Mức lệch còn phụ thuộc khoảng cách: với lệch góc, pixel offset gần như hằng; với lệch tịnh tiến, offset lớn hơn ở xe gần. Lệch nhỏ (vẫn qua gate) sẽ kéo vị trí track về phía sai, làm RMSE fused cao hơn LiDAR và residual LiDAR ngay sau đó tăng (hai cảm biến "giằng co"). Lệch lớn thì gate χ² chặn phần lớn đo camera, fused suy biến về LiDAR-only.
5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi `meas_list` rỗng thì không thể suy ra cảm biến từ `meas.sensor`. Nhưng frame LiDAR rỗng vẫn phải **trừ score** cho các track trong FOV và xoá track hết score, nếu không ghost sẽ sống mãi. Vì thế hàm luôn gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` với sensor tường minh. `TrackManager` dùng `sensor.name` để biết đây là lượt LiDAR (chấm miss, xoá, sinh track) hay lượt camera (`return` ngay). LiDAR đo 3D đầy đủ, có độ tin cậy về sự tồn tại và vị trí, nên là nguồn duy nhất để init (cần `x, y, z`) và quyết định score. Camera chỉ phủ FOV FRONT, không có độ sâu, và đo trong lab là từ nhãn GT. Nếu camera được cộng/trừ score, track ngoài FOV camera sẽ bị phạt sai, và nhãn GT sẽ "rò" vào quyết định tồn tại. Vì vậy camera chỉ refine trạng thái EKF (`filter_obj.update`), còn `handle_updated_track` bỏ qua camera.
6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   Khởi tạo: `score = 1/window = 1/6`, `state = "initialized"`, `x` = vị trí đo đổi sang hệ xe với vận tốc 0, `P` = khối vị trí `R_sv·R·R_svᵀ` cộng `σ_p44² = σ_p55² = 50²`, `σ_p66² = 5²`. Mỗi lượt LiDAR: hit → `score = min(1, score + 1/6)`; miss **trong FOV LiDAR** → `score −= 1/6`. Track chưa confirmed có hit thì thành `tentative`. **Xác nhận** khi `score > confirmed_threshold = 0.8` (≥ 5 hit liên tiếp kể từ lúc init). Track đã `confirmed` **không bị hạ** trạng thái khi miss: một miss từ 1.0 còn 0.833, vẫn confirmed. **Xoá** (`should_delete_track`, điều kiện OR): `P[0,0] > max_P` hoặc `P[1,1] > max_P` (9 m²), bất kể score; hoặc confirmed và `score < delete_threshold = 0.6` (≥ 3 miss liên tiếp từ 1.0); hoặc chưa confirmed và `score ≤ 0`. Chỉ lượt LiDAR mới đánh giá xoá.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (Anthropic) qua Claude Cowork
- Dùng cho phần nào (hàm, câu hỏi, debug): hỗ trợ viết các hàm Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`), dựng notebook Colab để tải data từ Drive khoá học và chạy `fusion-run-lab --fusion compare --seed 0`, debug lỗi chọn nhầm file weights (darknet thay cho FPN-ResNet), soạn nháp phần giải thích trong báo cáo
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` trên Colab: 128 passed, không có failed/xfailed; chạy Waymo frame 0–198 seed 0 (`validate_metrics_records` hợp lệ); đối chiếu công thức F, Q, S, K, chi² và pinhole với `docs/HUONG_DAN_KY_THUAT.md`, các con số trong báo cáo lấy trực tiếp từ `metrics.json`

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
