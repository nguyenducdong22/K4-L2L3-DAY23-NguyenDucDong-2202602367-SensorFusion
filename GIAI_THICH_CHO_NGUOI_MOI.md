# Giải thích Lab Day 23 — Sensor Fusion cho người mới

> Viết cho người chưa biết gì về xe tự hành hay bộ lọc Kalman. Đọc từ trên xuống.

---

## 1. Bài lab này để làm gì? (một câu)

**Dạy máy tính "nhìn" và "theo dõi" các xe xung quanh một chiếc xe tự hành, bằng cách kết hợp hai loại cảm biến: LiDAR và camera.**

Tưởng tượng bạn đang lái xe. Bạn không chỉ cần biết *"có một chiếc xe ở kia"*, mà còn cần biết:

- nó **ở đâu** (cách mình bao nhiêu mét),
- nó **đang đi về hướng nào, nhanh cỡ nào**,
- và chiếc xe ở giây này **có phải chính là** chiếc xe ở giây trước không.

Lab này xây đúng cái "bộ não" làm việc đó.

---

## 2. Hai "con mắt" của xe: LiDAR và Camera

| | LiDAR | Camera |
|---|---|---|
| Hoạt động thế nào | Bắn tia laser ra xung quanh, đo thời gian phản xạ → ra **khoảng cách** | Chụp ảnh như điện thoại |
| Cho ta biết | Vị trí **3D** thật (x, y, z tính bằng mét) | Vị trí **2D** trên ảnh (u, v tính bằng pixel) |
| Điểm mạnh | Biết chính xác vật **cách bao xa** | Rất sắc nét theo chiều ngang/dọc, nhìn được chi tiết |
| Điểm yếu | Điểm thưa, đôi khi bỏ sót | **Không biết độ sâu** — chỉ biết vật nằm "trên tia nhìn nào" |

👉 Mỗi cảm biến có điểm yếu riêng. **Sensor fusion (hợp nhất cảm biến)** = dùng cái mạnh của bên này bù cái yếu của bên kia.

---

## 3. "Phát hiện" (detection) khác "theo dõi" (tracking) thế nào?

- **Detection** — mỗi khung hình (frame), một mạng nơ-ron (FPN-ResNet, có sẵn trong lab) nhìn đám điểm LiDAR và nói: *"ở đây có 3 chiếc xe"*. Nhưng nó **không nhớ gì** giữa các frame, đôi khi bỏ sót, đôi khi báo nhầm.
- **Tracking** — nối các phát hiện qua thời gian thành **"track"** (đường đi của một chiếc xe), gắn cho mỗi xe một **ID**, ước lượng cả **vận tốc**, và làm mượt nhiễu.

Ví dụ: detector nói frame 10 có xe ở (20 m, 3 m), frame 11 có xe ở (21 m, 3 m). Tracker hiểu: *"đây là cùng xe số #5, đang đi tới với tốc độ ~10 m/s"*.

**Phần bạn viết trong lab chính là cái tracker (Part E–H).**

---

## 4. Toàn bộ pipeline (dây chuyền) trong một hình

```text
 Đám mây điểm LiDAR ──► ảnh nhìn từ trên xuống (BEV) ──► mạng FPN-ResNet ──► các hộp 3D
                                                                             │  (Part A–D: có sẵn)
                                                                             ▼
 MỖI FRAME (0.1 giây):
   ① DỰ ĐOÁN   — mỗi track đoán xe đã đi tới đâu            (Part E: Kalman)
   ② GHÉP LiDAR — đo nào thuộc track nào?  rồi CẬP NHẬT      (Part F: association)
   ③ GHÉP Camera — tinh chỉnh thêm vị trí bằng ảnh           (Part F + G: camera)
   ④ QUẢN LÝ   — tạo track mới, xác nhận, xoá track cũ       (Part H)
                                                                             │
                                                                             ▼
                         metrics.json: sai số (RMSE), số track ma (ghost), số xe bị sót (miss)
```

---

## 5. Nguyên lý từng phần — giải thích bằng ví dụ đời thường

### Part E — Bộ lọc Kalman: "đoán rồi sửa"

Hãy tưởng tượng bạn nhắm mắt đi trong phòng tối:

1. **Đoán (predict):** *"Mình đi thẳng 1 bước/giây, vậy sau 1 giây chắc mình ở cách cửa 1 mét."* — nhưng càng đi lâu càng không chắc (độ bất định tăng).
2. **Sửa (update):** Bạn chạm tay vào tường → *"À, thật ra mình lệch sang trái một chút"*. Bạn **không tin hoàn toàn** cái chạm (tay có thể sai) cũng **không tin hoàn toàn** cái đoán, mà **trộn** hai cái theo mức độ tin cậy.

Đó chính là bộ lọc Kalman. Trong lab:

- **Trạng thái** của một xe: `x = (px, py, pz, vx, vy, vz)` — 3 số vị trí + 3 số vận tốc.
- **Ma trận F** (mô hình chuyển động): *vị trí mới = vị trí cũ + vận tốc × 0.1 giây*.
- **Ma trận P**: "mình không chắc chắn đến mức nào". Mỗi lần đoán, P **to ra** (cộng thêm Q — nhiễu quá trình, vì xe có thể phanh/tăng tốc bất ngờ).
- **Khi có đo z:** tính **sai lệch** `γ = z − h(x)` (đo thực tế trừ đi cái mình kỳ vọng đo được).
  **Hệ số Kalman K** quyết định tin đo bao nhiêu:
  - đo rất chính xác (R nhỏ) → K lớn → kéo trạng thái về phía đo nhiều;
  - đo rất nhiễu (R lớn) → K nhỏ → gần như giữ dự đoán.
- **"Mở rộng" (Extended — EKF):** camera đo theo kiểu phi tuyến (chiếu 3D → 2D, có phép chia), nên phải dùng **đạo hàm (Jacobian H)** để xấp xỉ tuyến tính tại điểm hiện tại.

### Part G — Mô hình camera: "chiếu điểm 3D lên ảnh"

Camera giống như **lỗ kim** (pinhole): một điểm 3D cách camera `x_s` mét về phía trước, lệch `y_s` mét sang trái, `z_s` mét lên trên sẽ hiện lên ảnh tại:

```text
u = c_i − f_i · y_s / x_s        (cột pixel)
v = c_j − f_j · z_s / x_s        (hàng pixel)
```

- `f` = tiêu cự (độ "zoom"), `c` = tâm ảnh.
- Chia cho `x_s` → vật càng xa trông càng nhỏ, càng gần tâm ảnh.
- **Kiểm tra trước khi chiếu:** nếu xe ở **sau lưng** camera (`x_s ≤ 0`) thì phép chia vô nghĩa → phải từ chối. Và chỉ dùng camera khi xe nằm trong **góc nhìn (FOV)** của nó.

### Part F — Ghép đo với track: "cái này của ai?"

Mỗi frame có vài track và vài đo. Câu hỏi: **đo nào thuộc track nào?**

- **Khoảng cách Mahalanobis** — không dùng khoảng cách thẳng (mét) mà dùng khoảng cách **"chia cho độ không chắc chắn"**.
  Ví dụ: lệch 1 m với một track rất chắc chắn là **đáng ngờ**; lệch 1 m với một track đang rất mơ hồ là **bình thường**.
- **Cổng χ² (gating)** — nếu khoảng cách quá lớn (vượt ngưỡng thống kê 99.5%), **cấm ghép**, tránh việc một phát hiện nhầm kéo lệch track.
- **Ghép tham lam (greedy)** — lặp lại: chọn cặp có khoảng cách nhỏ nhất → ghép → gạch hàng và cột đó đi → tiếp tục.

### Part H — Vòng đời track: "sinh, xác nhận, chết"

Mỗi track có một **điểm tin cậy (score)** từ 0 đến 1:

| Sự kiện | Điều gì xảy ra |
|---|---|
| Có đo LiDAR mới không thuộc track nào | **Sinh** track mới, score = 1/6, trạng thái `initialized` |
| LiDAR thấy lại track (hit) | score **+1/6** (tối đa 1) → `tentative` |
| score > 0.8 (≈ 5 lần thấy liên tiếp) | **Xác nhận** → `confirmed` (chỉ track này mới được tính điểm) |
| LiDAR **không** thấy track trong tầm nhìn (miss) | score **−1/6** |
| Track confirmed mà score < 0.6, hoặc track chưa xác nhận score ≤ 0, hoặc P quá lớn (> 9 m²) | **Xoá** |

**Vì sao chỉ LiDAR được quyết định sinh/xoá, camera thì không?**

- LiDAR có đủ **3D** để tạo một track; camera không biết độ sâu nên không thể tạo track.
- Camera chỉ nhìn **phía trước**; nếu camera được trừ điểm, mọi xe ở phía sau sẽ bị xoá oan.
- Vì vậy camera chỉ được **tinh chỉnh vị trí** (cập nhật Kalman), không đụng vào score. Đây gọi là thiết kế **track-then-fuse**: *theo dõi bằng LiDAR trước, rồi dùng camera để làm mịn.*

---

## 6. Đọc kết quả của bài này

Chạy trên 199 frame (~20 giây lái xe thật) của bộ dữ liệu **Waymo**:

| Chỉ số | Chỉ LiDAR | LiDAR + Camera | Ý nghĩa |
|---|---|---|---|
| **RMSE** (sai số vị trí trung bình) | 0.150 m | **0.136 m** | Càng nhỏ càng tốt. ~14–15 cm — rất chính xác (ngưỡng đạt: ≤ 0.45 m) |
| **matches** (số lần track khớp xe thật) | 502 | 502 | Như nhau, vì camera không tạo/xoá track |
| **ghost** (track "ma" — không có xe thật) | 0 | 0 | Hoàn hảo: không báo nhầm |
| **missed** (xe thật nhưng không có track) | 239 | 239 | Phần lớn do detector bỏ sót ngay từ đầu (fn = 222) |

**Kết luận:** thêm camera giúp **giảm sai số ~10% (từ 15.0 cm xuống 13.6 cm)** mà **không làm tăng** track ma hay xe bị sót.

⚠️ **Lưu ý trung thực:** "đo camera" trong lab **không phải** từ một mạng nhận diện ảnh thật mà là **tâm hộp 2D đúng (ground-truth) + nhiễu ngẫu nhiên**. Nên con số cải thiện ở đây là lạc quan hơn thực tế.

---

## 7. Từ điển nhanh

| Thuật ngữ | Nghĩa đơn giản |
|---|---|
| **Sensor fusion** | Kết hợp dữ liệu nhiều cảm biến để ra kết quả tốt hơn từng cái |
| **LiDAR** | Cảm biến laser đo khoảng cách, cho đám điểm 3D |
| **BEV** (Bird's-Eye View) | Ảnh nhìn từ trên trời xuống, tạo từ đám điểm LiDAR |
| **Detection** | Phát hiện vật trong *một* frame |
| **Track** | Một vật được theo dõi qua *nhiều* frame, có ID |
| **Kalman filter / EKF** | Thuật toán "đoán rồi sửa", trộn dự đoán với đo theo độ tin cậy |
| **State `x`** | Những gì ta muốn biết: vị trí + vận tốc |
| **Covariance `P`** | Mức độ "không chắc chắn" về state |
| **Innovation `γ`** | Đo thực tế − đo dự đoán (độ "bất ngờ") |
| **Mahalanobis distance** | Khoảng cách đã tính đến độ không chắc chắn |
| **Gating (χ²)** | Ngưỡng thống kê để loại cặp ghép vô lý |
| **FOV** | Góc nhìn của cảm biến |
| **RMSE** | Sai số trung bình (căn bậc hai của trung bình bình phương) |
| **Ghost track** | Track không ứng với vật thật nào |
| **Track-then-fuse** | Theo dõi bằng một cảm biến chính, cảm biến phụ chỉ tinh chỉnh |

---

## 8. Muốn hiểu sâu hơn thì đọc gì?

1. File code bạn đã viết: `student/workspace/kalman.py` → `camera_fusion.py` → `association.py` → `track_management.py` (theo đúng thứ tự trên).
2. `docs/HUONG_DAN_KY_THUAT.md` mục 2 — sơ đồ pipeline chi tiết.
3. Video tìm trên YouTube: *"Kalman filter explained simply"* — xem 10 phút là nắm được ý "đoán rồi sửa".
4. Sách: Thrun, Burgard, Fox — *Probabilistic Robotics*, chương 3.
