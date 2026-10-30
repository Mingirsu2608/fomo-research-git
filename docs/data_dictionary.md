# Data dictionary

## Nguồn và cấu hình

Hai CSV đầu vào được tải từ Google Drive. URL riêng đọc từ biến môi trường
`FOMO_MARKET_URL`, `FOMO_TRADES_URL` hoặc `configs/data.local.json` được Git ignore;
không nằm trong `params.yaml`. Xem [runbook](runbook.md#cấu-hình-nguồn-dữ-liệu).
Chưa có phần đọc PostgreSQL.

## Raw market

| Column | Ý nghĩa |
|---|---|
| `timestamp` | Thời điểm quan sát giá. |
| `open`, `high`, `low`, `close` | Giá OHLC. |
| `volume` | Khối lượng giao dịch; cột bắt buộc cho feature volume surge. |

## Raw trades

| Column | Ý nghĩa |
|---|---|
| `order_id` | Định danh lệnh; cột bắt buộc. |
| `timestamp` | Thời điểm đặt lệnh. |
| `investor_id` | Định danh nhà đầu tư. |
| `side` | BUY hoặc SELL. |
| `price`, `quantity`, `amount`, `fee` | Giá, khối lượng, giá trị và phí; các cột bắt buộc. |
| `cash_after` | Số dư tiền sau lệnh; dùng suy ra tiền trước lệnh và mức sử dụng tiền. |

## Investor features

Mỗi dòng biểu diễn một nhà đầu tư trong một cửa sổ thời gian, định danh bằng
`investor_id`, `window_start`, `window_end`. Mặc định cửa sổ 180 ngày, bước 30 ngày,
có ít nhất 5 lệnh mua và 3 ngày mua; các ngưỡng được đọc từ `params.yaml`.
Bảng có `observed_days`, số lệnh/ngày hoạt động và 32 feature ứng viên trong
`CANDIDATE_32` của [`src/features/build.py`](../src/features/build.py).

Các feature mô tả phản ứng mua/bán theo rally, đỉnh giá, dip, mức sử dụng tiền,
nhịp giao dịch và thoát vị thế. Tín hiệu market dùng thông tin đến ngày t-1.
Feature có thể thiếu khi không đủ lịch sử hoặc ngữ cảnh; không được có infinity.
Stage cluster chọn feature qua coverage/dispersion/tương quan, chỉ giữ các
cửa sổ đủ giá trị hữu hạn trên tập feature đã chọn trước scaling/PCA.

CSV và kết quả sinh ra nằm trong `artifacts/`, được Git ignore và DVC quản lý.
Manifest raw lưu `as_of_date`, đường dẫn và SHA256; `as_of_date` là metadata,
chưa tự giới hạn dữ liệu theo ngày. Notebook có thể hiển thị dữ liệu mẫu;
xóa output trước khi lưu bản dùng để chia sẻ.
