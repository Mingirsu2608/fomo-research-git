# Runbook

## Setup

Chạy các lệnh pipeline từ thư mục gốc repo.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r env/requirements.lock
```

## Cấu hình nguồn dữ liệu

Pipeline hiện chỉ tải hai CSV từ Google Drive; chưa có kết nối PostgreSQL.
Tạo file local nếu chưa có (lệnh này giữ nguyên file đã tồn tại):

```bash
[ -f configs/data.local.json ] || cp configs/data.local.example.json configs/data.local.json
```

Điền URL được cấp quyền vào hai giá trị trong `configs/data.local.json`:

```json
{
  "FOMO_MARKET_URL": "<URL CSV market trên Drive>",
  "FOMO_TRADES_URL": "<URL CSV trades trên Drive>"
}
```

Các giá trị trong ví dụ là chỗ giữ chỗ, phải thay bằng URL thật trước khi chạy.
File mẫu được commit; file local chứa URL thật được `.gitignore` loại trừ.
Có thể đặt biến môi trường `FOMO_MARKET_URL` và `FOMO_TRADES_URL` thay cho JSON.
Với từng nguồn, code ưu tiên biến môi trường có giá trị; nếu không có, đọc file
`configs/data.local.json` đầu tiên tìm thấy tại thư mục hiện hành hoặc thư mục cha.
Nếu file đầu tiên thiếu khóa, code báo lỗi chứ không tìm tiếp file khác.
Code không tự đọc `.env`.

`params.yaml` lưu tên biến qua `data.market_url_env`, `data.trades_url_env`,
đường dẫn output và tham số xử lý/model. `dvc.yaml` định nghĩa các stage;
`dvc.lock` lưu tham số cùng hash dependency/output. Không đặt URL thật vào các
file này. JSON và YAML đều có thể dùng cho cấu hình database nếu sau này code
bổ sung driver và cách đọc tương ứng; hiện chưa triển khai phần đó.

DVC không theo dõi thay đổi của file local hay giá trị biến môi trường.
Khi đổi URL hoặc muốn tải nội dung mới từ cùng URL, chạy:

```bash
dvc repro --force extract
dvc repro
```

## Chạy từng stage

```bash
python -m src.pipelines.extract
python -m src.pipelines.features
python -m src.pipelines.cluster
python -m src.pipelines.evaluate
```

Hoặc:

```bash
dvc repro
```

## MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Lỗi thường gặp

- `Set FOMO_MARKET_URL` hoặc `Set FOMO_TRADES_URL`: cấu hình biến môi trường hoặc khóa tương ứng trong file JSON local.
- `Invalid Google Drive URL`: kiểm tra URL có file ID dạng `/d/...` hoặc tham số `id=...`.
- Drive không tải được: kiểm tra Internet và quyền chia sẻ của hai file. Luồng hiện tại không đăng nhập OAuth.
- Schema thiếu: đối chiếu `docs/data_dictionary.md`.
- Model chưa có: chạy stage `cluster` trước `evaluate`.
- `dvc.lock` chưa tồn tại: chạy `dvc repro` lần đầu để DVC tạo hash thật.


## Kiểm tra tự động trong flow

Mỗi entrypoint (`pipelines.all` hoặc từng stage qua DVC) kiểm tra output đã ghi
trước khi báo `status=PASS`. Nếu xử lý hoặc kiểm tra lỗi, log `status=FAIL` kèm
traceback, trả exit code khác 0 và dừng trước stage tiếp theo.

- Extract: manifest/SHA256, dữ liệu không rỗng, schema, timestamp, lịch market,
  investor ID và side hợp lệ.
- Features: đủ cột, không rỗng/trùng investor-window, không infinity, độ dài
  cửa sổ và điều kiện số lệnh/ngày mua đúng cấu hình. Missing feature được phép
  trước phễu lọc.
- Trong cluster: feature funnel, complete-case, scaling và PCA có log PASS/FAIL
  riêng; kiểm tra tương quan còn lại, ma trận hữu hạn, chuẩn hóa và phương sai PCA.
- Cluster: nạp lại artifacts đã lưu, kiểm tra feature/report, chiều PCA, đủ tổ
  hợp tuning, score hữu hạn và model được lưu đúng K tốt nhất.
- Evaluate: metric hữu hạn/đúng phạm vi; tổng số hàng, tỷ lệ cụm và profile khớp
  cohort và danh sách model.

Ví dụ log thành công:

```text
stage=extract status=PASS
stage=features status=PASS
step=feature_funnel status=PASS
step=complete_case status=PASS
step=scaling status=PASS
step=pca status=PASS
stage=cluster status=PASS
stage=evaluate status=PASS
pipeline status=PASS
```

`PASS` xác nhận các kiểm tra kỹ thuật, không đảm bảo chất lượng phân cụm tốt.
DVC bỏ qua stage không thay đổi nên stage đó không chạy lại kiểm tra và không
in PASS mới. Để bắt buộc chạy lại tất cả, dùng `dvc repro --force` hoặc
`python -m src.pipelines.all` (cả hai đều tải lại raw data).

Chạy bộ regression tests độc lập trước pipeline:

```bash
python -m pytest -q
```

## Notebook và chia sẻ repo

Notebook nhúng code và tham số nghiên cứu, không import `src` hay đọc `params.yaml`.
URL vẫn phải được cung cấp qua biến môi trường của kernel hoặc JSON local theo
quy tắc trên. Với kernel chạy từ `notebooks/`, file local ở gốc repo được tìm qua
thư mục cha. Nếu chỉ mang notebook sang máy khác, cần thiết lập lại nguồn riêng.

Chạy Restart Kernel → Run All để tạo lại bảng và biểu đồ trong bộ nhớ.
Notebook không xuất model/báo cáo ra file, nhưng khi lưu notebook sau khi chạy,
output hiển thị có thể được lưu bên trong `.ipynb`. Trước khi commit, dùng
Clear All Outputs rồi lưu; bản notebook hiện tại đã được xóa output.

Kiểm tra trước khi stage:

```bash
git check-ignore configs/data.local.json
git status --short
git diff --cached
```

Các commit cũ còn chứa URL và output notebook; thay đổi cấu hình hiện tại
không xóa lịch sử đó. Push cùng lịch sử cũ vẫn có thể công khai các nội dung này.

Dockerfile dùng `COPY . /app`; `.gitignore` không kiểm soát Docker build context.
Nếu build image, cần tự loại cấu hình riêng khỏi build context (ví dụ qua
`.dockerignore`) và cấp URL cho container lúc chạy; không chia sẻ image đã chứa
file local hoặc dữ liệu riêng.
