# Cây thư mục và công dụng các thành phần trong repo

Repo `fomo-research-git` dùng để nghiên cứu và chạy pipeline phân cụm hành vi
nhà đầu tư. Notebook phục vụ khám phá và thử nghiệm; `src/` chứa pipeline có
thể chạy lại, kiểm tra output và quản lý phiên bản kết quả bằng DVC.

## 1. Cây thư mục

```text
fomo-research-git/
├── README.md                      # Giới thiệu và hướng dẫn chạy nhanh
├── FILE_GUIDE.md                  # Tài liệu cây thư mục này
├── params.yaml                    # Tham số dữ liệu, feature, preprocessing, model
├── dvc.yaml                       # Định nghĩa các stage và dependency
├── dvc.lock                       # Hash dữ liệu/output và tham số của lần chạy
├── pyproject.toml                 # Metadata Python và cấu hình pytest
├── .gitignore                     # Những file Git không theo dõi
├── .dvcignore                     # Những đường dẫn DVC bỏ qua khi duyệt
├── .git/                          # Lịch sử commit, branch và cấu hình Git
├── .dvc/                          # Cấu hình, cache và trạng thái nội bộ DVC
├── configs/
│   ├── data.local.example.json    # Mẫu cấu hình nguồn dữ liệu, không chứa URL thật
│   └── data.local.json            # URL riêng trên máy, được Git ignore
├── env/
│   ├── requirements.lock          # Danh sách thư viện và ràng buộc phiên bản
│   └── Dockerfile                 # Đóng gói môi trường chạy pipeline
├── notebooks/
│   └── baseline_notenook.ipynb     # Notebook nghiên cứu độc lập
├── src/
│   ├── data/
│   │   └── extract.py             # Tải dữ liệu và tạo manifest/SHA256
│   ├── features/
│   │   └── build.py               # Tính 32 feature theo investor-window
│   ├── models/
│   │   ├── preprocess.py          # Phễu lọc, complete-case, scaling và PCA
│   │   └── train.py               # Tuning K, chọn và lưu model; log MLflow
│   ├── evaluation/
│   │   └── evaluate.py            # Metric, quy mô và profile các cụm
│   └── pipelines/
│       ├── all.py                 # Chạy toàn bộ pipeline theo thứ tự
│       ├── extract.py             # Entrypoint stage extract
│       ├── features.py            # Entrypoint stage features
│       ├── cluster.py             # Entrypoint stage cluster
│       ├── evaluate.py            # Entrypoint stage evaluate
│       ├── common.py              # Đọc config, log và chạy bước có kiểm tra
│       └── validation.py          # Kiểm tra output từng stage, PASS/FAIL
├── tests/
│   ├── test_data.py               # Chuyển URL, cấu hình riêng và kiểm tra URL không lọt vào file Git
│   ├── test_features.py           # Kiểm tra schema và hợp đồng 32 feature
│   ├── test_preprocessing.py      # Kiểm tra phễu lọc, scaling/PCA và lưu/nạp
│   ├── test_pipeline.py           # Kiểm tra tham số phục vụ tái lập
│   ├── test_stage_validation.py   # Kiểm tra stage, lỗi và cơ chế dừng
│   └── test_notebook_standalone.py # Kiểm tra notebook độc lập và khớp logic src
├── docs/
│   ├── runbook.md                 # Cách vận hành và xử lý lỗi
│   ├── data_dictionary.md         # Ý nghĩa/schema dữ liệu
│   ├── model_card.md              # Mục tiêu, phương pháp và giới hạn model
│   └── acceptance_evidence.md     # Bằng chứng nghiệm thu và snapshot kết quả
├── artifacts/                     # Dữ liệu và kết quả do pipeline sinh ra
│   ├── raw/                       # CSV đầu vào và manifest
│   ├── features/                  # Bảng feature theo investor-window
│   ├── models/                    # Model, preprocessing và báo cáo tuning
│   └── evaluation/                # Metric và thống kê cụm
├── mlflow.db                      # Database tracking MLflow cục bộ
├── mlruns/                        # Có thể được MLflow tạo khi chạy cục bộ
├── .venv/                         # Môi trường Python cục bộ
└── .pytest_cache/                 # Cache do pytest tạo
```

Cây trên lược bỏ `__init__.py`, `__pycache__/` và các file cache để dễ đọc.
`__init__.py` đánh dấu các package Python. Các thư mục kết quả/cache có thể
chưa tồn tại khi vừa clone repo; chúng xuất hiện sau khi setup hoặc chạy code.

## 2. Công dụng các folder

| Folder | Vai trò | Khi nào cần chỉnh/xem? |
| --- | --- | --- |
| `configs/` | File mẫu và cấu hình nguồn dữ liệu riêng trên máy; file local được Git ignore. | Khi thiết lập URL Google Drive. Pipeline cũng nhận URL qua biến môi trường; chưa có phần kết nối PostgreSQL. |
| `env/` | Khai báo thư viện và môi trường Docker. | Khi cài môi trường hoặc thêm dependency. `requirements.lock` hiện dùng khoảng phiên bản, chưa khóa chính xác mọi dependency. |
| `notebooks/` | Nghiên cứu EDA, cửa sổ, feature, preprocessing, tuning và diễn giải cụm. | Khi thử ý tưởng trước khi đưa vào pipeline. Notebook nhúng code/config, giữ kết quả trong bộ nhớ, không xuất model và không dùng MLflow. |
| `src/data/` | Thu thập dữ liệu đầu vào, lưu raw CSV và thông tin truy vết. | Khi đổi cách tải dữ liệu hoặc tạo snapshot. |
| `src/features/` | Định nghĩa công thức và tạo 32 feature trên cửa sổ 180 ngày. | Khi sửa hoặc bổ sung feature. |
| `src/models/` | Chọn feature, chuẩn hóa, giảm chiều và phân cụm. | Khi đổi ngưỡng phễu lọc, preprocessing hoặc thuật toán. Ngưỡng/K thường chỉnh qua `params.yaml`. |
| `src/evaluation/` | Đánh giá và diễn giải model đã train. | Khi thêm metric hoặc thay đổi báo cáo cụm. |
| `src/pipelines/` | Nối các bước thành flow chạy và kiểm tra output. | Khi thêm stage, đổi thứ tự hoặc bổ sung kiểm tra PASS/FAIL. |
| `tests/` | Regression tests dùng dữ liệu nhỏ để phát hiện lỗi code. | Khi thay đổi logic hoặc thêm chức năng. Chạy bằng `python -m pytest -q`. |
| `docs/` | Tài liệu vận hành, dữ liệu và model. | Khi tìm hiểu repo hoặc cập nhật phương pháp/giới hạn. |
| `artifacts/` | Dữ liệu, model và báo cáo của pipeline thực tế. | Khi xem kết quả chạy. Git bỏ qua thư mục này; các output được khai báo trong DVC. |
| `.dvc/` | Cấu hình và dữ liệu nội bộ của DVC. | Khi cấu hình DVC remote. Cache do DVC quản lý, không chỉnh tay. |
| `.git/` | Cơ sở dữ liệu lịch sử của Git. | Quản lý qua lệnh Git; không chỉnh nội dung thủ công. |
| `.venv/`, `.pytest_cache/`, `__pycache__/` | Môi trường Python và cache cục bộ. | Được công cụ tạo ra; không commit vào Git. |

## 3. Flow chạy và nơi nhận output

```text
pytest (chạy riêng trước pipeline)
  ↓
extract → kiểm tra → PASS/FAIL
  ↓
features → kiểm tra → PASS/FAIL
  ↓
feature funnel → complete-case → StandardScaler → PCA
  ↓
cluster → kiểm tra → PASS/FAIL
  ↓
evaluate → kiểm tra → PASS/FAIL
```

`pipelines.all` và các entrypoint của DVC dùng chung kiểm tra runtime. Khi lỗi,
stage báo `FAIL`, hiển thị nguyên nhân và dừng trước stage kế tiếp. Bộ `pytest`
chạy riêng, không tự chạy mỗi lần gọi pipeline.

| Bước | Input | Output chính |
| --- | --- | --- |
| Extract | Hai URL Google Drive qua biến môi trường hoặc JSON local được ignore | `artifacts/raw/btc_market_ohlcv.csv`, `btc_trade_log.csv`, `manifest.json` |
| Features | Hai raw CSV | `artifacts/features/investor_window_32features.csv` |
| Cluster | Bảng feature và cấu hình preprocessing/model | `artifacts/models/preprocessor.joblib`, `features.txt`, `feature_selection.csv`, `pca_variance.csv`, `tuning_results.csv` và ba model `.joblib` |
| Evaluate | Feature, danh sách feature được chọn, preprocessing và model đã lưu | `artifacts/evaluation/model_metrics.json`, `cluster_sizes.csv`, `cluster_profiles.csv` |

Các model hiện tại là KMeans, MiniBatchKMeans và GMM. Pipeline trong `src`
log thông tin thí nghiệm vào MLflow; notebook phục vụ nghiên cứu trực tiếp
trong bộ nhớ. Metric evaluation hiện được tính trên cohort training.

## 4. Git và DVC quản lý những gì?

| Thành phần | Cách quản lý |
| --- | --- |
| Code, notebook đã xóa output, tests, docs và cấu hình dùng chung | Commit/push bằng Git. |
| `configs/data.local.json` và thông tin truy cập riêng | Git ignore; chỉ giữ trên máy hoặc cấp qua biến môi trường. |
| `dvc.yaml`, `params.yaml`, `dvc.lock`, cấu hình DVC dùng chung | Commit bằng Git để truy vết flow và phiên bản chạy. |
| Raw data, feature, model và báo cáo được khai báo trong `dvc.yaml` | DVC quản lý; dùng `dvc push`/`dvc pull` khi đã cấu hình remote. |
| `.venv/`, cache, `mlflow.db`, `mlartifacts/` | Git đang bỏ qua theo `.gitignore`. Không được sao lưu tự động chỉ bằng `git push`. |
| `mlruns/` | Có thể xuất hiện từ MLflow; hiện chưa có quy tắc bỏ qua riêng trong `.gitignore`. Kiểm tra `git status` trước khi stage. |

`dvc.lock` chứa hash/tham số, không chứa toàn bộ dữ liệu hay model. Commit file
lock không thay thế việc sao lưu các artifact lên DVC remote. Các commit cũ
vẫn có thể chứa URL và output notebook, dù phiên bản hiện tại đã loại bỏ.

## 5. Các lệnh thường dùng

Chạy từ thư mục gốc repo, sau khi kích hoạt môi trường Python và cấu hình
nguồn dữ liệu theo [runbook](docs/runbook.md#cấu-hình-nguồn-dữ-liệu):

```bash
# Kiểm tra logic trước khi chạy dữ liệu thật
python -m pytest -q

# Chạy pipeline qua DVC và cập nhật lock khi cần
dvc repro

# Hoặc chạy toàn bộ trực tiếp, không qua DVC
python -m src.pipelines.all

# Xem giao diện tracking cho pipeline trong src
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Kiểm tra thay đổi trước khi đưa vào Git
git status
```

DVC bỏ qua stage không thay đổi, nên stage đó không in log kiểm tra mới.
Khi đổi URL local/biến môi trường, chạy `dvc repro --force extract` rồi
`dvc repro`; DVC không tự phát hiện thay đổi các giá trị riêng này.
Xem thêm [README](README.md) và [runbook](docs/runbook.md).
