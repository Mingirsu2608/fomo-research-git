# Đối chiếu yêu cầu nghiệm thu và bằng chứng trong repo

Tài liệu này trả lời lần lượt sáu yêu cầu nghiệm thu, chỉ ra implementation,
kết quả hiện có và ví dụ để người xem tự kiểm tra.

**Thời điểm đối chiếu:** 29/10/2026. **Commit nền của snapshot kết quả:** `b197c9f`.
Hướng dẫn cấu hình đã cập nhật theo working tree sau khi tách URL riêng;
chưa chạy lại end-to-end trên Drive cho thay đổi cấu hình này.
Các số liệu bên dưới là snapshot từ artifact/MLflow cục bộ tại thời điểm kiểm tra;
chúng có thể đổi sau lần chạy tiếp theo. File này không chứa dữ liệu giao dịch thô.

## Tóm tắt

| Yêu cầu | Repo đã có | Giới hạn cần nêu khi trình bày |
| --- | --- | --- |
| Tách module và lịch sử Git | Module data/features/models/evaluation/pipelines; các commit theo thay đổi | Tên folder thực tế dùng số nhiều ở `models`, `pipelines`. |
| Chạy end-to-end, output và run log | Đủ output bốn stage, MLflow run FINISHED, log PASS/FAIL, 25 test pass | Console log chưa được lưu thành file `.log` trong repo; quyền Drive cần được cấp khi chạy. |
| Truy vết data/artifact bằng DVC | `dvc.lock` đã commit, manifest SHA256, cache DVC khớp artifact hiện tại | Chưa cấu hình DVC remote; clone Git đơn thuần chưa tải được data/model. |
| Ghi run/parameters/metrics/artifacts | MLflow experiment, params, silhouette và ba báo cáo CSV | Model `.joblib` do DVC quản lý; các metric evaluation bổ sung nằm ở JSON, chưa log vào MLflow. |
| Setup/run, dependency, config, logs/exception | README/runbook, requirements, YAML config, wrapper kiểm tra và traceback | Dependency có khoảng phiên bản, chưa khóa tuyệt đối; chưa triển khai kết nối PostgreSQL. |
| Chênh lệch/refactor/giới hạn/tái lập | Notebook nghiên cứu, source tách module, model card và phần 6 dưới đây | Không khẳng định kết quả giống tuyệt đối notebook gốc hoặc đã kiểm tra ngoài mẫu. |

## 1. Tách các module theo source thực tế; Git có lịch sử hợp lý

**Trả lời:** repo đã chuyển logic nghiên cứu thành các module có trách nhiệm riêng.

| Phần | Bằng chứng | Công dụng |
| --- | --- | --- |
| Data | [`src/data/extract.py`](../src/data/extract.py) | Tải hai CSV từ Drive, tính SHA256 và ghi manifest. |
| Features | [`src/features/build.py`](../src/features/build.py) | Tính 32 feature theo investor-window, lọc cửa sổ đủ hoạt động. |
| Model | [`src/models/preprocess.py`](../src/models/preprocess.py), [`train.py`](../src/models/train.py) | Phễu lọc, complete-case, StandardScaler, PCA và tuning ba thuật toán. |
| Evaluation | [`src/evaluation/evaluate.py`](../src/evaluation/evaluate.py) | Metric, kích thước và profile cụm. |
| Pipeline | [`src/pipelines/all.py`](../src/pipelines/all.py), [`validation.py`](../src/pipelines/validation.py) | Điều phối bốn stage, kiểm tra output và dừng khi lỗi. |

Ví dụ lịch sử đang có:

```text
107dedd  Initialize reproducible FOMO research pipeline
c84a9eb  Add baseline notebook to notebooks directory
1c6781e  First git comit
a127124  Test record clustering
c1e71bf  add test case
5968354  update notebook md
b197c9f  update README and FILE_GUIDE
```

**Cách xem:**

```bash
git log --oneline --graph --all
git show --stat c1e71bf
git ls-files src tests notebooks
```

`git show --stat` cho thấy file nào thay đổi trong commit; có thể mở từng file
để đối chiếu trách nhiệm module với bảng trên. Xem cây thư mục chi tiết ở
[`FILE_GUIDE.md`](../FILE_GUIDE.md).

## 2. Chạy end-to-end với dữ liệu được cấp quyền; có output và run log; lỗi rõ ràng

**Trả lời:** repo có flow chạy đầy đủ:

```text
extract → features → feature funnel → complete-case → scaling → PCA → cluster → evaluate
```

Bốn stage chính được khai báo trong [`dvc.yaml`](../dvc.yaml); các bước
preprocessing chạy bên trong stage `cluster`.

**Bằng chứng output hiện có:**

| Stage | Ví dụ output để xem |
| --- | --- |
| Extract | [`artifacts/raw/manifest.json`](../artifacts/raw/manifest.json) và hai CSV raw |
| Features | `artifacts/features/investor_window_32features.csv` |
| Cluster | `preprocessor.joblib`, `kmeans.joblib`, `minibatch_kmeans.joblib`, `gmm.joblib`, `features.txt`, `feature_selection.csv`, `pca_variance.csv`, `tuning_results.csv` trong `artifacts/models/` |
| Evaluate | [`model_metrics.json`](../artifacts/evaluation/model_metrics.json), `cluster_sizes.csv`, `cluster_profiles.csv` trong `artifacts/evaluation/` |

Các link artifact chỉ mở được trên máy đã có data/model; GitHub không chứa
những file bị `.gitignore` loại trừ.

MLflow cục bộ có run `2e6e847fa7dc41b2aa66b9627d210cf4`, tên `cluster`,
trạng thái **FINISHED**. Kết quả ghi nhận **118.246 investor-windows**, còn
**70.653 complete cases**, giữ **20 feature**, PCA còn **14 chiều**.

**Lệnh chạy và lưu console log khi trình diễn:**

```bash
source .venv/bin/activate
python -m pytest -q
mkdir -p outputs/logs
set -o pipefail
dvc repro 2>&1 | tee outputs/logs/dvc-run.log
```

`outputs/` đang được Git bỏ qua. `pipefail` giữ exit code lỗi nếu DVC thất bại.
Nếu stage không đổi, DVC bỏ qua và không in PASS mới. Để trình diễn toàn bộ:

```bash
dvc repro --force 2>&1 | tee outputs/logs/dvc-full-run.log
```

Lệnh `--force` chạy lại cả extract và tải lại dữ liệu Drive. Chỉ chạy khi đã
có quyền truy cập hai URL trong cấu hình local được ignore hoặc biến môi trường.

**Ví dụ định dạng log mong đợi** — minh họa, không phải file log đã lưu:

```text
stage=extract status=PASS
stage=features status=PASS
step=feature_funnel status=PASS
step=complete_case status=PASS
step=scaling status=PASS
step=pca status=PASS
stage=cluster status=PASS
stage=evaluate status=PASS
```

`python -m src.pipelines.all` còn có log `pipeline status=PASS` cho toàn flow.
Khi lỗi, wrapper ở [`common.py`](../src/pipelines/common.py) log `FAIL` và
traceback, rồi ném lại exception để trả exit code khác 0. Ví dụ: schema thiếu,
output rỗng, PCA không hữu hạn hoặc số hàng cluster không khớp cohort.

**Kiểm tra đã thực hiện:** `python -m pytest -q --disable-warnings` trả
**25 passed** sau thay đổi cấu hình nguồn dữ liệu. Bao gồm kiểm tra ưu tiên
biến môi trường/file local, thiếu cấu hình, URL không còn trong file Git ở
working tree khi có file local để đối chiếu, output lỗi và dừng trước stage tiếp theo.
Các test dùng fixture nhỏ, không phải một lần tải lại toàn bộ Drive.
Artifact hiện tại và run FINISHED là bằng chứng kết quả chạy đầy đủ đã có.

Nếu thiếu thư viện, Python báo `ModuleNotFoundError`; cài lại bằng
`python -m pip install -r env/requirements.lock`. Nếu Drive bị từ chối quyền,
kiểm tra quyền chia sẻ; pipeline không tự cấp hoặc bỏ qua quyền dữ liệu.

## 3. Truy vết tối thiểu một phiên bản data/artifact bằng DVC; không đưa raw data vào Git

**Trả lời:** repo đã có phiên bản DVC được lưu trong lịch sử Git. Commit
`c1e71bf` chứa `dvc.lock` với các hash tương ứng bộ artifact đang đối chiếu.

| Output | Hash DVC trong snapshot |
| --- | --- |
| `artifacts/raw` | `b520c677ccb3ef2343dce26561cf0adc.dir` |
| `artifacts/features` | `fc6a80cb8c808c115537831cee5aeb8f.dir` |
| `artifacts/models` | `d7f5754112cfcde6cc100aaa4170b13a.dir` |

Manifest raw ghi `as_of_date: 2026-10-29` và SHA256 từng CSV, ví dụ market:

```text
af720799ff0b86bb8c08500e912c30349065b1e6eb55970831be4771b04845d4
```

DVC dùng hash thư mục `.dir` để tham chiếu danh sách file bên trong; SHA256
trong manifest là hash riêng từng raw file, không phải cùng loại hash trên.
Đã kiểm tra trực tiếp các file raw/features/models khớp manifest cache DVC.
Ở snapshot trước thay đổi cấu hình, `dvc status` trả
**Data and pipelines are up to date.** khi dùng site cache tạm có quyền ghi.
Sau khi đổi code/params và loại URL khỏi lock, DVC có thể báo stage đã thay đổi;
chạy lại pipeline để cập nhật hash, không coi lock hiện tại là bằng chứng đã
chạy lại code mới.

**Ví dụ truy vết chỉ đọc, không thay đổi working tree:**

```bash
git log --oneline -- dvc.lock
git show c1e71bf:dvc.lock
git show c1e71bf:params.yaml
dvc status
```

Xem thay đổi giữa hai phiên bản:

```bash
dvc diff a127124 c1e71bf
git diff a127124 c1e71bf -- params.yaml dvc.lock
```

Khôi phục một phiên bản để xem lại, sau khi đã commit/stash thay đổi đang làm:

```bash
git switch --detach c1e71bf
dvc checkout
```

`git switch` khôi phục code/config/lock; `dvc checkout` lấy data/model từ cache.
Không chạy lại extract chỉ để xem phiên bản cũ vì link Drive có thể đổi nội dung.

**Kiểm tra raw data không được Git track:**

```bash
git ls-files artifacts mlflow.db .env mlartifacts mlruns
git check-ignore artifacts/raw/btc_trade_log.csv artifacts/models/kmeans.joblib
```

Tại thời điểm đối chiếu, lệnh thứ nhất không trả file nào; lệnh thứ hai xác nhận
raw/model bị ignore. Đây là kiểm tra trạng thái tracked hiện tại, không phải
chứng nhận quét bí mật toàn bộ lịch sử Git. Dữ liệu nhạy cảm vẫn cần được
kiểm tra nội dung trước khi chia sẻ bất kỳ file mới nào.

**Giới hạn:** `.dvc/config` hiện chưa có remote. Phiên bản đang khôi phục được
nhờ cache cục bộ; để người khác khôi phục sau khi clone cần DVC remote được cấp
quyền và `dvc push`/`dvc pull`. GitHub chỉ có lock, không có raw/model.

## 4. Ghi run, parameters, metrics và artifacts phù hợp pipeline FOMO

**Trả lời:** [`train.py`](../src/models/train.py) ghi MLflow experiment
`fomo-clustering`, tracking URI `sqlite:///mlflow.db`.

**Run ví dụ:** `2e6e847fa7dc41b2aa66b9627d210cf4` — `FINISHED`.

| Nhóm | Giá trị ví dụ đã ghi |
| --- | --- |
| Parameters | `as_of_date=2026-10-29`, `seed=42`, `feature_set=fomo_32_v1` |
| Preprocessing | coverage `0.8`, std/IQR `0.02`, Spearman `0.7`, PCA target `0.9` |
| Số chiều/cohort | `selected_features=20`, `pca_components=14`, `cohort_rows=70653`, `excluded_rows=47593` |
| K đã chọn | KMeans `2`, MiniBatchKMeans `2`, GMM `2` |
| MLflow artifacts | `feature_selection.csv`, `pca_variance.csv`, `tuning_results.csv` |

Metric silhouette tuning trong run:

| Model | Silhouette tuning |
| --- | ---: |
| KMeans | 0.231723 |
| MiniBatchKMeans | 0.228806 |
| GMM | 0.234195 |

**Cách xem trên giao diện:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Mở địa chỉ terminal thông báo, thường là `http://127.0.0.1:5000`, chọn experiment
`fomo-clustering`, mở run nêu trên và xem **Parameters / Metrics / Artifacts**.
Database và artifact MLflow phải còn trên máy; chúng không được Git push tự động.

**Cách xem metric evaluation bổ sung:**

```bash
python -m json.tool artifacts/evaluation/model_metrics.json
```

| Model | Silhouette evaluation | Calinski–Harabasz | Davies–Bouldin |
| --- | ---: | ---: | ---: |
| KMeans | 0.230257 | 20461.172850 | 1.722741 |
| MiniBatchKMeans | 0.226729 | 20417.044765 | 1.738378 |
| GMM | 0.233232 | 17961.065401 | 1.786111 |

Silhouette tuning và evaluation có thể khác vì cách lấy mẫu. Các metric
Calinski–Harabasz/Davies–Bouldin và profile được ghi vào output evaluation,
chưa được log vào MLflow. Ba file model và preprocessing được lưu trong
`artifacts/models/`, quản lý bằng DVC; hiện không được upload làm MLflow artifacts.
Notebook nghiên cứu giữ model trong bộ nhớ và không dùng MLflow.

## 5. Setup/run; dependencies có version; config tách source; logs/exception

**Trả lời:** repo đã có các thành phần sau:

| Thành phần | Bằng chứng |
| --- | --- |
| Setup/run | [`README.md`](../README.md), [`docs/runbook.md`](runbook.md) |
| Dependency có ràng buộc phiên bản | [`env/requirements.lock`](../env/requirements.lock) |
| Python tối thiểu và pytest | [`pyproject.toml`](../pyproject.toml): Python ≥ 3.11 |
| Tham số tách source | [`params.yaml`](../params.yaml) |
| Mẫu cấu hình nguồn dữ liệu riêng | [`configs/data.local.example.json`](../configs/data.local.example.json); file local chứa URL được Git ignore |
| Log/exception | [`src/pipelines/common.py`](../src/pipelines/common.py) và [`validation.py`](../src/pipelines/validation.py) |
| Đóng gói môi trường tùy chọn | [`env/Dockerfile`](../env/Dockerfile) |

**Ví dụ setup từ đầu:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r env/requirements.lock
[ -f configs/data.local.json ] || cp configs/data.local.example.json configs/data.local.json
# Điền URL riêng vào file local theo docs/runbook.md trước khi chạy pipeline.
python -m pytest -q
dvc repro
```

Các dependency có version/range, ví dụ `numpy>=2.2,<2.3`,
`scikit-learn>=1.7,<1.8`, `pytest>=8.0`. File mang tên `.lock` nhưng chưa khóa
chính xác mọi version/transitive dependency, nên không khẳng định hai máy cài
vào thời điểm khác nhau sẽ có môi trường hoàn toàn giống nhau.

Muốn thay K hoặc ngưỡng lọc, sửa YAML thay vì sửa thuật toán:

```yaml
preprocessing:
  min_coverage: 0.80
  min_std: 0.02
  min_iqr: 0.02
  max_spearman: 0.70
  pca_variance_target: 0.90

model:
  k_candidates: [2, 3, 4, 5, 6]
```

Đây là trích đoạn; giữ nguyên các khóa còn lại trong file thực tế. Sau khi sửa,
chạy tests rồi `dvc repro`; các stage phụ thuộc tham số sẽ được cập nhật.
Source hiện đọc URL Google Drive từ cấu hình local hoặc biến môi trường;
chưa triển khai kết nối PostgreSQL. Biến môi trường có ưu tiên hơn file local;
code không tự đọc `.env`. Khi đổi nguồn riêng, chạy `dvc repro --force extract`
rồi `dvc repro` vì DVC không theo dõi giá trị này. Xem [runbook](runbook.md).

## 6. Chênh lệch source gốc, refactor, giới hạn và cách tái lập

### 6.1. Chênh lệch đã xác định

Mốc so sánh notebook nghiên cứu ban đầu trong lịch sử Git: **`c84a9eb`**.
Notebook hiện tại đã được sửa thành bản nghiên cứu nhất quán với logic source,
vì vậy dùng commit cũ để xem thiết kế trước refactor.

| Nội dung | Notebook ở mốc cũ | Implementation hiện tại |
| --- | --- | --- |
| Số feature sau phễu | Assert cố định 25 sau coverage/dispersion, 22 sau loại ba feature | Tính từ dữ liệu; Spearman loại tự động theo ưu tiên; snapshot hiện giữ 20. |
| Preprocessing | Complete-case, StandardScaler, PCA mục tiêu 90% | Giữ phương pháp; đưa thành hàm dùng lại và lưu scaler/PCA cho pipeline. |
| K | KMeans/GMM thử từ 2 đến 8 | `params.yaml` đặt từ 2 đến 6. |
| GMM | Thử spherical/tied/diag/full và nhiều lần khởi tạo | Covariance `diag`, `n_init=2`, `max_iter` theo cấu hình. |
| MiniBatchKMeans | Bayesian tuning bằng Optuna | Thử K cấu hình, batch size 1024; chưa chuyển Optuna vào pipeline. |
| Stability | Có phần bootstrap/ARI stability trong nghiên cứu cũ | Chưa đưa vào flow hiện tại; chọn K bằng silhouette. |
| Tổ chức chạy | State nối giữa các cell | Module riêng, config YAML, DVC, MLflow, tests và runtime validation. |
| Notebook hiện tại | — | Code/thuật toán nhúng, URL đọc từ biến môi trường hoặc JSON local; xử lý trong bộ nhớ, không xuất model/không tracking; output đã được xóa khỏi bản chia sẻ. |

Ngoài ra, docstring của `src/features/build.py` ghi rõ: **6 feature giữ công thức
Week 1 có source thực thi; 26 feature còn lại được tái dựng từ định nghĩa trong
`Clustering_Presentation.md`** vì thiếu Python gốc. Tài liệu tham chiếu này không
nằm trong repo hiện tại; không tuyên bố tương đương tuyệt đối với source ban đầu.

**Ví dụ xem bản gốc và so sánh:**

```bash
git show c84a9eb:notebooks/baseline_notenook.ipynb
git diff c84a9eb HEAD -- notebooks/baseline_notenook.ipynb src params.yaml
python -m pytest -q tests/test_notebook_standalone.py
```

Tests notebook kiểm tra công thức/hàm dùng chung, vòng lặp fit/tuning và metric
vẫn khớp source; phần I/O được phép khác do notebook làm nghiên cứu trong bộ nhớ.

### 6.2. Giới hạn hiện tại

- Metric dùng trên cohort training; chưa có đánh giá ngoài mẫu hoặc stability theo thời gian trong flow hiện tại.
- Complete-case loại 47.593/118.246 cửa sổ trong run ví dụ, có thể thiên về nhà đầu tư hoạt động thường xuyên.
- `as_of_date` hiện được ghi metadata, chưa tự cắt dữ liệu đến ngày đó; cần snapshot/hash để xác định đúng input.
- Silhouette tuning có seed lấy mẫu; evaluation hiện chưa truyền `random_state` cho lấy mẫu silhouette nên có thể dao động giữa các lần chạy.
- Các commit cũ còn URL và output notebook; cấu hình local hiện tại không xóa chúng khỏi lịch sử Git.
- DVC không theo dõi giá trị URL riêng; phải chủ động chạy lại extract khi đổi nguồn.
- URL Drive có thể thay nội dung. Chạy lại từ URL không thay thế khôi phục đúng snapshot DVC.
- Chưa có DVC remote; chưa khóa tuyệt đối môi trường thư viện.
- MLflow run chưa tự gắn Git commit/data hash thành parameter. Hiện phải đối chiếu run, config và DVC/Git cùng nhau.
- Log runtime đang ẩn warnings; test vẫn có warnings thư viện. `PASS` xác nhận các điều kiện kỹ thuật được kiểm tra, không chứng minh model đã đạt chất lượng nghiệp vụ.

### 6.3. Hai cách tái lập

**A. Khôi phục đúng artifact đã có:** dùng code/config/lock của commit, rồi
`dvc checkout` từ cache; hoặc `dvc pull` nếu đã có remote. Đây là cách xem đúng
phiên bản model/data mà không train lại.

**B. Tính lại từ snapshot raw đã khôi phục:** sau khi checkout commit và raw,
cài dependency, chạy tests và chạy ba stage downstream, không tải lại Drive:

```bash
python -m pytest -q
python -m src.pipelines.features
python -m src.pipelines.cluster
python -m src.pipelines.evaluate
```

Cách B chạy lại phép tính; có thể chênh nhẹ do môi trường, số học hoặc sampling
silhouette evaluation. Khi thay đổi code/tham số để tạo thí nghiệm mới, chạy
`dvc repro` và commit code/config/lock của phiên bản mới.

## Checklist trình diễn ngắn

1. Mở `FILE_GUIDE.md`, chỉ ra các module và chạy `git log --oneline`.
2. Chạy `python -m pytest -q`, đối chiếu số test thực tế.
3. Chạy `dvc status`, mở `dvc.lock` và `artifacts/raw/manifest.json`.
4. Mở `feature_selection.csv`, `pca_variance.csv`, `tuning_results.csv` và `model_metrics.json` để trình bày output.
5. Mở MLflow UI, chọn run FINISHED để xem params, metric và ba artifact CSV.
6. Chỉ ra mục chênh lệch/giới hạn ở trên; nếu cần chạy lại toàn bộ thì dùng `dvc repro --force` và lưu log.
