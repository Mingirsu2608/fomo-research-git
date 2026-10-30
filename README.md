# FOMO Research Git

## Cấu trúc

```text
fomo-research-git/
├── dvc.yaml                 pipeline và dependency
├── dvc.lock                 DVC sinh sau lần `dvc repro` đầu tiên
├── params.yaml              as_of_date, feature set, model, K, seed
├── configs/                 mẫu cấu hình và file local riêng được ignore
├── src/
│   ├── data/                extract Google Drive, snapshot, manifest
│   ├── features/            định nghĩa và tạo feature
│   ├── models/              preprocessing và clustering
│   ├── evaluation/          metric và cluster profile
│   └── pipelines/           entrypoint cho từng stage
├── tests/                   schema, feature và config tests
├── env/                     dependency lock và Dockerfile
├── notebooks/
│   └── baseline_notenook.ipynb  notebook nghiên cứu tham chiếu
└── docs/                    model card, data dictionary, runbook
```

## Thiết lập và vận hành

Yêu cầu Python >= 3.11. Chạy các lệnh từ thư mục gốc repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r env/requirements.lock
[ -f configs/data.local.json ] || cp configs/data.local.example.json configs/data.local.json
```

Điền URL dữ liệu vào `FOMO_MARKET_URL` và `FOMO_TRADES_URL` trong
`configs/data.local.json` (file được Git ignore), sau đó chạy pipeline:

```bash
dvc repro
```

Hoặc chạy trực tiếp bằng Python:

```bash
python -m src.pipelines.all
```
