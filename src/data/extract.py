import hashlib
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import urlretrieve


def data_source_url(data, name):
    """Resolve a private URL without storing it in versioned parameters."""
    env_name = data[f"{name}_url_env"]
    value = os.environ.get(env_name)
    if not value:
        for directory in (Path.cwd(), Path.cwd().parent):
            local_path = directory / "configs" / "data.local.json"
            if local_path.is_file():
                with local_path.open(encoding="utf-8") as file:
                    value = json.load(file).get(env_name)
                break
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set {env_name} or configure configs/data.local.json")
    return value.strip()


def google_drive_download_url(shared_url):
    parts = urlparse(shared_url)
    file_id = parts.path.split("/d/")[1].split("/")[0] if "/d/" in parts.path else ""
    if not file_id:
        file_id = parse_qs(parts.query).get("id", [""])[0]
    if not file_id:
        raise ValueError("Invalid Google Drive URL")
    return (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(shared_url, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(google_drive_download_url(shared_url), output)
    if output.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {output}")
    return output


def extract(config):
    data = config["data"]
    market = download(data_source_url(data, "market"), data["market_path"])
    trades = download(data_source_url(data, "trades"), data["trades_path"])
    manifest = {
        "as_of_date": config["as_of_date"],
        "files": {
            "market": {"path": str(market), "sha256": sha256(market)},
            "trades": {"path": str(trades), "sha256": sha256(trades)},
        },
    }
    manifest_path = Path(data["manifest_path"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

