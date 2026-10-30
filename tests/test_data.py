from src.data.extract import google_drive_download_url


def test_google_drive_url_is_converted():
    url = "https://drive.google.com/file/d/example-id/view"
    assert "id=example-id" in google_drive_download_url(url)


def test_private_source_environment_overrides_local_file(tmp_path, monkeypatch):
    from src.data.extract import data_source_url
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/data.local.json').write_text('{"FOMO_MARKET_URL": "local-value"}')
    monkeypatch.delenv('FOMO_MARKET_URL', raising=False)
    config = {'market_url_env': 'FOMO_MARKET_URL'}
    assert data_source_url(config, 'market') == 'local-value'
    monkeypatch.setenv('FOMO_MARKET_URL', 'environment-value')
    assert data_source_url(config, 'market') == 'environment-value'


def test_missing_source_reports_variable_without_url(tmp_path, monkeypatch):
    import pytest
    from src.data.extract import data_source_url
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('FOMO_MARKET_URL', raising=False)
    with pytest.raises(ValueError, match='Set FOMO_MARKET_URL'):
        data_source_url({'market_url_env': 'FOMO_MARKET_URL'}, 'market')


def test_private_urls_are_absent_from_tracked_working_files():
    import json
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    local = root / 'configs/data.local.json'
    if not local.exists():
        return
    urls = [value for value in json.loads(local.read_text()).values() if value]
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    for name in filter(None, files):
        path = root / name
        if path.is_file():
            content = path.read_text(errors='replace')
            assert not any(url in content for url in urls), name
