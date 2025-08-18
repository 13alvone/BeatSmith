import random
import sys
from pathlib import Path
from unittest.mock import Mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from beatsmith import ia


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", headers=None):
        self.status_code = status_code
        self._json = json_data or {}
        self._content = content
        self.headers = headers or {}

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def iter_content(self, chunk_size=1):
        yield self._content


def test_ia_search_random_retries(monkeypatch):
    responses = [
        DummyResponse(status_code=500),
        DummyResponse(status_code=500),
        DummyResponse(status_code=200, json_data={"response": {"docs": [{"identifier": "ok"}]}}),
    ]
    mget = Mock(side_effect=responses)
    monkeypatch.setattr(ia.requests, "get", mget)
    monkeypatch.setattr(ia.time, "sleep", lambda s: None)
    rng = random.Random(0)
    docs = ia.ia_search_random(rng, rows=1, query_bias=None, allow_tokens=[], strict=False)
    assert docs and docs[0]["identifier"] == "ok"
    assert mget.call_count == 3


def test_ia_pick_files_for_item_retries(monkeypatch):
    meta_json = {"files": [{"name": "a.wav"}], "metadata": {"title": "t", "licenseurl": "l"}}
    responses = [DummyResponse(500), DummyResponse(200, json_data=meta_json)]
    mget = Mock(side_effect=responses)
    monkeypatch.setattr(ia.requests, "get", mget)
    monkeypatch.setattr(ia.time, "sleep", lambda s: None)
    files = ia.ia_pick_files_for_item("ident")
    assert files and files[0]["name"] == "a.wav"
    assert mget.call_count == 2


def test_http_get_cached_retries(monkeypatch, tmp_path):
    responses = [
        DummyResponse(500),
        DummyResponse(200, content=b"data"),
    ]
    mget = Mock(side_effect=responses)
    monkeypatch.setattr(ia.requests, "get", mget)
    monkeypatch.setattr(ia.time, "sleep", lambda s: None)
    b = ia.http_get_cached("http://example.com", tmp_path)
    assert b == b"data"
    assert mget.call_count == 2


def test_http_get_cached_gives_up(monkeypatch, tmp_path):
    responses = [DummyResponse(500), DummyResponse(500), DummyResponse(500)]
    mget = Mock(side_effect=responses)
    monkeypatch.setattr(ia.requests, "get", mget)
    monkeypatch.setattr(ia.time, "sleep", lambda s: None)
    b = ia.http_get_cached("http://example.com", tmp_path)
    assert b is None
    assert mget.call_count == 3

