import hashlib
import os
import random
import time
from typing import List, Dict, Optional

import requests

from .. import lw, ld
from . import AUDIO_EXTS, license_ok

IA_ADV_URL = "https://archive.org/advancedsearch.php"
IA_META_URL = "https://archive.org/metadata/"


def _get_with_retry(
    url: str,
    *,
    params=None,
    headers=None,
    timeout=20,
    stream=False,
    max_retries: int = 3,
    backoff: float = 1.0,
):
    """requests.get with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            r = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
                stream=stream,
            )
            if r.status_code == 429 or r.status_code >= 500:
                wait = backoff
                if r.status_code == 429:
                    ra = r.headers.get("Retry-After")
                    if ra and ra.isdigit():
                        wait = float(ra)
                lw(f"HTTP {r.status_code} for {url}, retrying in {wait}s")
                r.close()
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    backoff *= 2
                    continue
                break
            return r
        except Exception as e:
            lw(f"GET failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
            else:
                break
    lw(f"Exceeded max retries for {url}")
    return None


def ia_search_random(
    rng: random.Random,
    rows: int,
    query_bias: Optional[str],
    allow_tokens: List[str],
    strict: bool,
) -> List[Dict]:
    q_parts = ['mediatype:(audio)']
    if query_bias:
        q_parts.append(f'({query_bias})')
    q = " AND ".join(q_parts)
    params = {
        "q": q,
        "fl[]": ["identifier", "title", "licenseurl", "downloads"],
        "sort[]": "downloads desc",
        "rows": max(10, rows),
        "page": 1,
        "output": "json",
    }
    headers = {"User-Agent": "BeatSmith/3.0 (+open source art engine)"}
    r = _get_with_retry(IA_ADV_URL, params=params, timeout=20, headers=headers)
    if not r:
        return []
    try:
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        lw(f"IA search failed: {e}")
        return []
    finally:
        r.close()
    docs = data.get("response", {}).get("docs", [])
    if strict:
        docs = [d for d in docs if license_ok(d.get("licenseurl"), allow_tokens)]
    rng.shuffle(docs)
    return docs


def ia_pick_files_for_item(identifier: str) -> List[Dict]:
    url = IA_META_URL + identifier
    headers = {"User-Agent": "BeatSmith/3.0 (+open source art engine)"}
    r = _get_with_retry(url, timeout=20, headers=headers)
    if not r:
        return []
    try:
        r.raise_for_status()
        meta = r.json()
    except Exception as e:
        lw(f"IA metadata failed for {identifier}: {e}")
        return []
    finally:
        r.close()
    files = meta.get("files", []) or []
    picked = []
    for f in files:
        name = f.get("name") or ""
        lower = name.lower()
        if any(lower.endswith(ext) for ext in AUDIO_EXTS):
            picked.append(
                {
                    "identifier": identifier,
                    "name": name,
                    "title": meta.get("metadata", {}).get("title"),
                    "licenseurl": meta.get("metadata", {}).get("licenseurl"),
                    "url": f"https://archive.org/download/{identifier}/{name}",
                }
            )
    return picked


def cache_path_for(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, h + ".bin")


def http_get_cached(
    url: str,
    cache_dir: str,
    timeout: int = 30,
    max_bytes: int = 50_000_000,
) -> Optional[bytes]:
    path = cache_path_for(url, cache_dir)
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                b = f.read()
                if b:
                    ld(f"Cache hit: {url}")
                    return b
        except Exception as e:
            lw(f"Cache read failed (will re-download): {e}")
    headers = {"User-Agent": "BeatSmith/3.0 (+open source art engine)"}
    r = _get_with_retry(url, timeout=timeout, headers=headers, stream=True)
    if not r:
        return None
    try:
        with r:
            r.raise_for_status()
            chunks = []
            size = 0
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > max_bytes:
                        lw(f"Truncating download > {max_bytes} bytes: {url}")
                        break
            b = b"".join(chunks)
            if b:
                try:
                    with open(path, "wb") as f:
                        f.write(b)
                except Exception as e:
                    lw(f"Cache write failed: {e}")
            return b
    except Exception as e:
        lw(f"Download failed: {e} :: {url}")
        return None


class InternetArchiveProvider:
    def search(
        self,
        rng: random.Random,
        wanted: int,
        query_bias: Optional[str],
        allow_tokens: List[str],
        strict: bool,
    ) -> List[Dict]:
        docs = ia_search_random(
            rng,
            rows=max(50, wanted * 15),
            query_bias=query_bias,
            allow_tokens=allow_tokens,
            strict=strict,
        )
        files: List[Dict] = []
        for doc in docs:
            ident = doc.get("identifier")
            files.extend(ia_pick_files_for_item(ident) if ident else [])
            if len(files) >= wanted * 3:
                break
        rng.shuffle(files)
        return files

    def fetch(self, file_info: Dict, cache_dir: str) -> Optional[bytes]:
        url = file_info.get("url")
        if not url:
            return None
        return http_get_cached(url, cache_dir)

    def license(self, file_info: Dict) -> Optional[str]:
        return file_info.get("licenseurl")
