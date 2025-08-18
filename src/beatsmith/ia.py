import hashlib
import os
import random
from typing import List, Dict, Optional

import requests

from . import lw, ld

IA_ADV_URL = "https://archive.org/advancedsearch.php"
IA_META_URL = "https://archive.org/metadata/"
AUDIO_EXTS = {".wav", ".wave", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".oga", ".m4a"}

def ia_license_ok(licenseurl: Optional[str], allow_tokens: List[str]) -> bool:
    if not licenseurl:
        return False
    low = licenseurl.lower()
    for tok in allow_tokens:
        if tok.strip() and tok.strip().lower() in low:
            return True
    return False

def ia_search_random(rng: random.Random, rows: int, query_bias: Optional[str],
                     allow_tokens: List[str], strict: bool) -> List[Dict]:
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
    try:
        r = requests.get(IA_ADV_URL, params=params, timeout=20, headers=headers)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        lw(f"IA search failed: {e}")
        return []
    docs = data.get("response", {}).get("docs", [])
    if strict:
        docs = [d for d in docs if ia_license_ok(d.get("licenseurl"), allow_tokens)]
    rng.shuffle(docs)
    return docs

def ia_pick_files_for_item(identifier: str) -> List[Dict]:
    url = IA_META_URL + identifier
    headers = {"User-Agent": "BeatSmith/3.0 (+open source art engine)"}
    try:
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()
        meta = r.json()
    except Exception as e:
        lw(f"IA metadata failed for {identifier}: {e}")
        return []
    files = meta.get("files", []) or []
    picked = []
    for f in files:
        name = f.get("name") or ""
        lower = name.lower()
        if any(lower.endswith(ext) for ext in AUDIO_EXTS):
            picked.append({
                "identifier": identifier,
                "name": name,
                "title": meta.get("metadata", {}).get("title"),
                "licenseurl": meta.get("metadata", {}).get("licenseurl"),
                "url": f"https://archive.org/download/{identifier}/{name}"
            })
    return picked

def cache_path_for(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, h + ".bin")

def http_get_cached(url: str, cache_dir: str, timeout: int = 30, max_bytes: int = 50_000_000) -> Optional[bytes]:
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
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
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
