import os
import random
from typing import List, Dict, Optional

from . import AUDIO_EXTS, license_ok


class LocalProvider:
    """Search for audio files in a local directory.

    This provider recursively scans ``root`` for files whose extensions match
    :data:`~beatsmith.providers.AUDIO_EXTS`.  When ``query_bias`` is provided it
    is split on whitespace into tokens and only files whose name (case
    insensitive) contains at least one of those tokens will be returned.

    License information can optionally be supplied via ``<filename>.license``
    sidecar files containing a license URL.  When ``strict`` is ``True`` only
    files with a license that matches one of ``allow_tokens`` will be yielded.
    """

    def __init__(self, root: str = "."):
        self.root = root

    def search(
        self,
        rng: random.Random,
        wanted: int,
        query_bias: Optional[str],
        allow_tokens: List[str],
        strict: bool,
    ) -> List[Dict]:
        tokens = []
        if query_bias:
            tokens = [t.strip().lower() for t in query_bias.split() if t.strip()]
        files: List[Dict] = []
        for dirpath, _, filenames in os.walk(self.root):
            for fn in filenames:
                lower = fn.lower()
                if tokens and not any(t in lower for t in tokens):
                    continue
                if any(lower.endswith(ext) for ext in AUDIO_EXTS):
                    path = os.path.join(dirpath, fn)
                    licenseurl: Optional[str] = None
                    lic_path = path + ".license"
                    if os.path.isfile(lic_path):
                        try:
                            with open(lic_path, "r", encoding="utf-8") as lf:
                                licenseurl = lf.read().strip() or None
                        except Exception:
                            licenseurl = None
                    if strict and not license_ok(licenseurl, allow_tokens):
                        continue
                    files.append(
                        {
                            "identifier": path,
                            "name": fn,
                            "title": fn,
                            "licenseurl": licenseurl,
                            "url": path,
                        }
                    )
        rng.shuffle(files)
        return files

    def fetch(self, file_info: Dict, cache_dir: str) -> Optional[bytes]:
        path = file_info.get("url")
        if not path:
            return None
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def license(self, file_info: Dict) -> Optional[str]:
        return file_info.get("licenseurl")
