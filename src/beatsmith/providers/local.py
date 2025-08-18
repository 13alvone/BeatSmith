import os
import random
from typing import List, Dict, Optional

from . import AUDIO_EXTS


class LocalProvider:
    """Simple local-folder provider scaffold."""

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
        files: List[Dict] = []
        for dirpath, _, filenames in os.walk(self.root):
            for fn in filenames:
                lower = fn.lower()
                if any(lower.endswith(ext) for ext in AUDIO_EXTS):
                    path = os.path.join(dirpath, fn)
                    files.append(
                        {
                            "identifier": path,
                            "name": fn,
                            "title": fn,
                            "licenseurl": None,
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
