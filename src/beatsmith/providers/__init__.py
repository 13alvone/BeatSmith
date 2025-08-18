from __future__ import annotations

import random
from typing import Protocol, List, Dict, Optional

AUDIO_EXTS = {
    ".wav", ".wave", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".oga", ".m4a"
}

class Provider(Protocol):
    def search(
        self,
        rng: random.Random,
        wanted: int,
        query_bias: Optional[str],
        allow_tokens: List[str],
        strict: bool,
    ) -> List[Dict]:
        """Return a list of candidate file infos."""

    def fetch(self, file_info: Dict, cache_dir: str) -> Optional[bytes]:
        """Fetch raw bytes for the given file."""

    def license(self, file_info: Dict) -> Optional[str]:
        """Return license information for the given file."""

def license_ok(licenseurl: Optional[str], allow_tokens: List[str]) -> bool:
    if not licenseurl:
        return False
    low = licenseurl.lower()
    for tok in allow_tokens:
        t = tok.strip().lower()
        if t and t in low:
            return True
    return False
