import sys
from pathlib import Path

# Ensure src/ is on the path for tests without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
