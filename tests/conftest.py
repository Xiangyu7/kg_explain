"""pytest configuration: ensure kg_explain is importable from src/ layout."""
import sys
from pathlib import Path

# Add src/ to sys.path so kg_explain is importable
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
