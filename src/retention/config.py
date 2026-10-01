"""Paths and dataset discovery. No absolute paths anywhere else in the project."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW, DATA_PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
REPORTS, FIGURES, TABLES = ROOT / "reports", ROOT / "reports" / "figures", ROOT / "reports" / "tables"
APP_DATA = ROOT / "app_data"
RESULTS_JSON = REPORTS / "results.json"
LINES_PARQUET = DATA_PROCESSED / "lines.parquet"
SKIP_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".ipynb_checkpoints"}

#: Optional manual override, relative to the project root (e.g. "data/raw/online_retail_II.csv").
#: Leave as None to auto-discover. Resolved on this machine to: online_retail_II.csv
DATASET_PATH: str | None = None


def find_dataset(root: Path = ROOT) -> Path:
    """Locate the Online Retail II file anywhere under the project root. CSV preferred."""
    if DATASET_PATH is not None:
        override = (root / DATASET_PATH).resolve()
        if not override.is_file():
            raise FileNotFoundError(f"DATASET_PATH is set but {override} does not exist.")
        return override
    hits = [p for p in root.rglob("*")
            if p.is_file() and p.name.lower().startswith("online_retail")
            and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
            and not SKIP_DIRS & set(p.relative_to(root).parts)
            and DATA_PROCESSED not in p.parents]
    if not hits:
        raise FileNotFoundError(
            f"No file matching 'online_retail*.(csv|xlsx|xls)' under {root}. "
            "Place the dataset in the project folder or set DATASET_PATH in config.py.")
    hits.sort(key=lambda p: (p.suffix.lower() != ".csv", len(str(p)), str(p)))
    return hits[0]


def relative_to_root(path: Path, root: Path = ROOT) -> str:
    """Render a path relative to the project root when possible (for logs and results.json)."""
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (DATA_RAW, DATA_PROCESSED, REPORTS, FIGURES, TABLES, APP_DATA):
        d.mkdir(parents=True, exist_ok=True)
