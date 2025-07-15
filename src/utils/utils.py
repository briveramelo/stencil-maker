from pathlib import Path


def ensure_out_dir(path: Path) -> None:
    """Create *path* (and parents) if missing, raise if not writable."""
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(path)
