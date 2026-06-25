"""Path helpers for manifest portability.

Manifests should be easy to move with the project. New rows store paths
relative to the project root whenever possible; readers still accept the
older absolute paths already present in local CSVs.
"""
from pathlib import Path

from config import PROJECT_ROOT


def manifest_path_for(path: str | Path, root: Path = PROJECT_ROOT) -> str:
    """Return a project-relative manifest path when `path` lives under root."""
    raw_path = Path(path)
    resolved = raw_path.resolve() if raw_path.is_absolute() else (root / raw_path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_manifest_path(
    path: str | Path,
    manifest_file: str | Path | None = None,
    root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a filepath value read from a manifest."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    if manifest_file is not None:
        beside_manifest = Path(manifest_file).resolve().parent / candidate
        if beside_manifest.exists():
            return beside_manifest

    return root / candidate
