"""Path utilities for finding project root and common directories."""

from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[Path] = None, marker: str = ".project_root") -> Path:
    """Find the project root directory by looking for a marker file.
    
    Walks up the directory tree from start_path (or current directory) until
    it finds a directory containing the marker file.
    
    Args:
        start_path: Path to start searching from. If None, uses current working directory.
        marker: Name of the marker file to look for (default: ".project_root").
    
    Returns:
        Path to the project root directory.
    
    Raises:
        FileNotFoundError: If project root cannot be found.
    
    Example:
        >>> project_root = find_project_root()
        >>> tools_dir = project_root / "tools"
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()
    
    current = start_path.resolve()
    
    # Walk up the directory tree
    while current != current.parent:
        marker_path = current / marker
        if marker_path.exists():
            return current
        current = current.parent
    
    # If not found, try common fallback locations
    # This handles cases where notebooks are in deeply nested subdirectories
    fallback_paths = [
        start_path.parent.parent.parent,  # tests/xxx/xxx_files/
        start_path.parent.parent,          # tests/xxx/
        start_path.parent,                 # tests/
    ]
    
    for fallback in fallback_paths:
        if fallback.exists() and (fallback / marker).exists():
            return fallback
    
    raise FileNotFoundError(
        f"Could not find project root (looking for '{marker}'). "
        f"Started search from: {start_path}"
    )


def get_project_paths(project_root: Optional[Path] = None) -> dict:
    """Get common project paths as a dictionary.
    
    Args:
        project_root: Project root path. If None, will be found automatically.
    
    Returns:
        Dictionary with keys: 'root', 'tools', 'utils', 'tests', 'docs'
    
    Example:
        >>> paths = get_project_paths()
        >>> tools_dir = paths['tools']
        >>> utils_dir = paths['utils']
    """
    if project_root is None:
        project_root = find_project_root()
    
    return {
        'root': project_root,
        'tools': project_root / 'tools',
        'utils': project_root / 'utils',
        'tests': project_root / 'tests',
        'docs': project_root / 'docs',
    }


def list_files(dir_path: Path) -> list[Path]:
    """Recursively list all files in a directory.

    Returns sorted list of file paths (empty list if directory doesn't exist).
    """
    if not dir_path.exists():
        return []
    return sorted(p for p in dir_path.rglob("*") if p.is_file())


def format_paths(paths: list[Path] | Path | None, root: Optional[Path] = None) -> str:
    """Format path(s) as strings, optionally relative to a root directory.

    For lists, paths are joined with semicolons.
    Returns empty string for None or empty lists.
    If root is None, returns absolute paths.

    Args:
        paths: Single path, list of paths, or None.
        root: Directory to strip as a prefix, producing relative paths. If None, absolute paths are returned.
    """
    if paths is None:
        return ""
    if isinstance(paths, Path):
        return str(paths.relative_to(root)) if root else str(paths)
    if not paths:
        return ""
    return ";".join(str(p.relative_to(root)) if root else str(p) for p in paths)


def find_subdir(parent: Path, prefix: str) -> Optional[Path]:
    """Find the first subdirectory in parent that starts with a given prefix.

    Returns path to the first matching subdirectory, or None if not found.
    """
    if not parent.exists():
        return None
    matches = sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix))
    return matches[0] if matches else None


def find_log_file(
    log_dir: Path,
    prefix: str,
    extensions: set[str] = {".log", ".txt"},
    exclude_pattern: Optional[str] = None,
) -> Optional[Path]:
    """Find a log file in a directory matching a given prefix.

    Searches for files whose name contains prefix and whose suffix is in extensions.
    If exclude_pattern is given, prefers files that do not contain that pattern
    in their name (falls back to the full candidate list if none remain).

    Args:
        log_dir: Directory to search in.
        prefix: Substring that must appear in the filename.
        extensions: Allowed file extensions (default: .log and .txt).
        exclude_pattern: If provided, deprioritize files containing this substring.

    Returns:
        Path to the first matching file, or None if not found or directory doesn't exist.
    """
    if not log_dir.exists():
        return None
    candidates = sorted(
        p for p in log_dir.iterdir()
        if p.is_file() and prefix in p.name and p.suffix in extensions
    )
    if exclude_pattern:
        filtered = [p for p in candidates if exclude_pattern not in p.name]
        if filtered:
            return filtered[0]
    return candidates[0] if candidates else None

