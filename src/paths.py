"""Cross-platform project path utilities."""

from pathlib import Path


def find_project_root(start_path=None):
    """Find the project root by looking for marker files.

    Searches for .git, pyproject.toml, or CLAUDE.md in the current or parent directories.
    This works cross-platform (Windows/Mac/Linux) and handles Jupyter notebooks where
    __file__ is unavailable and CWD is unpredictable.

    Args:
        start_path: Optional starting directory (defaults to current directory)

    Returns:
        Path object pointing to the project root

    Raises:
        RuntimeError: If no marker file is found
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    markers = ['.git', 'pyproject.toml', 'CLAUDE.md', '.claude']

    for p in [start_path] + list(start_path.parents):
        for marker in markers:
            if (p / marker).exists():
                return p

    raise RuntimeError(
        f"Could not find project root starting from {start_path}. "
        f"Looked for: {', '.join(markers)}"
    )
