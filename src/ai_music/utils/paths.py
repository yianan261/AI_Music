"""Path utilities."""

from pathlib import Path


def get_project_root() -> Path:
    """Project root (directory containing src/)."""
    return Path(__file__).resolve().parent.parent.parent.parent
