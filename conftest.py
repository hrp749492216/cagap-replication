"""Repo-root pytest conftest. Adds the project root to sys.path so test modules
can `from analysis.x import y`, `from extractor.y import z`, etc., without
requiring `PYTHONPATH=.` at the command line.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
