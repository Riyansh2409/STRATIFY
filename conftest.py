"""
conftest.py
===========
Pytest configuration — adds the llm_pipeline/ root to sys.path
so all package imports (preprocessing.*, rag.*, analysis.*) work
when running:  pytest tests/ -v
"""
import sys
from pathlib import Path

# Insert project root (llm_pipeline/) into path — runs once at session start
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
