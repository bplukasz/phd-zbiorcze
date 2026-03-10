# conftest.py — makes pytest treat this dir as a rootdir without importing __init__.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
