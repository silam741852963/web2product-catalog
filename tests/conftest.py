import sys
from pathlib import Path

# Put project root (the folder that contains /scraper and /tests) on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
