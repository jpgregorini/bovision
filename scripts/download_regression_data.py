#!/usr/bin/env python3
"""
Alias / convenience entry-point: delegates to collect_data.py.

If you prefer a separate file name, run:
    python scripts/download_regression_data.py

It simply imports and calls the main() function from collect_data.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_data import main  # noqa: E402

if __name__ == "__main__":
    main()
