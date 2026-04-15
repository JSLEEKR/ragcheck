"""Allow `python -m ragcheck ...` invocation."""
from __future__ import annotations

import sys

from ragcheck.cli import main

if __name__ == "__main__":
    sys.exit(main())
