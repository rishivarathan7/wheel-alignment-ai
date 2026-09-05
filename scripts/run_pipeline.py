"""
Helper script to execute the monitoring pipeline directly.
"""

from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import main

if __name__ == "__main__":
    main()
