"""
Download and save MuLD AO3 dataset to data/raw/muld_ao3/
Run from project root: python scripts/download_muld.py
"""
import os
import sys
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

SAVE_PATH = Path("data/raw/muld_ao3")

def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed. Running: pip install datasets==2.19.0")
        os.system("pip install datasets==2.19.0")
        from datasets import load_dataset

    if SAVE_PATH.exists():
        print(f"[!] {SAVE_PATH} already exists. Delete it first if you want to re-download.")
        return

    print("Downloading MuLD AO3 Style Change Detection dataset...")
    print("This may take a few minutes depending on your connection.\n")

    ds = load_dataset("ghomasHudson/muld", "AO3 Style Change Detection")

    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(SAVE_PATH))

    print(f"\nDone! Saved to {SAVE_PATH}")
    print(f"Splits: {list(ds.keys())}")
    for split, data in ds.items():
        print(f"  {split}: {len(data)} examples")

if __name__ == "__main__":
    main()