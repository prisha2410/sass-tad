"""
Inspect raw MuLD AO3 format to fix label parsing.

Run: python scripts/inspect_muld.py
"""
from datasets import load_from_disk

ds = load_from_disk("data/raw/muld_ao3")
print("Splits:", list(ds.keys()))

split = list(ds.keys())[0]
print(f"\nFirst row of '{split}' split:")
row = ds[split][0]
for k, v in row.items():
    val_preview = str(v)[:200] if v else None
    print(f"  {k!r}: {type(v).__name__} = {val_preview}")
