"""
Precompute and cache SBERT embeddings for all datasets.

Run once from project root:
    python scripts/precompute_embeddings.py

Options:
    --year    2023 | 2024 | 2025 | all   (default: all)
    --tier    easy | medium | hard | all  (default: all)
    --muld    include MuLD AO3            (default: True)
    --device  cuda | cpu                  (default: auto)

Examples:
    # All datasets
    python scripts/precompute_embeddings.py

    # Just PAN 2024 hard (fastest for development)
    python scripts/precompute_embeddings.py --year 2024 --tier hard --no-muld

    # CPU only
    python scripts/precompute_embeddings.py --device cpu
"""

import argparse
from pathlib import Path

from src.utils.data_loader import load_pan_labeled, MuLDDataset, split_documents
from src.utils.sbert_cache import SBERTEncoder, compute_and_save

CACHE_DIR  = Path("data/processed/embeddings")
DATA_ROOT  = Path("data/raw")
SEED       = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--year",   default="all", choices=["2023","2024","2025","all"])
    p.add_argument("--tier",   default="all", choices=["easy","medium","hard","all"])
    p.add_argument("--muld",   default=True,  action=argparse.BooleanOptionalAction)
    p.add_argument("--device", default=None,  help="cuda | cpu (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()

    encoder = SBERTEncoder(device=args.device)

    years = ["2023","2024","2025"] if args.year == "all" else [args.year]
    tiers = ["easy","medium","hard"] if args.tier == "all" else [args.tier]

    # --- PAN datasets ---
    for year in years:
        for tier in tiers:
            print(f"\n{'='*50}")
            print(f"PAN {year} — {tier}")
            print(f"{'='*50}")

            try:
                train, val, test = load_pan_labeled(
                    DATA_ROOT, year=year, tier=tier, seed=SEED
                )
            except FileNotFoundError as e:
                print(f"[Skip] {e}")
                continue

            source = f"pan{year}"
            for split_name, docs in [("train", train), ("val", val), ("test", test)]:
                cache_path = CACHE_DIR / f"{source}_{tier}_{split_name}.pt"
                compute_and_save(
                    docs=docs,
                    cache_path=cache_path,
                    encoder=encoder,
                    source=source,
                    tier=tier,
                    split=split_name,
                )

    # --- MuLD AO3 ---
    if args.muld:
        print(f"\n{'='*50}")
        print("MuLD AO3")
        print(f"{'='*50}")

        try:
            docs = MuLDDataset("data/raw/muld_ao3").load()
            train, val, test = split_documents(docs, seed=SEED)

            for split_name, split_docs in [("train", train), ("val", val), ("test", test)]:
                cache_path = CACHE_DIR / f"muld_ao3_{split_name}.pt"
                compute_and_save(
                    docs=split_docs,
                    cache_path=cache_path,
                    encoder=encoder,
                    source="muld_ao3",
                    tier=None,
                    split=split_name,
                )
        except Exception as e:
            print(f"[MuLD] Error: {e}")

    print("\nDone. List cached files:")
    for f in sorted(CACHE_DIR.glob("*.pt")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
