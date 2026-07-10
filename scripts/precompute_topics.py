"""
Precompute pseudo-topic cluster IDs using BERTopic.
Run once — saves to data/cache/topic_clusters.pt

Usage:
    python scripts/precompute_topics.py --year 2024,2025 --n-topics 20
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.data_loader import load_pan_labeled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, default="2024,2025")
    parser.add_argument("--n-topics", type=int, default=20)
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    args = parser.parse_args()

    years = [y.strip() for y in args.year.split(",")]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load all documents
    all_docs = []
    for year in years:
        for tier in ["easy", "medium", "hard"]:
            train, val, test = load_pan_labeled("data/raw", year=year, tier=tier)
            all_docs.extend(train + val + test)

    print(f"Total documents: {len(all_docs)}")

    # Collect all sentences with doc/sentence indices
    print("Collecting sentences...")
    all_sentences = []
    doc_sent_map = []  # (doc_idx, sent_idx) for each sentence
    for doc_idx, doc in enumerate(all_docs):
        for sent_idx, sent in enumerate(doc.sentences):
            all_sentences.append(sent)
            doc_sent_map.append((doc_idx, sent_idx))

    print(f"Total sentences: {len(all_sentences)}")

    # Encode with sentence-transformers (CPU is fine, one-time cost)
    print("Encoding sentences with all-MiniLM-L6-v2 (this takes a few minutes)...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sbert.encode(
        all_sentences,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Cluster with BERTopic
    print(f"Running BERTopic with nr_topics={args.n_topics}...")
    t0 = time.time()
    from bertopic import BERTopic
    from sklearn.cluster import MiniBatchKMeans
    from umap import UMAP

    umap_model = UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine", random_state=42)
    cluster_model = MiniBatchKMeans(n_clusters=args.n_topics, random_state=42)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=cluster_model,
        nr_topics=args.n_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(all_sentences, embeddings=embeddings)
    print(f"BERTopic done in {time.time() - t0:.1f}s")

    # Build per-boundary topic labels
    # For boundary i (between sentence i and i+1), use the topic of sentence i+1
    # (the "incoming" sentence's topic is what we want to disentangle)
    print("Building per-boundary topic labels...")
    boundary_topics = {}  # doc_id -> list of topic_ids for each boundary

    sent_topics = {}  # (doc_idx, sent_idx) -> topic_id
    for idx, (doc_idx, sent_idx) in enumerate(doc_sent_map):
        sent_topics[(doc_idx, sent_idx)] = max(topics[idx], 0)  # -1 (outlier) -> 0

    for doc_idx, doc in enumerate(all_docs):
        n_boundaries = len(doc.sentences) - 1
        doc_topics = []
        for i in range(n_boundaries):
            # Use average of both sentences' topics? No — just use sentence i+1
            t = sent_topics.get((doc_idx, i + 1), 0)
            doc_topics.append(t)
        boundary_topics[doc.doc_id] = doc_topics

    # Save
    out_path = cache_dir / "topic_clusters.pt"
    torch.save({
        "n_topics": args.n_topics,
        "boundary_topics": boundary_topics,
        "years": years,
    }, out_path)

    print(f"\nSaved to {out_path}")
    print(f"  n_topics: {args.n_topics}")
    print(f"  docs with topics: {len(boundary_topics)}")

    # Print topic distribution
    all_t = [t for ts in boundary_topics.values() for t in ts]
    unique, counts = np.unique(all_t, return_counts=True)
    print(f"  topic distribution: {len(unique)} unique topics")
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
        print(f"    topic {u}: {c} boundaries ({100*c/len(all_t):.1f}%)")


if __name__ == "__main__":
    main()