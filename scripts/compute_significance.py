#!/usr/bin/env python3
"""
Paired bootstrap significance test + McNemar's test between two models'
predictions on the same test set (document-level resampling).
Vectorized for speed — avoids Python-level loops over documents.
"""
import argparse
import numpy as np
from scipy.stats import chi2


def reconstruct_doc_counts(preds, labels, lengths):
    """Precompute per-document (tp, fp, fn) counts — vectorized."""
    n_docs = len(lengths)
    tp = np.zeros(n_docs, dtype=np.int64)
    fp = np.zeros(n_docs, dtype=np.int64)
    fn = np.zeros(n_docs, dtype=np.int64)
    idx = 0
    for i, n in enumerate(lengths):
        p = preds[idx:idx+n]
        l = labels[idx:idx+n]
        tp[i] = np.sum((p == 1) & (l == 1))
        fp[i] = np.sum((p == 1) & (l == 0))
        fn[i] = np.sum((p == 0) & (l == 1))
        idx += n
    return tp, fp, fn


def f1_from_counts(tp_sum, fp_sum, fn_sum):
    prec = np.where((tp_sum + fp_sum) > 0, tp_sum / np.maximum(tp_sum + fp_sum, 1), 0.0)
    rec  = np.where((tp_sum + fn_sum) > 0, tp_sum / np.maximum(tp_sum + fn_sum, 1), 0.0)
    denom = prec + rec
    f1 = np.where(denom > 0, 2 * prec * rec / np.maximum(denom, 1e-12), 0.0)
    return f1


def paired_bootstrap_vectorized(a_tp, a_fp, a_fn, b_tp, b_fp, b_fn, n_boot=10000, seed=42):
    n_docs = len(a_tp)
    rng = np.random.default_rng(seed)

    f1_a = f1_from_counts(a_tp.sum(), a_fp.sum(), a_fn.sum())
    f1_b = f1_from_counts(b_tp.sum(), b_fp.sum(), b_fn.sum())
    observed_delta = f1_a - f1_b

    # Generate all bootstrap index samples at once: (n_boot, n_docs)
    idx_matrix = rng.integers(0, n_docs, size=(n_boot, n_docs))

    # Vectorized sum over sampled docs for each bootstrap iteration
    a_tp_boot = a_tp[idx_matrix].sum(axis=1)
    a_fp_boot = a_fp[idx_matrix].sum(axis=1)
    a_fn_boot = a_fn[idx_matrix].sum(axis=1)
    b_tp_boot = b_tp[idx_matrix].sum(axis=1)
    b_fp_boot = b_fp[idx_matrix].sum(axis=1)
    b_fn_boot = b_fn[idx_matrix].sum(axis=1)

    fa_boot = f1_from_counts(a_tp_boot, a_fp_boot, a_fn_boot)
    fb_boot = f1_from_counts(b_tp_boot, b_fp_boot, b_fn_boot)
    deltas = fa_boot - fb_boot

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_value = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "f1_a": float(f1_a), "f1_b": float(f1_b), "observed_delta": float(observed_delta),
        "ci_95": (float(ci_low), float(ci_high)), "p_value": float(p_value),
    }


def mcnemar_test(a_preds, b_preds, labels):
    a_correct = (a_preds == labels)
    b_correct = (b_preds == labels)
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - chi2.cdf(stat, df=1)
    return {"n01": n01, "n10": n10, "statistic": stat, "p_value": p_value}


def main(args):
    a = np.load(args.a)
    b = np.load(args.b)

    a_preds   = a[f"{args.tier}__preds"]
    a_labels  = a[f"{args.tier}__labels"]
    a_lengths = a[f"{args.tier}__lengths"]
    b_preds   = b[f"{args.tier}__preds"]
    b_labels  = b[f"{args.tier}__labels"]
    b_lengths = b[f"{args.tier}__lengths"]

    if len(a_preds) != len(b_preds) or not np.array_equal(a_labels, b_labels):
        print("WARNING: prediction arrays differ in length or labels don't match.")
        print(f"  {args.a_name}: {len(a_preds)} pairs")
        print(f"  {args.b_name}: {len(b_preds)} pairs")
        return

    print(f"Comparing {args.a_name} vs {args.b_name} on {args.tier}")
    print(f"  ({len(a_lengths)} documents, {len(a_preds)} total boundary pairs)\n")

    a_tp, a_fp, a_fn = reconstruct_doc_counts(a_preds, a_labels, a_lengths)
    b_tp, b_fp, b_fn = reconstruct_doc_counts(b_preds, b_labels, b_lengths)

    boot = paired_bootstrap_vectorized(a_tp, a_fp, a_fn, b_tp, b_fp, b_fn, n_boot=args.n_boot)
    print("--- Paired Bootstrap (document-level resampling, vectorized) ---")
    print(f"  {args.a_name} F1: {boot['f1_a']:.4f}")
    print(f"  {args.b_name} F1: {boot['f1_b']:.4f}")
    print(f"  Delta: {boot['observed_delta']:.4f}")
    print(f"  95% CI: [{boot['ci_95'][0]:.4f}, {boot['ci_95'][1]:.4f}]")
    print(f"  p-value: {boot['p_value']:.4f}  {'(significant at 0.05)' if boot['p_value'] < 0.05 else '(NOT significant)'}")

    mc = mcnemar_test(a_preds, b_preds, a_labels)
    print("\n--- McNemar's Test (per-boundary-call level) ---")
    print(f"  n(A wrong, B right) = {mc['n01']}, n(A right, B wrong) = {mc['n10']}")
    print(f"  chi2 statistic: {mc['statistic']:.4f}")
    print(f"  p-value: {mc['p_value']:.4f}  {'(significant at 0.05)' if mc['p_value'] < 0.05 else '(NOT significant)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--tier", required=True)
    p.add_argument("--a-name", default="Model A")
    p.add_argument("--b-name", default="Model B")
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()
    main(args)