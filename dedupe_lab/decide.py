"""Duplicate decision rules. Two systems are compared on the same pairs:

 baseline : Jaccard over raw page text (what a naive near-dup filter does)
 improved : strip learned boilerplate, then Jaccard OR containment-with-length-guard
"""
from .text import shingles, jaccard, containment


def baseline_score(page_a: str, page_b: str) -> float:
    return jaccard(shingles(page_a), shingles(page_b))


def improved_signals(body_a: str, body_b: str):
    sa, sb = shingles(body_a), shingles(body_b)
    j = jaccard(sa, sb)
    if len(sa) <= len(sb):
        c, ratio = containment(sa, sb), len(sa) / max(1, len(sb))
    else:
        c, ratio = containment(sb, sa), len(sb) / max(1, len(sa))
    return j, c, ratio


def improved_decide(sig, t_j, t_c, min_ratio):
    j, c, ratio = sig
    return j >= t_j or (c >= t_c and ratio >= min_ratio)
