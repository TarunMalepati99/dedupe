"""MinHash + LSH banding: the scale path for candidate generation (exact
pairwise comparison is O(n^2)). Pure stdlib, deterministic."""
import hashlib
import random


def _h(x: tuple, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(repr((seed, x)).encode(), digest_size=8).digest(), "big")


class MinHasher:
    def __init__(self, num_perm: int = 64, seed: int = 7):
        rnd = random.Random(seed)
        self.seeds = [rnd.randrange(1 << 30) for _ in range(num_perm)]

    def signature(self, shingle_set):
        if not shingle_set:
            return tuple([0] * len(self.seeds))
        return tuple(min(_h(s, sd) for s in shingle_set) for sd in self.seeds)

    @staticmethod
    def estimate(sig_a, sig_b) -> float:
        return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


def lsh_candidates(signatures: dict, bands: int = 16):
    """signatures: id -> signature tuple. Returns set of candidate id pairs that
    collide in at least one band."""
    rows = len(next(iter(signatures.values()))) // bands
    buckets = {}
    for i, sig in signatures.items():
        for b in range(bands):
            buckets.setdefault((b, sig[b * rows:(b + 1) * rows]), []).append(i)
    cands = set()
    for ids in buckets.values():
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                cands.add(tuple(sorted((ids[x], ids[y]))))
    return cands
