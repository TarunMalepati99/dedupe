"""Text normalisation, boilerplate learning, and similarity signals."""
import re
from collections import Counter

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


def norm_line(line: str) -> str:
    """Canonical form of a line for boilerplate matching (digits collapsed so
    volatile lines like dates / visitor counters still match across pages)."""
    return _DIGITS.sub("0", _WS.sub(" ", line.strip().lower()))


def words(text: str):
    return re.findall(r"[a-z0-9']+", text.lower())


def shingles(text: str, k: int = 5):
    w = words(text)
    if len(w) < k:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def jaccard(a, b) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def containment(small, big) -> float:
    return len(small & big) / len(small) if small else 0.0


class BoilerplateModel:
    """Learns which lines are page furniture from an *unlabelled* crawl sample:
    a (normalised) line that appears on many distinct pages is template, not
    content. No labels are used."""

    def __init__(self, min_pages: int = 3, min_frac: float = 0.2):
        """min_frac is the fraction of a HOST's pages a line must appear on to be
        called template. Fit one model per host: a content line can repeat across
        mirrors of one document, but it will not appear on a fifth of a host."""
        self.min_pages = min_pages
        self.min_frac = min_frac
        self.df = Counter()
        self.n_pages = 0
        self.drop = set()

    def fit(self, pages):
        for p in pages:
            self.n_pages += 1
            for ln in {norm_line(x) for x in p.splitlines() if x.strip()}:
                self.df[ln] += 1
        cutoff = max(self.min_pages, int(self.min_frac * self.n_pages))
        self.drop = {ln for ln, c in self.df.items() if c >= cutoff}
        return self

    def strip(self, page: str) -> str:
        return "\n".join(x for x in page.splitlines()
                         if x.strip() and norm_line(x) not in self.drop)
