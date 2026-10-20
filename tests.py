from dedupe_lab.text import norm_line, shingles, jaccard, BoilerplateModel
from dedupe_lab.decide import improved_signals, improved_decide
from dedupe_lab.minhash import MinHasher

def test_norm_line_collapses_volatile_digits():
    assert norm_line("Page views: 1234") == norm_line("page   views: 99")

def test_boilerplate_removed_but_content_kept():
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    pages = [f"NAV one\nNAV two\nunique content about {w} topic" for w in words]
    m = BoilerplateModel(min_pages=3, min_frac=0.5).fit(pages)
    out = m.strip(pages[0])
    assert "NAV" not in out and "unique content" in out

def test_identical_bodies_are_duplicates_and_different_are_not():
    a = "the quick brown fox jumps over the lazy dog near the river bank today"
    b = "completely unrelated text about database indexing and query planning here now"
    assert improved_decide(improved_signals(a, a), 0.4, 0.7, 0.5)
    assert not improved_decide(improved_signals(a, b), 0.4, 0.7, 0.5)

def test_minhash_estimates_jaccard_roughly():
    a = {tuple(f"w{i}{j}" for j in range(3)) for i in range(100)}
    b = {tuple(f"w{i}{j}" for j in range(3)) for i in range(50, 150)}
    mh = MinHasher(256)
    est = mh.estimate(mh.signature(a), mh.signature(b))
    assert abs(est - jaccard(a, b)) < 0.1

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("ok", n)
