"""Does MinHash-LSH candidate generation keep the true duplicates while avoiding
O(n^2) comparisons? Runs on the stripped bodies of the held-out-test pages."""
import json, collections
from dedupe_lab.text import BoilerplateModel, shingles
from dedupe_lab.minhash import MinHasher, lsh_candidates

B = json.load(open("data/benchmark.json")); pages, pairs = B["pages"], B["pairs"]
by_site = collections.defaultdict(list)
for p in pages.values(): by_site[p["site"]].append(p["text"])
bp = {s: BoilerplateModel().fit(t) for s, t in by_site.items()}
test_pages = {k: v for k, v in pages.items() if v["split"] == "test"}
mh = MinHasher(64)
sigs = {k: mh.signature(shingles(bp[v["site"]].strip(v["text"]))) for k, v in test_pages.items()}
cands = lsh_candidates(sigs, bands=16)
dup = {tuple(sorted((p["a"], p["b"]))) for p in pairs if p["split"] == "test" and p["label"] == 1}
n = len(sigs); total = n * (n - 1) // 2
print(f"test pages={n}  all pairs={total}  LSH candidate pairs={len(cands)} ({len(cands)/total:.2%} of all pairs)")
print(f"labelled duplicate pairs recalled by LSH: {len(dup & cands)}/{len(dup)} ({len(dup & cands)/len(dup):.1%})")
