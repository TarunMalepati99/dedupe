"""Calibrate on dev, report on held-out test. Both systems get the same treatment
(threshold chosen on dev to maximise F1), so the comparison is fair."""
import json, itertools, collections, sys
from dedupe_lab.text import BoilerplateModel
from dedupe_lab.decide import baseline_score, improved_signals, improved_decide

B = json.load(open("data/benchmark.json"))
pages, pairs = B["pages"], B["pairs"]

# --- unlabelled crawl pool: learn boilerplate per host (no labels used) ---
by_site = collections.defaultdict(list)
for p in pages.values():
    by_site[p["site"]].append(p["text"])
bp = {s: BoilerplateModel().fit(t) for s, t in by_site.items()}
body = {pid: bp[p["site"]].strip(p["text"]) for pid, p in pages.items()}


def prf(preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels)); fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum((not p) and l for p, l in zip(preds, labels))
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0), tp, fp, fn


def split(name):
    return [p for p in pairs if p["split"] == name]


dev, test = split("dev"), split("test")
dl = [p["label"] for p in dev]; tl = [p["label"] for p in test]

# baseline: raw-page Jaccard
bs_dev = [baseline_score(pages[p["a"]]["text"], pages[p["b"]]["text"]) for p in dev]
bs_test = [baseline_score(pages[p["a"]]["text"], pages[p["b"]]["text"]) for p in test]
grid = [i / 100 for i in range(5, 100, 5)]
best_b = max(grid, key=lambda t: (prf([s >= t for s in bs_dev], dl)[2], -t))

# improved: stripped body, Jaccard OR containment w/ length guard
sg_dev = [improved_signals(body[p["a"]], body[p["b"]]) for p in dev]
sg_test = [improved_signals(body[p["a"]], body[p["b"]]) for p in test]
combos = list(itertools.product(grid, [0.7, 0.8, 0.9, 0.95], [0.5, 0.7]))
best_i = max(combos, key=lambda c: (prf([improved_decide(s, *c) for s in sg_dev], dl)[2], c[0]))


def report(title, preds, labels, items):
    P, R, F, tp, fp, fn = prf(preds, labels)
    print(f"{title:44s} P={P:.3f} R={R:.3f} F1={F:.3f}  (tp={tp} fp={fp} fn={fn})")
    k = collections.defaultdict(lambda: [0, 0])
    for p, y, it in zip(preds, labels, items):
        k[it["kind"]][0] += int(p == bool(y)); k[it["kind"]][1] += 1
    return {kk: f"{a}/{n}" for kk, (a, n) in sorted(k.items())}


print(f"dev pairs={len(dev)} test pairs={len(test)}  (documents are disjoint between splits)")
print(f"baseline threshold chosen on dev: {best_b};  improved (t_j,t_c,min_ratio) chosen on dev: {best_i}\n")
print("=== HELD-OUT TEST ===")
res = {}
res["baseline@0.5 (naive default)"] = report("baseline raw Jaccard @0.5", [s >= 0.5 for s in bs_test], tl, test)
res["baseline@dev-tuned"] = report(f"baseline raw Jaccard @{best_b} (dev-tuned)", [s >= best_b for s in bs_test], tl, test)
ip = [improved_decide(s, *best_i) for s in sg_test]
res["improved"] = report("improved (boilerplate-stripped, dev-tuned)", ip, tl, test)
print("\nper-kind accuracy on test:")
kinds = sorted({p["kind"] for p in test})
print(f"{'kind':26s}" + "".join(f"{k:>34s}" for k in res))
for kd in kinds:
    print(f"{kd:26s}" + "".join(f"{res[k][kd]:>34s}" for k in res))
print("\n=== TEST ERRORS of improved system ===")
for p, pr, sg in zip(test, ip, sg_test):
    if pr != bool(p["label"]):
        print(p["kind"], pages[p["a"]]["doc"], "|", pages[p["b"]]["doc"], "j=%.2f c=%.2f r=%.2f" % sg, "pred=", pr)
# boilerplate learned (sanity)
print("\nboilerplate lines learned per host:", {s: len(m.drop) for s, m in bp.items()})
