import json
import numpy as np
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")

# List your 5 run files here
RUN_FILES = [
    RESULTS_DIR / "japan_day0_vsm13_run1.json",
    RESULTS_DIR / "japan_day0_vsm13_run2.json",
    RESULTS_DIR / "japan_day0_vsm13_run3.json",
    RESULTS_DIR / "japan_day0_vsm13_run4.json",
    RESULTS_DIR / "japan_day0_vsm13_run5.json",
]

DIMENSIONS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]

# ── Load runs ────────────────────────────────────────────────────────────────
runs = []
for f in RUN_FILES:
    path = Path(f)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {f}")
    with open(path) as fh:
        data = json.load(fh)
    runs.append(data["dimension_scores"])

# ── Compute stats ────────────────────────────────────────────────────────────
print(f"{'Dimension':<10} {'Mean':>8} {'SD':>8} {'Min':>8} {'Max':>8} {'Range':>8}")
print("-" * 52)

stats = {}
for dim in DIMENSIONS:
    scores = np.array([r[dim] for r in runs])
    mean   = np.mean(scores)
    sd     = np.std(scores, ddof=1)   # sample SD
    lo     = np.min(scores)
    hi     = np.max(scores)
    rang   = hi - lo
    stats[dim] = {"mean": mean, "sd": sd, "min": lo, "max": hi, "range": rang, "scores": scores.tolist()}
    print(f"{dim:<10} {mean:>8.1f} {sd:>8.1f} {lo:>8.0f} {hi:>8.0f} {rang:>8.0f}")

# ── Noise assessment ─────────────────────────────────────────────────────────
print("\nNoise assessment (SD < 5 = low, 5–10 = moderate, > 10 = high):")
for dim, s in stats.items():
    level = "low" if s["sd"] < 5 else "moderate" if s["sd"] <= 10 else "high"
    print(f"  {dim}: {level} (SD = {s['sd']:.1f})")

# ── Per-run scores for inspection ────────────────────────────────────────────
print("\nPer-run scores:")
print(f"{'Dimension':<10}", end="")
for i in range(len(runs)):
    print(f"  Run {i+1:>2}", end="")
print()
print("-" * (10 + 8 * len(runs)))
for dim in DIMENSIONS:
    print(f"{dim:<10}", end="")
    for s in stats[dim]["scores"]:
        print(f"  {s:>5.0f}", end="")
    print()