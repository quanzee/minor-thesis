"""
Computes WVS drift from the per-day results saved by run_wvs_measurement()
in simulation.py. No model calls are made.

Usage:
    python wvs/compute_wvs_drift.py
    python wvs/compute_wvs_drift.py --results_dir results/side_study

Reads results/wvs/wvs_day{N}.json for every day present and writes, per
agent, results/wvs/drift/{agent_name}_wvs_drift_summary.json containing
day 0 versus day N and day-to-day comparisons, each in the format returned
by wvs.compute_drift().
"""

import argparse
import json
import re
from pathlib import Path

import wvs

_BASE_DIR = Path(__file__).parent.parent


def load_day_file(path: Path) -> dict:
    """
    Loads a wvs_day{N}.json file and returns each agent's canonical
    (majority-vote) responses as {agent_name: responses}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        name: agent_data["canonical_responses"]
        for name, agent_data in data["agents"].items()
    }


def discover_day_files(wvs_dir: Path) -> dict:
    """Returns {day_int: Path} for every wvs_dayN.json found in wvs_dir."""
    day_files = {}
    for p in wvs_dir.glob("wvs_day*.json"):
        match = re.match(r"wvs_day(\d+)\.json$", p.name)
        if match:
            day_files[int(match.group(1))] = p
    return day_files


def main():
    parser = argparse.ArgumentParser(description="Compute day0-vs-dayN WVS drift from saved simulation results")
    parser.add_argument("--results_dir", type=Path, default=None,
                         help="Defaults to results/ at project root")
    args = parser.parse_args()

    results_dir = args.results_dir or (_BASE_DIR / "results")
    wvs_dir = results_dir / "wvs"

    day_files = discover_day_files(wvs_dir)
    if 0 not in day_files:
        print(f"ERROR: {wvs_dir / 'wvs_day0.json'} not found — cannot compute drift without a T0 baseline.")
        return

    if len(day_files) == 1:
        print(f"WARNING: only Day 0 found in {wvs_dir} — nothing to compare it against yet.")
        return

    baseline_by_agent = load_day_file(day_files[0])
    print(f"Loaded Day 0 baseline for {len(baseline_by_agent)} agents: {', '.join(sorted(baseline_by_agent))}")

    # drift relative to the Day 0 baseline
    drift_by_agent = {name: {} for name in baseline_by_agent}

    for day in sorted(d for d in day_files if d != 0):
        day_scores = load_day_file(day_files[day])
        print(f"Computing drift for Day {day}...")
        for agent_name, baseline_scores in baseline_by_agent.items():
            if agent_name not in day_scores:
                print(f"  WARNING: {agent_name} missing from Day {day} results — skipping.")
                continue
            drift_by_agent[agent_name][f"day0_vs_day{day}"] = wvs.compute_drift(
                baseline_scores, day_scores[agent_name]
            )

    # day-to-day drift
    for day in sorted(d for d in day_files if d != 0):
        prev_day = day - 1
        if prev_day not in day_files:
            continue
        prev_scores = load_day_file(day_files[prev_day])
        day_scores = load_day_file(day_files[day])
        for agent_name, prev_agent_scores in prev_scores.items():
            if agent_name not in day_scores:
                continue
            drift_by_agent[agent_name][f"day{prev_day}_vs_day{day}"] = wvs.compute_drift(
                prev_agent_scores, day_scores[agent_name]
            )

    drift_dir = wvs_dir / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)
    for agent_name, drift_summary in drift_by_agent.items():
        out_path = drift_dir / f"{agent_name}_wvs_drift_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(drift_summary, f, indent=2)
        print(f"Saved {out_path}")

    print(f"\nDone. Drift summaries for {len(drift_by_agent)} agents saved to {drift_dir}")


if __name__ == "__main__":
    main()
