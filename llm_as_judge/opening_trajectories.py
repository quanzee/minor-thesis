"""
Opening-stance trajectories from existing LLM-as-judge scores.

Uses only the judge's score for each agent's FIRST UTTERANCE when that agent
OPENED the conversation (first speaker). No new judge calls are made.

Why first speakers only: a second speaker's first utterance is already a
reply to the opener and is pulled toward the partner's stance, so it mixes
the agent's own position with its partner's. Every agent opens exactly one
conversation per round, so first-speaker openings give a balanced design:
3 agents per culture per round (Morning, Afternoon, Evening) on each day.

Each day has its own scenario and Hofstede dimension, so trajectories are
WITHIN a day (Morning -> Afternoon -> Evening), not across days.

Reads the scores written by rescore_openings_holistic.py. Run from the
project root (label=run_dir, relative to the project root):
    python -m llm_as_judge.opening_trajectories --run openai=past_runs/<dir>
Several models at once:
    python -m llm_as_judge.opening_trajectories \
        --run openai=past_runs/<dir> --run deepseek=past_runs/<dir> --run grok=past_runs/<dir>

Each run_dir must contain:
    logs/interactions.jsonl
    results/llm_as_judge_openings_holistic/llm_as_judge_results.jsonl

Outputs (in --out-dir, default results/llm_as_judge_openings_holistic_summary/), one row per
model x day x ..., with a `model` column so models can be compared:
  openings_per_agent.csv     every first-speaker opening: score, partner
  openings_agent_change.csv  per agent per day: Morning/Afternoon/Evening
                             scores; n_later_rounds (0-2) used below;
                             signed_departure = mean(A - M, E - M)
                             and its up/down/flat label (direction);
                             abs_departure = mean(|A - M|, |E - M|) (magnitude)
                             Both use whichever later rounds exist (so a value
                             may rest on one round); None if Morning is missing
  openings_magnitude.csv     per culture per day (plus an "All" row): mean and
                             max of the agents' abs_departure (magnitude)
  openings_by_culture.csv   per culture per day per round: mean, min, max, n,
                             change from Morning; plus how many of the
                             culture's agents rose / fell / stayed that day
                             (by signed_departure)
  openings_spread.csv        per day per round: STDEV.P of the three culture
                             means, and its change from Morning
                             (negative = convergence, positive = divergence)
"""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROUNDS = ["Morning", "Afternoon", "Evening"]
CULTURES = ["Australia", "Japan", "Malaysia"]
DIMENSIONS = {1: "PDI", 2: "IDV", 3: "MAS", 4: "UAI", 5: "LTO", 6: "IVR"}
FLAT_THRESHOLD = 0  # "flat" means a signed_departure of exactly 0


def _key(day, round_, a, b):
    return (day, round_, frozenset((a, b)))


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_speaker_openings(model: str, run_dir: Path, results_subdir: str = "llm_as_judge_openings_holistic") -> list[dict]:
    conversations = {
        _key(c["day"], c["round"], c["agent_a"], c["agent_b"]): c
        for c in load_jsonl(run_dir / "logs" / "interactions.jsonl")
    }
    results = load_jsonl(run_dir / "results" / results_subdir / "llm_as_judge_results.jsonl")

    rows = []
    for r in results:
        conv = conversations.get(_key(r["day"], r["round"], r["agent"], r["partner"]))
        if conv is None:
            print(f"[warn] {model}: no conversation for {r['agent']}/{r['partner']} "
                  f"day {r['day']} {r['round']}, skipping")
            continue
        if conv["agent_a"] != r["agent"]:
            continue  # only initiators are included
        if conv["dialogue"][0][0] != r["agent"]:
            # the initiator's first utterance failed and the partner spoke
            # first, so the conversation has no valid opening
            print(f"[info] {model}: {r['agent']} initiated day {r['day']} {r['round']} "
                  f"with {r['partner']} but did not speak first; excluded")
            continue

        culture = conv["culture_a"] if conv["agent_a"] == r["agent"] else conv["culture_b"]
        partner_culture = conv["culture_b"] if conv["agent_a"] == r["agent"] else conv["culture_a"]
        rows.append({
            "model": model,
            "day": r["day"],
            "dimension": DIMENSIONS.get(r["day"], r.get("dimension_name")),
            "round": r["round"],
            "agent": r["agent"],
            "culture": culture,
            "partner": r["partner"],
            "partner_culture": partner_culture,
            "cross_cultural": culture != partner_culture,
            "opening_score": r["before_score"],
            "score_reason": r["before_score_reason"],
            "extracted_span": r["before_extracted_span"],
        })

    # Check for one opening per agent per round per day.
    seen = defaultdict(int)
    for row in rows:
        seen[(row["day"], row["round"], row["agent"])] += 1
    dupes = {k: v for k, v in seen.items() if v > 1}
    if dupes:
        print(f"[warn] {model}: agents opening more than once in a round: {dupes}")
    return rows


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def r1(x):
    return None if x is None else round(x, 1)


def _later_diffs(scores: dict) -> list:
    """Differences (later round - Morning) for whichever of Afternoon/Evening has a
    score. Empty if the Morning opening (the within-day baseline) is missing."""
    m = scores.get("Morning")
    if m is None:
        return []
    return [scores[r] - m for r in ("Afternoon", "Evening") if scores.get(r) is not None]


def signed_departure(scores: dict):
    """Mean signed shift of the agent's later openings from its Morning opening
    (direction), over whichever later rounds are available: mean(A - M, E - M), or
    just one of them if the other round is missing. Positive = toward the
    dimension's high pole. None if Morning is missing or no later round exists."""
    diffs = _later_diffs(scores)
    return sum(diffs) / len(diffs) if diffs else None


def abs_departure(scores: dict):
    """Mean distance of the agent's later openings from its Morning (within-day
    baseline) opening, over whichever later rounds are available (magnitude).
    Mirrors VSM13 magnitude (mean over Days 1-6 of |Day - Day 0|). None if Morning
    is missing or no later round exists."""
    diffs = _later_diffs(scores)
    return sum(abs(d) for d in diffs) / len(diffs) if diffs else None


def build_tables(rows: list[dict]):
    by_agent = defaultdict(dict)       # (model, day, culture, agent) -> {round: score}
    by_culture = defaultdict(list)     # (model, day, culture, round) -> [scores]
    for row in rows:
        by_agent[(row["model"], row["day"], row["culture"], row["agent"])][row["round"]] = row["opening_score"]
        by_culture[(row["model"], row["day"], row["culture"], row["round"])].append(row["opening_score"])

    # Per-agent change.
    agent_change = []
    for (model, day, culture, agent), scores in sorted(by_agent.items()):
        m, e = scores.get("Morning"), scores.get("Evening")
        change = signed_departure(scores)
        if change is None:
            direction = "missing"
        elif change > FLAT_THRESHOLD:
            direction = "up"
        elif change < -FLAT_THRESHOLD:
            direction = "down"
        else:
            direction = "flat"
        agent_change.append({
            "model": model, "day": day, "dimension": DIMENSIONS.get(day), "culture": culture,
            "agent": agent, "morning": m, "afternoon": scores.get("Afternoon"), "evening": e,
            "n_later_rounds": len(_later_diffs(scores)),
            "signed_departure": r1(change),
            "abs_departure": r1(abs_departure(scores)),
            "direction": direction,
        })

    # Magnitude per culture per day (plus an "All" row pooling every agent that day):
    # mean of each agent's abs_departure, i.e. how far its Afternoon and Evening
    # openings sit from its Morning opening, on average.
    grouped = defaultdict(list)
    for a in agent_change:
        if a["abs_departure"] is None:
            continue
        grouped[(a["model"], a["day"], a["culture"])].append(a["abs_departure"])
        grouped[(a["model"], a["day"], "All")].append(a["abs_departure"])
    magnitude_rows = []
    for (model, day, culture), deps in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], (CULTURES + ["All"]).index(kv[0][2]))):
        magnitude_rows.append({
            "model": model, "day": day, "dimension": DIMENSIONS.get(day), "culture": culture,
            "n": len(deps),
            "mean_abs_departure": r1(statistics.mean(deps)),
            "max_abs_departure": max(deps),
        })

    directions = defaultdict(lambda: defaultdict(int))
    for a in agent_change:
        directions[(a["model"], a["day"], a["culture"])][a["direction"]] += 1

    # Per-culture means by round.
    models_days = sorted({(r["model"], r["day"]) for r in rows})
    culture_rows, spread_rows = [], []
    for model, day in models_days:
        means_by_round = {}
        for culture in CULTURES:
            morning_mean = mean_or_none(by_culture[(model, day, culture, "Morning")])
            for round_ in ROUNDS:
                scores = [s for s in by_culture[(model, day, culture, round_)] if s is not None]
                mean = mean_or_none(scores)
                means_by_round.setdefault(round_, {})[culture] = mean
                d = directions[(model, day, culture)]
                culture_rows.append({
                    "model": model, "day": day, "dimension": DIMENSIONS.get(day),
                    "culture": culture, "round": round_, "n": len(scores),
                    "mean": r1(mean), "min": min(scores) if scores else None,
                    "max": max(scores) if scores else None,
                    "change_from_morning": r1(mean - morning_mean)
                    if (mean is not None and morning_mean is not None) else None,
                    "agents_up": d["up"], "agents_down": d["down"],
                    "agents_flat": d["flat"],
                })

        def spread(round_):
            vals = [v for v in means_by_round[round_].values() if v is not None]
            return statistics.pstdev(vals) if len(vals) == len(CULTURES) else None

        morning_spread = spread("Morning")
        for round_ in ROUNDS:
            s = spread(round_)
            spread_rows.append({
                "model": model, "day": day, "dimension": DIMENSIONS.get(day), "round": round_,
                **{f"mean_{c}": r1(means_by_round[round_][c]) for c in CULTURES},
                "spread_stdev_p": r1(s),
                "spread_change_from_morning": r1(s - morning_spread)
                if (s is not None and morning_spread is not None) else None,
            })

    return agent_change, magnitude_rows, culture_rows, spread_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[warn] nothing to write for {path.name}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):>4} rows -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=None, metavar="LABEL=RUN_DIR",
                        help="Model label and run directory (repeatable). "
                             "Default: openai=past_runs/C1")
    parser.add_argument("--results-subdir", default="llm_as_judge_openings_holistic",
                        help="Folder under <run_dir>/results/ holding llm_as_judge_results.jsonl. "
                             "Default: llm_as_judge_openings_holistic")
    parser.add_argument("--out-dir", default=None,
                        help="Default: results/llm_as_judge_openings_holistic_summary")
    args = parser.parse_args()

    runs = args.run or ["openai=past_runs/C1"]
    rows = []
    for spec in runs:
        label, _, run_dir = spec.partition("=")
        if not run_dir:
            parser.error(f"--run must look like LABEL=RUN_DIR, got {spec!r}")
        model_rows = first_speaker_openings(label, PROJECT_ROOT / run_dir, args.results_subdir)
        print(f"{label}: {len(model_rows)} first-speaker openings")
        rows.extend(model_rows)

    agent_change, magnitude_rows, culture_rows, spread_rows = build_tables(rows)

    out_dir_name = args.out_dir or ("results/llm_as_judge_openings" if args.results_subdir == "llm_as_judge"
                                    else "results/llm_as_judge_openings_holistic_summary")
    out_dir = PROJECT_ROOT / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "openings_per_agent.csv",
              sorted(rows, key=lambda r: (r["model"], r["day"], r["culture"], r["agent"],
                                          ROUNDS.index(r["round"]))))
    write_csv(out_dir / "openings_agent_change.csv", agent_change)
    write_csv(out_dir / "openings_magnitude.csv", magnitude_rows)
    write_csv(out_dir / "openings_by_culture.csv", culture_rows)
    write_csv(out_dir / "openings_spread.csv", spread_rows)


if __name__ == "__main__":
    main()