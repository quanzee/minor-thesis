"""
LLM-as-judge scoring of conversation openings.

Scores the first utterance of each conversation, where the initiating
agent (agent_a) also spoke first, against the day's scenario anchor
(config.DAY_ANCHORS). The judge reads the whole utterance and scores its
overall stance on the day's Hofstede dimension, taking qualifications and
balancing statements into account. This whole-utterance ("holistic")
scoring gives the module, output folder and scoring_method their names.
These scores are the opening-stance measure read by
opening_trajectories.py, and are computed the same way for single_tier
and two_tier runs.

Source:
  <run_dir>/logs/interactions.jsonl
      Records which agent initiated each conversation and holds the full
      dialogue, including conversations whose post-conversation summary
      call failed and so have no chat node in the memory snapshot.

Run from the project root, straight after simulation.py (default run dir
is the project root, where simulation.py writes logs/ and results/):
    python -m llm_as_judge.rescore_openings_holistic
Other run directories (relative to the project root):
    python -m llm_as_judge.rescore_openings_holistic --run c2=past_runs/<dir>

Output, per run:
  <run_dir>/results/llm_as_judge_openings_holistic/llm_as_judge_results.jsonl
      One record per scored opening: agent, partner, culture,
      partner_culture, day, round, dimension_name, first_utterance,
      before_score, before_score_reason, before_score_rationale,
      before_extracted_span and scoring_method ("holistic_opening").
      before_extracted_span is always None and is kept only because
      opening_trajectories.py reads it.
  token_usage.jsonl in the same folder.

Score reasons: "ok"; "abstained" (judge returned NONE); "error" (API call
failed after retries, a data gap rather than a judge decision).
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import openai

_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

from llm_as_judge.config import DAY_ANCHORS, JUDGE_MODEL, JUDGE_TEMPERATURE, PROJECT_ROOT
from simulation_architecture.lib import token_tracker
from simulation_architecture.lib.config import client

OPENINGS_SUBDIR = "llm_as_judge_openings_holistic"

CALL_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5  # multiplied by the attempt number

_SCORE_RE = re.compile(r"SCORE:\s*(.*?)\s*(?:\n\s*REASON:|$)", re.IGNORECASE | re.DOTALL)
_REASON_RE = re.compile(r"REASON:\s*(.*)", re.IGNORECASE | re.DOTALL)


def build_prompt(text: str, anchor: dict[str, str]) -> str:
    return f"""You are scoring text for the cultural dimension {anchor['dimension_name']}.

Read the WHOLE text and judge the overall stance it expresses on this dimension. Take
qualifications, hedges and balancing statements into account: if the text weighs both
sides, the score should reflect that balance rather than its most explicit phrase.

Rate the text on a 1-100 scale for this dimension, where:
1-20 = strongly {anchor['pole_low']} — e.g. "{anchor['example_low']}"
40-60 = mixed/balanced — e.g. "{anchor['example_mid']}"
80-100 = strongly {anchor['pole_high']} — e.g. "{anchor['example_high']}"
If the text expresses no stance on this dimension at all, the score is NONE.

Then, in one sentence, explain which parts of the text determined the score and why they
reveal a stance on THIS SPECIFIC dimension, not just a generally positive/warm/team-oriented
tone. A phrase like "foster team spirit" or "enhance team dynamics" is NOT by itself
evidence for any particular dimension unless it specifically implies a position on
{anchor['dimension_name']}.

Respond in EXACTLY this format, nothing else:
SCORE: <a number 1-100, or NONE>
REASON: <one sentence>

Text: {text}"""


def parse_response(raw: str) -> tuple[float | None, str, str | None]:
    """Returns (score, reason, rationale)."""
    score_match = _SCORE_RE.search(raw)
    reason_match = _REASON_RE.search(raw)
    rationale = reason_match.group(1).strip() if reason_match else None
    score_text = (score_match.group(1) if score_match else raw).strip()

    if score_text.upper().startswith("NONE"):
        return None, "abstained", rationale
    num = re.search(r"-?\d+(\.\d+)?", score_text)
    if num:
        return float(num.group()), "ok", rationale
    print(f"[warn] unparseable judge response, treating as abstained: {raw!r}", flush=True)
    return None, "abstained", rationale


def score_holistic(text: str, anchor: dict[str, str], tracker=None) -> tuple[float | None, str, str | None]:
    prompt = build_prompt(text, anchor)
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=JUDGE_TEMPERATURE,
                timeout=CALL_TIMEOUT_SECONDS,
            )
            if tracker:
                tracker.add(response.usage)
            return parse_response(response.choices[0].message.content.strip())
        except (openai.APITimeoutError, openai.APIConnectionError,
                httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            print(f"[warn] scoring call failed (attempt {attempt}/{MAX_RETRIES}): "
                  f"{type(e).__name__}: {e}", flush=True)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    print(f"[warn] giving up after {MAX_RETRIES} attempts (reason=error). Last error: {last_error}",
          flush=True)
    return None, "error", None


class HolisticCache:
    """Caches scores for identical (text, anchor) pairs within a run."""

    def __init__(self, tracker=None):
        self._cache: dict[tuple[str, str], tuple] = {}
        self.tracker = tracker
        self.calls = 0
        self.hits = 0

    def score(self, text: str, anchor: dict[str, str]) -> tuple[float | None, str, str | None]:
        key = (text, anchor["example_low"])  # example text is unique per day anchor
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.calls += 1
        self._cache[key] = score_holistic(text, anchor, tracker=self.tracker)
        return self._cache[key]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def start_tracker(out_dir: Path, label: str):
    tracker = token_tracker.TokenTracker()
    tracker.start_run(out_dir / "token_usage.jsonl", run_label=label)
    return tracker


def finish_tracker(tracker, out_dir: Path, label: str) -> None:
    tracker.report(label)
    tracker.append_log(label, out_dir / "token_usage.jsonl")


def score_openings(label: str, run_dir: Path) -> None:
    """Scores first-speaker openings for one run."""
    interactions = load_jsonl(run_dir / "logs" / "interactions.jsonl")

    selected, counts = [], Counter()
    for c in interactions:
        counts["conversations"] += 1
        dialogue = c.get("dialogue") or []
        if not dialogue:
            counts["empty_dialogue"] += 1
            print(f"[info] {label}: empty dialogue, {c['agent_a']}->{c['agent_b']} "
                  f"day {c['day']} {c['round']}; excluded", flush=True)
            continue
        if dialogue[0][0] != c["agent_a"]:
            counts["initiator_not_first"] += 1
            print(f"[info] {label}: {c['agent_a']} initiated day {c['day']} {c['round']} "
                  f"with {c['agent_b']} but did not speak first; excluded", flush=True)
            continue
        if c["day"] not in DAY_ANCHORS:
            counts["no_anchor"] += 1
            print(f"[warn] {label}: no anchor for day {c['day']}; excluded", flush=True)
            continue
        selected.append(c)

    out_dir = run_dir / "results" / OPENINGS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    run_label = f"Holistic openings ({label})"
    tracker = start_tracker(out_dir, run_label)
    cache = HolisticCache(tracker)
    reasons = Counter()
    start = time.monotonic()

    print(f"\n{label}: scoring {len(selected)} first-speaker openings "
          f"({counts['conversations']} conversations logged)", flush=True)
    with open(out_dir / "llm_as_judge_results.jsonl", "w", encoding="utf-8") as f:
        for i, c in enumerate(selected, start=1):
            anchor = DAY_ANCHORS[c["day"]]
            opening = c["dialogue"][0][1]
            score, reason, rationale = cache.score(opening, anchor)
            reasons[reason] += 1
            rec = {
                "agent": c["agent_a"], "partner": c["agent_b"],
                "culture": c["culture_a"], "partner_culture": c["culture_b"],
                "day": c["day"], "round": c["round"],
                "dimension_name": anchor["dimension_name"],
                "first_utterance": opening,
                "before_score": score,
                "before_score_reason": reason,
                "before_score_rationale": rationale,
                "before_extracted_span": None,
                "scoring_method": "holistic_opening",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(selected)}] {c['agent_a']} day={c['day']} {c['round']} "
                  f"score={score} [{reason}] | {time.monotonic() - start:.0f}s", flush=True)

    print(f"\n{label} openings: {len(selected)} scored of {counts['conversations']} conversations "
          f"| excluded: {counts['initiator_not_first']} initiator did not speak first, "
          f"{counts['empty_dialogue']} empty, {counts['no_anchor']} no anchor "
          f"| ok {reasons['ok']}, abstained {reasons['abstained']}, error {reasons['error']} "
          f"| {cache.calls} API calls, {cache.hits} cache hits")
    finish_tracker(tracker, out_dir, run_label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", default=None, metavar="LABEL=RUN_DIR",
                        help="Label and run directory relative to the project root (repeatable). "
                             "Default: current=. (the project root, where simulation.py writes)")
    args = parser.parse_args()

    for spec in args.run or ["current=."]:
        label, _, run_dir = spec.partition("=")
        if not run_dir:
            parser.error(f"--run must look like LABEL=RUN_DIR, got {spec!r}")
        path = (PROJECT_ROOT / run_dir).resolve()
        if not (path / "logs" / "interactions.jsonl").exists():
            parser.error(f"{label}: {path}/logs/interactions.jsonl not found")
        score_openings(label, path)


if __name__ == "__main__":
    main()