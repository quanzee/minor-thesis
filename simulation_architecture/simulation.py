"""
Main simulation loop for the cultural drift study.

Each RQ3 condition and RQ2 cross-model run is selected through the
memory_mode, frozen_core, use_reasoning, anti_compromise and model_key
flags (see run_all.py for the command used for each condition).

--frozen_core (two_tier only): the CoreBeliefs seeded on Day 0 are never
revised (end-of-day deep reflection is skipped), and VSM13, WVS and the
persona context read the fixed core beliefs plus the agent's most relevant
Tier 2 thoughts.

VSM13 and WVS are administered at T0 and at the end of every simulation day.

The default flags (two_tier without --frozen_core, with private reasoning)
run a revisable-core configuration that is not one of the study's
conditions. See run_all.py for the flags used for each condition.
"""

import json
import re
import yaml
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import sys

_SIM_DIR = Path(__file__).parent
_DATA_DIR = Path(__file__).parent / "data"
_LIB_DIR = Path(__file__).parent / "lib"
_BASE_DIR = Path(__file__).parent.parent
_WVS_DIR = Path(__file__).parent.parent / "wvs"

sys.path.insert(0, str(_SIM_DIR))
sys.path.insert(0, str(_LIB_DIR))
sys.path.insert(0, str(_WVS_DIR))

from agent import Agent
from town import Town, ROUNDS
from vsm13 import run_vsm13, compute_dimensions, retrieve_all_dimension_beliefs, log_vsm13_retrieval
from wvs import run_wvs, WVS_QUESTIONS, retrieve_all_beliefs, log_wvs_retrieval
from kendall_tau import compute_kendall_tau, HUMAN_SCORES
from config import MODEL_NAME, MODEL_NAME_3, MODEL_NAME_4, \
    client, deepseek_client, grok_client
from retrieve import get_embedding
from token_tracker import TokenTracker
from deep_reflect import run_deep_reflect
print("Imports loaded successfully")

# Model registry for RQ2 cross-model runs
CLIENT_REGISTRY = {
    "openai":   (client, MODEL_NAME),
    "deepseek": (deepseek_client, MODEL_NAME_3),
    "grok":     (grok_client, MODEL_NAME_4),
}

DEFAULT_TOTAL_DAYS = 6
DEFAULT_SCENARIOS_FILE = "scenarios_main.yaml"


@dataclass
class RunConfig:
    memory_mode: str        # "single_tier" or "two_tier"
    use_reasoning: bool
    anti_compromise: bool
    client: object
    model: str
    total_days: int
    logs_dir: Path
    results_dir: Path
    scenarios_path: Path
    skip_surveys: bool = False
    frozen_core: bool = False   # two_tier only

def build_run_config(args) -> "RunConfig":
    run_client, run_model = CLIENT_REGISTRY[args.model_key]
    logs_dir = _BASE_DIR / "logs"
    results_dir = _BASE_DIR / "results"
    scenarios_path = _DATA_DIR / args.scenarios_file
    return RunConfig(
        memory_mode=args.memory_mode,
        use_reasoning=args.use_reasoning,
        anti_compromise=args.anti_compromise,
        client=run_client,
        model=run_model,
        total_days=args.total_days,
        logs_dir=logs_dir,
        results_dir=results_dir,
        scenarios_path=scenarios_path,
        skip_surveys=args.skip_surveys,
        frozen_core=args.frozen_core,
    )


# Logging helpers
def append_log(filepath: Path, entry: dict):
    """Appends a single JSON entry to a .jsonl log file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def setup_logs(cfg: RunConfig):
    """Creates log and results directories."""
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)


def find_resume_day(cfg: RunConfig) -> int:
    """
    Returns the highest day with a complete memory_snapshots/dayXX folder,
    i.e. a day whose rounds, deep reflection (if applicable) and surveys
    all finished. Returns 0 for a fresh run.
    """
    snap_dir = cfg.results_dir / "memory_snapshots"
    if not snap_dir.exists():
        return 0
    days = []
    for d in snap_dir.iterdir():
        if d.is_dir() and d.name.startswith("day") and d.name != "day00":
            try:
                days.append(int(d.name.replace("day", "")))
            except ValueError:
                continue
    return max(days) if days else 0


def prune_logs_after(cfg: RunConfig, last_complete_day: int):
    """
    Before resuming, removes log entries from any day after the last
    completed one. A day that crashed part-way is re-run from its start,
    so without this its conversations would appear twice in the logs.
    """
    for path in sorted(cfg.logs_dir.glob("*.jsonl")):
        if path.name == "token_usage.jsonl":
            continue
        kept, removed = [], 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    removed += 1    # partially written line from an interrupted run
                    continue
                day = entry.get("day")
                if day is None:
                    m = re.match(r"Day (\d+)", str(entry.get("curr_time", "")))
                    day = int(m.group(1)) if m else None
                if day is not None and day > last_complete_day:
                    removed += 1
                    continue
                kept.append(line if line.endswith("\n") else line + "\n")
        if removed:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(kept)
            print(f"  Resume: removed {removed} entries after Day {last_complete_day} from {path.name}")


def check_run_config(cfg: RunConfig, model_key: str, resuming: bool):
    """
    Records the run's flags in results/run_config.json and, on resume,
    exits if the current flags differ from the saved ones, so that a run
    cannot change configuration part-way through.

    A fresh run always writes the file. A resume with no saved file
    writes one and continues.
    """
    path = cfg.results_dir / "run_config.json"
    current = {
        "memory_mode": cfg.memory_mode,
        "use_reasoning": cfg.use_reasoning,
        "anti_compromise": cfg.anti_compromise,
        "frozen_core": cfg.frozen_core,
        "model_key": model_key,
    }
    if resuming and path.exists():
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        diff = {k: (saved[k], v) for k, v in current.items() if k in saved and saved[k] != v}
        if diff:
            details = ", ".join(f"{k}: started with {a!r}, now {b!r}" for k, (a, b) in diff.items())
            sys.exit(f"Resume refused: this run was started with different flags ({details}). "
                     f"Re-run with the original flags.")
        return
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)


# VSM13 measurement
def run_measurement(agents: list, day: int, cfg: RunConfig, sim_time=None, tracker=None, n_runs=3):
    print(f"\n{'='*60}")
    print(f"VSM13 Measurement — Day {day} ({n_runs} runs per agent)")
    print(f"{'='*60}")

    cultures = ["Australia", "Malaysia", "Japan"]
    culture_responses = {c: {} for c in cultures}
    culture_explanations = {c: {} for c in cultures}
    culture_all_runs = {c: {} for c in cultures}
    culture_run_dimensions = {c: {} for c in cultures}

    for agent in agents:
        print(f"  Administering VSM13 to {agent.name} ({n_runs} runs)...")

        belief_contexts = retrieve_all_dimension_beliefs(agent, curr_time=sim_time)
        log_vsm13_retrieval(agent, sim_time, belief_contexts)

        all_runs = []
        last_explanations = {}
        per_run_dims = {}

        for run in range(n_runs):
            print(f"    Run {run + 1}/{n_runs}...")
            responses, explanations = run_vsm13(
                agent, curr_time=sim_time, tracker=tracker, belief_contexts=belief_contexts
            )
            all_runs.append(responses)
            last_explanations = explanations
            per_run_dims[f"run_{run + 1}"] = compute_dimensions(responses)

        averaged = {}
        for q in range(1, 25):
            averaged[q] = sum(r[q] for r in all_runs) / n_runs

        culture_responses[agent.country][agent.name] = averaged
        culture_explanations[agent.country][agent.name] = last_explanations
        culture_all_runs[agent.country][agent.name] = {
            f"run_{i+1}": {str(k): v for k, v in run.items()}
            for i, run in enumerate(all_runs)
        }
        culture_run_dimensions[agent.country][agent.name] = per_run_dims

    pooled = {}
    individual_dimensions = {c: {} for c in cultures}
    pooled_run_dimensions = {c: {} for c in cultures}

    for culture in cultures:
        agent_responses = culture_responses[culture]
        if not agent_responses:
            continue
        n = len(agent_responses)

        for name, responses in agent_responses.items():
            individual_dimensions[culture][name] = compute_dimensions(responses)

        pooled_means = {}
        for q in range(1, 25):
            pooled_means[q] = sum(
                agent_responses[name][q] for name in agent_responses
            ) / n
        pooled[culture] = compute_dimensions(pooled_means)

        for run_idx in range(n_runs):
            run_key = f"run_{run_idx + 1}"
            run_pooled_means = {}
            for q in range(1, 25):
                run_pooled_means[q] = sum(
                    culture_all_runs[culture][name][run_key][str(q)]
                    for name in culture_all_runs[culture]
                ) / n
            pooled_run_dimensions[culture][run_key] = compute_dimensions(run_pooled_means)

    tau_results = compute_kendall_tau(HUMAN_SCORES, pooled)
    tau_per_run = {}
    for run_idx in range(n_runs):
        run_key = f"run_{run_idx + 1}"
        run_pooled = {
            c: pooled_run_dimensions[c][run_key]
            for c in cultures if run_key in pooled_run_dimensions[c]
        }
        tau_per_run[run_key] = compute_kendall_tau(HUMAN_SCORES, run_pooled)

    output = {
        "day": day,
        "n_runs": n_runs,
        "individual_responses_averaged": {
            culture: {
                name: {str(k): v for k, v in resp.items()}
                for name, resp in responses.items()
            }
            for culture, responses in culture_responses.items()
        },
        "individual_all_runs": culture_all_runs,
        "individual_explanations_last_run": {
            culture: {
                name: {str(k): v for k, v in expl.items()}
                for name, expl in explanations.items()
            }
            for culture, explanations in culture_explanations.items()
        },
        "individual_dimension_scores": individual_dimensions,
        "individual_run_dimensions": culture_run_dimensions,
        "pooled_dimension_scores": pooled,
        "pooled_run_dimensions": pooled_run_dimensions,
        "kendall_tau": tau_results,
        "kendall_tau_per_run": tau_per_run,
    }

    vsm13_dir = cfg.results_dir / "vsm13"
    vsm13_dir.mkdir(parents=True, exist_ok=True)
    path = vsm13_dir / f"vsm13_day{day}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Average Kendall tau (Day {day}): {tau_results.get('average_tau', 'N/A'):.4f}")
    print(f"  Results saved to {path}")


# WVS measurement
def run_wvs_measurement(agents: list, day: int, cfg: RunConfig, sim_time=None, tracker=None, n_runs=3):
    """
    Administers the WVS items to each agent n_runs times and saves the
    results to results/wvs/wvs_day{day}.json.

    Unlike VSM13, which averages responses across runs, the canonical
    response per item is the majority vote across runs, because WVS items
    use different scale lengths and drift is measured as changed versus
    unchanged (see wvs.compute_drift). Ties are broken by the earliest
    response among the tied values.

    Beliefs are retrieved once per agent per day and reused across runs,
    so repeated runs capture generation variability only.

    Drift is computed separately, downstream of this function.

    Returns {agent_name: canonical_responses}.
    """
    print(f"\n{'='*60}")
    print(f"WVS Measurement — Day {day} ({n_runs} runs per agent)")
    print(f"{'='*60}")

    agent_results = {}

    for agent in agents:
        print(f"  Administering WVS to {agent.name} ({n_runs} runs)...")

        belief_contexts = retrieve_all_beliefs(agent, curr_time=sim_time)
        log_wvs_retrieval(agent, sim_time, belief_contexts)

        all_runs = []

        for run in range(n_runs):
            print(f"    Run {run + 1}/{n_runs}...")
            responses, explanations = run_wvs(
                agent, curr_time=sim_time, tracker=tracker, belief_contexts=belief_contexts
            )
            all_runs.append({
                "responses": responses,
                "explanations": explanations,
            })

        canonical_responses = {}
        canonical_explanations = {}
        for q_id in WVS_QUESTIONS:
            votes = [r["responses"][q_id] for r in all_runs]
            counts = Counter(votes)
            max_count = max(counts.values())
            winner = next(v for v in votes if counts[v] == max_count)
            canonical_responses[q_id] = winner
            winning_rep = next(r for r in all_runs if r["responses"][q_id] == winner)
            canonical_explanations[q_id] = winning_rep["explanations"][q_id]

        agent_results[agent.name] = {
            "culture": agent.country,
            "all_runs": all_runs,
            "canonical_responses": canonical_responses,
            "canonical_explanations": canonical_explanations,
        }

    wvs_dir = cfg.results_dir / "wvs"
    wvs_dir.mkdir(parents=True, exist_ok=True)
    path = wvs_dir / f"wvs_day{day}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "day": day,
            "n_runs": n_runs,
            "agents": agent_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"  WVS results saved to {path}")
    return {name: r["canonical_responses"] for name, r in agent_results.items()}


# Scenario helper functions
def load_scenarios(cfg: RunConfig):
    with open(cfg.scenarios_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["scenarios"]


def inject_scenarios(town, scenarios, day, round_name, sim_time, tracker=None):
    matching = [
        s for s in scenarios
        if s["day"] == day and s["round"] == round_name
    ]

    if not matching:
        return

    for scenario in matching:
        location = scenario["location"]
        agents_at_location = [
            a for a in town.agents
            if a.location == location
        ]
        if not agents_at_location:
            print(f"      WARNING: No agents at {location} for Day {day} {round_name} scenario")
            continue

        for agent in agents_at_location:
            agent.active_scenario = {
                "text": scenario["text"],
                "day": day,
                "round": round_name,
                "location": location,
                "sim_time": sim_time
            }
            print(f"      Scenario activated for {agent.name} ({location})")


# Round execution
def run_round(town: Town, round_name: str, day: int,
              sim_time: str, scenarios: list, cfg: RunConfig, tracker=None):
    print(f"\n  [{round_name} Round]")

    if round_name == "Morning":
        for agent in town.agents:
            if agent.active_scenario and agent.active_scenario["day"] < day:
                agent.active_scenario = None

    print(f"  Assigning locations...")
    town.assign_locations(round_name, sim_time, tracker=tracker)
    print(f"  Processing {len(town.agents)} agents...")
    town.current_round = round_name

    already_conversed = set()

    inject_scenarios(town, scenarios, day, round_name, sim_time, tracker)

    agents_this_round = list(town.agents)
    random.shuffle(agents_this_round)

    # Every agent chooses a partner and prepares its opening line before
    # any conversation in the round takes place.
    planned = []
    for agent in agents_this_round:
        print(f"    Planning {agent.name}...")

        co_present = town.get_co_present_agents(agent)
        available = [
            a for a in co_present
            if tuple(sorted([agent.name, a.name])) not in already_conversed
        ]

        target = agent.plan(available, sim_time, tracker=tracker) if available else None
        print(f"      Plan: {'talking to ' + target.name if target else 'no conversation'}")
        if not target:
            continue

        already_conversed.add(tuple(sorted([agent.name, target.name])))
        opening = agent.prepare_opening(target, sim_time, tracker=tracker)
        planned.append((agent, target, opening))

    # Conversations then take place in the same order.
    for agent, target, opening in planned:
        print(f"    {agent.name} ({agent.country}) → "
              f"{target.name} ({target.country})")

        conversation, reasoning_log = agent.converse(
            target, sim_time, tracker=tracker, opening=opening
        )

        agent.recent_partners.append(target.name)
        target.recent_partners.append(agent.name)

        agent.recent_partners = agent.recent_partners[-3:]
        target.recent_partners = target.recent_partners[-3:]

        append_log(
            cfg.logs_dir / "interactions.jsonl",
            {
                "day": day,
                "round": round_name,
                "agent_a": agent.name,
                "culture_a": agent.country,
                "agent_b": target.name,
                "culture_b": target.country,
                "location": agent.location,
                "cross_cultural": agent.country != target.country,
                "reasoning": reasoning_log,
                "dialogue": conversation
            }
        )

    # Reflection runs once all conversations in the round have finished.
    for agent in agents_this_round:
        reflected, focal_points, all_insights = agent.reflect(sim_time, tracker=tracker)

        if reflected:
            append_log(
                cfg.logs_dir / "reflections.jsonl",
                {
                    "day": day,
                    "round": round_name,
                    "agent": agent.name,
                    "culture": agent.country,
                    "focal_points": focal_points,
                    "insights": all_insights
                }
            )


# Main simulation loop
def run_simulation(agents: list, cfg: RunConfig, start_day: int = 1, skip_t0_measurement: bool = False):
    """
    Runs the simulation for cfg.total_days days with three rounds per day,
    administering VSM13 and WVS at the end of each day. Deep reflection
    (the Tier 1 update) runs for two_tier agents unless cfg.frozen_core
    is set.
    """
    scenarios = load_scenarios(cfg)
    setup_logs(cfg)
    town = Town(agents)
    tracker = TokenTracker()

    TOKEN_LOG = cfg.logs_dir / "token_usage.jsonl"
    tracker.start_run(TOKEN_LOG, run_label="Test Dimension 1 Simulation")

    if start_day == 1 and not skip_t0_measurement:
        if not cfg.skip_surveys:
            run_measurement(agents, day=0, cfg=cfg, sim_time="Day 00", tracker=tracker)
            tracker.report("Day 0 (T0 Measurement)")
            tracker.append_log("Day 00 T0 Measurement", TOKEN_LOG)
            tracker.reset()

            run_wvs_measurement(agents, day=0, cfg=cfg, sim_time="Day 00", tracker=tracker)
            tracker.report("Day 0 (T0 WVS Measurement)")
            tracker.append_log("Day 00 T0 WVS Measurement", TOKEN_LOG)
            tracker.reset()

        snapshot_dir = cfg.results_dir / "memory_snapshots" / "day00"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for agent in agents:
            agent.save(str(snapshot_dir))

        if cfg.memory_mode == "two_tier":
            core_snapshot_dir = cfg.results_dir / "core_snapshots" / "day00"
            core_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for agent in agents:
                agent.core.save(str(core_snapshot_dir / f"{agent.name}_core.json"))
    elif start_day == 1 and skip_t0_measurement:
        print("Skipping Day 0 measurement - using pre-loaded snapshot.")
    else:
        print(f"Resuming from Day {start_day} (skipping T0 measurement and completed days)")

    try:
        for day in range(start_day, cfg.total_days + 1):
            tracker.reset()
            print(f"\n{'='*60}")
            print(f"Day {day}")
            print(f"{'='*60}")

            town.current_day = day

            for round_name in ROUNDS:
                if round_name == "Morning":
                    sim_time = f"Day {day:02d}, Morning"
                elif round_name == "Afternoon":
                    sim_time = f"Day {day:02d}, Afternoon"
                else:
                    sim_time = f"Day {day:02d}, Evening"

                run_round(town, round_name, day, sim_time, scenarios, cfg, tracker)

            tracker.report(f"Day {day}")
            tracker.append_log(f"Day {day:02d} Rounds", TOKEN_LOG)

            # End-of-day deep reflection updates CoreBeliefs (two_tier only).
            # It is skipped when the core is frozen.
            if cfg.memory_mode == "two_tier":
                if cfg.frozen_core:
                    print(f"\n  [Core frozen — deep reflection skipped, Day {day}]")
                else:
                    print(f"\n  [Deep Reflection — Day {day}]")
                    eod_time = f"Day {day:02d}, Evening"
                    for agent in agents:
                        deep_result = run_deep_reflect(
                            agent, current_day=day, curr_time=eod_time,
                            client=cfg.client, model=cfg.model, tracker=tracker
                        )
                        append_log(cfg.logs_dir / "deep_reflections.jsonl", deep_result)

                    tracker.report(f"Day {day} Deep Reflection")
                    tracker.append_log(f"Day {day:02d} Deep Reflection", TOKEN_LOG)

                # Core snapshots are saved daily, including when frozen, and
                # are used when resuming.
                core_snapshot_dir = cfg.results_dir / "core_snapshots" / f"day{day:02d}"
                core_snapshot_dir.mkdir(parents=True, exist_ok=True)
                for agent in agents:
                    agent.core.save(str(core_snapshot_dir / f"{agent.name}_core.json"))

            if not cfg.skip_surveys:
                run_measurement(agents, day=day, cfg=cfg, sim_time=sim_time, tracker=tracker)
                tracker.report(f"Day {day} VSM13 Measurement")
                tracker.append_log(f"Day {day:02d} VSM13 Measurement", TOKEN_LOG)

                run_wvs_measurement(agents, day=day, cfg=cfg, sim_time=sim_time, tracker=tracker)
                tracker.report(f"Day {day} WVS Measurement")
                tracker.append_log(f"Day {day:02d} WVS Measurement", TOKEN_LOG)

            # Written only after the day's rounds, deep reflection and surveys
            # have completed, so it marks the day as complete for resuming.
            day_snapshot_dir = cfg.results_dir / "memory_snapshots" / f"day{day:02d}"
            day_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for agent in agents:
                agent.save(str(day_snapshot_dir))

        print(f"\n{'='*60}")
        print("Simulation complete.")
        print(f"Logs saved to: {cfg.logs_dir}")
        print(f"Results saved to: {cfg.results_dir}")
        print(f"{'='*60}")

    finally:
        # A final snapshot is written only if the run exited early, since a
        # completed run's last day snapshot already holds the same state.
        last_day_snapshot_dir = cfg.results_dir / "memory_snapshots" / f"day{cfg.total_days:02d}"
        if last_day_snapshot_dir.exists():
            print(f"Run completed through Day {cfg.total_days}; final snapshot "
                  f"skipped (identical to {last_day_snapshot_dir})")
        else:
            final_snapshot_dir = cfg.results_dir / "memory_snapshots" / "final"
            final_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for agent in agents:
                agent.save(str(final_snapshot_dir))
            print(f"Final memory (and core belief, if two_tier) snapshot saved to: {final_snapshot_dir}")

        tracker.end_run(TOKEN_LOG)


# Entry point
def parse_args():
    parser = argparse.ArgumentParser(description="Cultural drift simulation (config-driven)")
    parser.add_argument("--memory_mode", choices=["single_tier", "two_tier"], default="two_tier",
                         help="single_tier = Park et al. baseline, two_tier = CoreBeliefs + AssociativeMemory")
    parser.add_argument("--use_reasoning", dest="use_reasoning", action="store_true", default=True,
                         help="Enable private pre-utterance reasoning")
    parser.add_argument("--no_reasoning", dest="use_reasoning", action="store_false",
                         help="Disable private reasoning")
    parser.add_argument("--anti_compromise", action="store_true", default=False,
                         help="Add the anti-compromise instruction to the conversation prompt (RQ3 C4)")
    parser.add_argument("--frozen_core", action="store_true", default=False,
                         help="two_tier only: CoreBeliefs are never revised (no deep reflection); "
                              "surveys and persona context read core beliefs plus Tier 2 thoughts")
    parser.add_argument("--model_key", choices=list(CLIENT_REGISTRY.keys()), default="openai",
                         help="Which client/model to run agents on (RQ2 cross-model comparison)")
    parser.add_argument("--total_days", type=int, default=DEFAULT_TOTAL_DAYS)
    parser.add_argument("--scenarios_file", default=DEFAULT_SCENARIOS_FILE)
    parser.add_argument("--start_day", type=int, default=None,
                         help="Force a start day instead of auto-detecting resume point")
    parser.add_argument("--skip_surveys", action="store_true", default=False,
                         help="Smoke test only: skip VSM13 and WVS")
    return parser.parse_args()


def main():
    print("Starting simulation...")
    args = parse_args()
    cfg = build_run_config(args)

    if cfg.frozen_core and cfg.memory_mode != "two_tier":
        sys.exit("--frozen_core requires --memory_mode two_tier")

    def embedding_fn(text: str):
        return get_embedding(text, client)

    agent_dirs = {
        "australia": _DATA_DIR / "personas" / "agents_australia",
        "malaysia": _DATA_DIR / "personas" / "agents_malaysia",
        "japan": _DATA_DIR / "personas" / "agents_japan",
    }

    agents = []
    for culture, directory in agent_dirs.items():
        yaml_files = sorted(directory.glob("*.yaml"))
        for yaml_path in yaml_files:
            agent = Agent.from_yaml(
                yaml_path, embedding_fn,
                client=cfg.client, model=cfg.model,
                memory_mode=cfg.memory_mode,
                use_reasoning=cfg.use_reasoning,
                anti_compromise=cfg.anti_compromise,
                logs_dir=cfg.logs_dir,
                frozen_core=cfg.frozen_core,
            )
            agents.append(agent)
            print(f"Loaded: {agent.name} ({agent.country})")
            print(f"{agent.name}: {len(agent.memory.seq_thought)} seed thoughts loaded")

    print(f"\nTotal agents loaded: {len(agents)}")
    print(f"Config: memory_mode={cfg.memory_mode}, frozen_core={cfg.frozen_core}, "
          f"use_reasoning={cfg.use_reasoning}, anti_compromise={cfg.anti_compromise}, "
          f"model_key={args.model_key}")

    lock = cfg.logs_dir / "RUN.lock"
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock.open("x").close()
    except FileExistsError:
        sys.exit(f"{lock} exists: a simulation is already running here, or one was killed. "
                 f"Check, then delete it.")

    try:
        if (args.start_day is None and find_resume_day(cfg) == 0
                and (cfg.logs_dir / "interactions.jsonl").exists()):
            sys.exit("logs/ has data from an earlier run. Move or empty logs/ and results/ first.")

        if args.start_day is not None:
            start_day = args.start_day
            resume_snapshot = cfg.results_dir / "memory_snapshots" / f"day{start_day - 1:02d}"
            resume_core_snapshot = cfg.results_dir / "core_snapshots" / f"day{start_day - 1:02d}"
            skip_t0 = False
            check_run_config(cfg, args.model_key,
                             resuming=(start_day > 1 and resume_snapshot.exists()))
            if resume_snapshot.exists():
                print(f"Forced start at Day {start_day} — loading snapshot from Day {start_day - 1}")
                for agent in agents:
                    agent.load_memory(str(resume_snapshot), core_dir=str(resume_core_snapshot))
                prune_logs_after(cfg, start_day - 1)
                skip_t0 = (start_day == 1)  # day00 snapshot loaded, so T0 is not re-measured
            run_simulation(agents, cfg, start_day=start_day, skip_t0_measurement=skip_t0)
        else:
            resume_day = find_resume_day(cfg)
            check_run_config(cfg, args.model_key, resuming=(resume_day > 0))
            if resume_day > 0:
                resume_snapshot = cfg.results_dir / "memory_snapshots" / f"day{resume_day:02d}"
                resume_core_snapshot = cfg.results_dir / "core_snapshots" / f"day{resume_day:02d}"
                print(f"Found completed Day {resume_day} — resuming from Day {resume_day + 1}")
                for agent in agents:
                    agent.load_memory(str(resume_snapshot), core_dir=str(resume_core_snapshot))
                prune_logs_after(cfg, resume_day)
                run_simulation(agents, cfg, start_day=resume_day + 1)
            else:
                run_simulation(agents, cfg)
    finally:
        lock.unlink(missing_ok=True)

if __name__ == "__main__":
    main()