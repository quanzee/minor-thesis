"""
Large-population side study for RQ1, run in the standard configuration
only (single_tier memory, no private reasoning, no anti-compromise
instruction, OpenAI).

Design:
    6 days x 4 timepoints x 7 conversations = 168 conversations.
    14 agents are sampled per timepoint from a pool of 99
        (33 Australia, 33 Japan, 33 Malaysia).
    24 scenarios, 4 per Hofstede dimension, with one scenario shared by
        all conversations in a timepoint and none repeated. The scenario
        for each timepoint is: slot = (day - 1) * 4 + (timepoint - 1).
    VSM13 and WVS are administered at the end of each day, with a Day 0
        baseline on the full pool.

Sampling and measurement:
    1. Sampling is independent per timepoint, with replacement across
       timepoints and days, so an agent may appear zero, one or several
       times. Memory accumulates on each agent across the full study.
    2. End-of-day surveys are administered only to agents who had at least
       one conversation that day. The Day 0 baseline covers all 99 agents.
    3. Job type (ANZSCO major groups), education level (ABS HEAP broad
       levels) and age come from 33 fixed triples shared across all three
       cultures, so demographics are held constant across culture. Names
       are the only attribute that varies by culture. Memory seeds are
       taken from one existing persona YAML per culture.
"""

import json
import random
import argparse
import itertools
from pathlib import Path
import sys

# data/ and lib/ sit inside simulation_architecture/; wvs/, logs/ and
# results/ sit in the project root.
_THIS_DIR = Path(__file__).parent
_DATA_DIR = Path(__file__).parent / "data"
_LIB_DIR = Path(__file__).parent / "lib"
_WVS_DIR = Path(__file__).parent.parent / "wvs"
_BASE_DIR = Path(__file__).parent.parent

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_LIB_DIR))
sys.path.insert(0, str(_WVS_DIR))

import yaml

from agent import Agent
from simulation import (
    RunConfig, run_measurement, run_wvs_measurement,
    append_log, setup_logs,
)
from config import client as OPENAI_CLIENT, MODEL_NAME as OPENAI_MODEL
from retrieve import get_embedding
from token_tracker import TokenTracker

# Study design constants
TOTAL_DAYS = 6
TIMEPOINTS_PER_DAY = 4
CONVERSATIONS_PER_TIMEPOINT = 7
AGENTS_PER_TIMEPOINT = 14
POOL_SIZE = 99
N_SCENARIOS = 24
HOFSTEDE_DIMENSIONS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]

CULTURE_POOL_SPLIT = {
    "Australia": 33,
    "Japan": 33,
    "Malaysia": 33,
}

DEFAULT_SCENARIOS_FILE = "scenarios_side_study.yaml"

# Demographic attribute pools, shared across cultures. The 33 fixed
# (job_type, education, age) triples are sampled from these in
# generate_agent_pool().

ANZSCO_JOB_TYPES = [
    "Manager",
    "Professional",
    "Technician/Trades",
    "Community/Personal Service",
    "Clerical/Admin",
    "Sales",
    "Machinery Operator/Driver",
    "Labourer",
]

EDUCATION_LEVELS = ["no qualification", "Year 12/ secondary",
                    "vocational", "bachelor's degree", "postgraduate degree"]
AGES = list(range(22, 60))

# Names are the only attribute that varies by culture. Each culture's
# names are shuffled independently before being paired with the shared
# demographic triples.
CULTURE_NAMES = {
    "Australia": [
        # (name, gender) pairs, so gender always matches the name
        ("Amelia", "Female"), ("Charlotte", "Female"), ("Isla", "Female"),
        ("Mia", "Female"), ("Ava", "Female"), ("Grace", "Female"),
        ("Chloe", "Female"), ("Ruby", "Female"), ("Zoe", "Female"),
        ("Ella", "Female"), ("Sophie", "Female"), ("Lily", "Female"),
        ("Matilda", "Female"), ("Harper", "Female"), ("Willow", "Female"),
        ("Evie", "Female"), ("Lucy", "Female"),
        ("James", "Male"), ("Oliver", "Male"), ("Jack", "Male"),
        ("Noah", "Male"), ("William", "Male"), ("Lucas", "Male"),
        ("Henry", "Male"), ("Thomas", "Male"), ("Leo", "Male"),
        ("Ethan", "Male"), ("Mason", "Male"), ("Cooper", "Male"),
        ("Archer", "Male"), ("Hudson", "Male"), ("Levi", "Male"),
        ("Xavier", "Male"),
    ],
    "Japan": [
        ("Mei", "Female"), ("Rin", "Female"), ("Yui", "Female"),
        ("Aoi", "Female"), ("Sakura", "Female"), ("Hina", "Female"),
        ("Yuna", "Female"), ("Mio", "Female"), ("Riko", "Female"),
        ("Saki", "Female"), ("Akari", "Female"), ("Honoka", "Female"),
        ("Kokoro", "Female"), ("Nanami", "Female"), ("Ichika", "Female"),
        ("Yuzuki", "Female"), ("Emiri", "Female"),
        ("Haruto", "Male"), ("Sora", "Male"), ("Ren", "Male"),
        ("Kaito", "Male"), ("Yuto", "Male"), ("Sota", "Male"),
        ("Daiki", "Male"), ("Ryota", "Male"), ("Kenta", "Male"),
        ("Takumi", "Male"), ("Hayato", "Male"), ("Riku", "Male"),
        ("Yamato", "Male"), ("Shota", "Male"), ("Kazuki", "Male"),
        ("Tsubasa", "Male"),
    ],
    "Malaysia": [
        # Malay, Chinese and Indian names, reflecting Malaysia's population
        ("Siti", "Female"), ("Nur", "Female"), ("Aisyah", "Female"),
        ("Zainab", "Female"), ("Farah", "Female"), ("Nurul", "Female"),
        ("Aina", "Female"), ("Fatimah", "Female"),
        ("Mei Ling", "Female"), ("Xin Yi", "Female"), ("Hui Min", "Female"),
        ("Li Ying", "Female"), ("Wei Ling", "Female"),
        ("Kavya", "Female"), ("Priya", "Female"), ("Deepa", "Female"),
        ("Anjali", "Female"),
        ("Abdul", "Male"), ("Hafiz", "Male"), ("Aziz", "Male"),
        ("Farid", "Male"), ("Zulkifli", "Male"), ("Amir", "Male"),
        ("Rahman", "Male"), ("Hakim", "Male"),
        ("Wei Ming", "Male"), ("Jun Wei", "Male"), ("Kok Wei", "Male"),
        ("Zhi Hao", "Male"), ("Chee Keong", "Male"),
        ("Arjun", "Male"), ("Ravi", "Male"), ("Kumar", "Male"),
    ],
}


def _load_culture_memory_seeds(culture: str) -> tuple:
    """
    Reads memory_seeds from the first persona YAML for a culture. The same
    seeds are used for every pool agent of that culture, so agents differ
    only by demographic attributes.

    Returns (source_name, memory_seeds). Seed descriptions contain the
    source agent's name (e.g. "James holds that..."), which callers
    replace with the pool agent's own name.
    """
    dir_name = f"agents_{culture.lower()}"
    directory = _DATA_DIR / "personas" / dir_name
    yaml_files = sorted(directory.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(
            f"No existing agent YAMLs found in {directory} to source "
            f"memory_seeds from for {culture}. The side-study reuses the "
            f"main study's per-culture seeds rather than inventing new ones."
        )
    with open(yaml_files[0], encoding="utf-8") as f:
        data = yaml.safe_load(f)["agent"]
    return data["name"], data["memory_seeds"]


def generate_agent_pool(embedding_fn, client, model, memory_mode, logs_dir):
    """
    Builds the pool of 99 agents (CULTURE_POOL_SPLIT per culture).

    A single set of 33 (job_type, education, age) triples is sampled once
    and assigned to every culture, so each triple is held by exactly one
    agent per culture and cross-cultural differences cannot be attributed
    to demographics. Names are shuffled independently per culture before
    pairing with the triples.

    All agents use the standard configuration (no reasoning, no
    anti-compromise instruction).
    """
    all_triples = list(itertools.product(ANZSCO_JOB_TYPES, EDUCATION_LEVELS, AGES))
    counts = set(CULTURE_POOL_SPLIT.values())
    if len(counts) != 1:
        raise ValueError(
            "Fixed-triple design requires an equal agent count per "
            f"culture; got {CULTURE_POOL_SPLIT}."
        )
    n_triples = counts.pop()
    if len(all_triples) < n_triples:
        raise ValueError(
            f"Only {len(all_triples)} (job_type, education, age) combinations "
            f"available, need {n_triples}. Widen ANZSCO_JOB_TYPES / "
            f"EDUCATION_LEVELS / AGES."
        )
    fixed_triples = random.sample(all_triples, n_triples)

    pool = []
    for culture, count in CULTURE_POOL_SPLIT.items():
        source_name, raw_seeds = _load_culture_memory_seeds(culture)
        names = list(CULTURE_NAMES[culture])
        if len(names) != count:
            raise ValueError(
                f"CULTURE_NAMES[{culture!r}] has {len(names)} names, "
                f"need exactly {count} for a 1:1 pairing with the "
                f"{n_triples} fixed triples."
            )
        random.shuffle(names)

        for (name, gender), (job_type, education, age) in zip(names, fixed_triples):
            # Replace the source persona's name with this agent's name
            agent_seeds = [
                {**seed, "description": seed["description"].replace(source_name, name)}
                for seed in raw_seeds
            ]

            agent = Agent(
                name=name,
                gender=gender,
                age=age,
                country=culture,
                education=education,
                job_type=job_type,
                memory_seeds=agent_seeds,
                embedding_fn=embedding_fn,
                client=client,
                model=model,
                memory_mode=memory_mode,
                use_reasoning=False,
                anti_compromise=False,
                logs_dir=logs_dir,
            )
            pool.append(agent)

    return pool


def load_side_study_scenarios(scenarios_path: Path):
    with open(scenarios_path, encoding="utf-8") as f:
        scenarios = yaml.safe_load(f)["scenarios"]

    if len(scenarios) != N_SCENARIOS:
        print(f"  WARNING: expected {N_SCENARIOS} scenarios, found {len(scenarios)}.")

    dim_counts = {d: 0 for d in HOFSTEDE_DIMENSIONS}
    for s in scenarios:
        dim = s.get("dimension")
        if dim in dim_counts:
            dim_counts[dim] += 1
        else:
            print(f"  WARNING: scenario has unrecognised/missing dimension: {s}")
    off_target = {d: n for d, n in dim_counts.items() if n != N_SCENARIOS // len(HOFSTEDE_DIMENSIONS)}
    if off_target:
        print(f"  WARNING: scenario counts per dimension are uneven: {dim_counts}")

    return scenarios


def build_side_study_config(args) -> RunConfig:
    return RunConfig(
        memory_mode="single_tier",
        use_reasoning=False,
        anti_compromise=False,
        client=OPENAI_CLIENT,
        model=OPENAI_MODEL,
        total_days=TOTAL_DAYS,
        logs_dir=_BASE_DIR / "logs" / "side_study",
        results_dir=_BASE_DIR / "results" / "side_study",
        scenarios_path=_DATA_DIR / args.scenarios_file,
    )


def run_side_study(cfg: RunConfig, pool: list, scenarios: list):
    tracker = TokenTracker()
    setup_logs(cfg)

    # Day 0 baseline on the full pool
    print(f"\n{'='*60}")
    print("Day 0 Baseline — full pool of 99")
    print(f"{'='*60}")
    run_measurement(pool, day=0, cfg=cfg, sim_time="Day 00", tracker=tracker)
    run_wvs_measurement(pool, day=0, cfg=cfg, sim_time="Day 00", tracker=tracker)
    tracker.report("Day 0 Baseline (full pool)")
    tracker.append_log("Day 00 Baseline (full pool)", cfg.logs_dir / "token_usage.jsonl")

    # Day 0 memory snapshot for the full pool
    snapshot_dir = cfg.results_dir / "memory_snapshots" / "day00"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for agent in pool:
        agent.save(str(snapshot_dir))

    try:
        for day in range(1, TOTAL_DAYS + 1):
            print(f"\n{'#'*60}")
            print(f"# DAY {day}")
            print(f"{'#'*60}")

            day_participants = {}  # name -> agent, unique across timepoints

            for timepoint in range(1, TIMEPOINTS_PER_DAY + 1):
                slot_idx = (day - 1) * TIMEPOINTS_PER_DAY + (timepoint - 1)
                scenario = scenarios[slot_idx]
                sim_time = f"Day {day:02d}, Timepoint {timepoint}"

                print(f"\n  [Timepoint {timepoint}] dimension={scenario.get('dimension')}")

                sampled = random.sample(pool, AGENTS_PER_TIMEPOINT)
                random.shuffle(sampled)
                pairs = [
                    (sampled[i], sampled[i + 1])
                    for i in range(0, AGENTS_PER_TIMEPOINT, 2)
                ]
                assert len(pairs) == CONVERSATIONS_PER_TIMEPOINT

                for agent_a, agent_b in pairs:
                    for ag in (agent_a, agent_b):
                        ag.active_scenario = {
                            "text": scenario["text"],
                            "day": day,
                            # must match the round part of sim_time
                            "round": f"Timepoint {timepoint}",
                            "location": None,
                            "sim_time": sim_time,
                        }

                    print(f"    {agent_a.name} ({agent_a.country}) <-> "
                          f"{agent_b.name} ({agent_b.country})")

                    conversation, reasoning_log = agent_a.converse(
                        agent_b, sim_time, tracker=tracker
                    )

                    agent_a.recent_partners.append(agent_b.name)
                    agent_b.recent_partners.append(agent_a.name)
                    agent_a.recent_partners = agent_a.recent_partners[-3:]
                    agent_b.recent_partners = agent_b.recent_partners[-3:]

                    append_log(
                        cfg.logs_dir / "interactions.jsonl",
                        {
                            "day": day,
                            "timepoint": timepoint,
                            "scenario_index": slot_idx,
                            "dimension": scenario.get("dimension"),
                            "agent_a": agent_a.name,
                            "culture_a": agent_a.country,
                            "agent_b": agent_b.name,
                            "culture_b": agent_b.country,
                            "cross_cultural": agent_a.country != agent_b.country,
                            "reasoning": reasoning_log,
                            "dialogue": conversation,
                        }
                    )

                    for ag in (agent_a, agent_b):
                        reflected, focal_points, all_insights = ag.reflect(sim_time, tracker=tracker)
                        if reflected:
                            append_log(
                                cfg.logs_dir / "reflections.jsonl",
                                {
                                    "day": day,
                                    "timepoint": timepoint,
                                    "agent": ag.name,
                                    "culture": ag.country,
                                    "focal_points": focal_points,
                                    "insights": all_insights,
                                }
                            )
                        day_participants[ag.name] = ag

                # Clear the scenario so it does not carry into the next timepoint
                for ag in sampled:
                    ag.active_scenario = None

            # End-of-day measurement for that day's participants
            participants = list(day_participants.values())
            print(f"\n  Day {day} participants: {len(participants)} unique agents")

            run_measurement(participants, day=day, cfg=cfg, sim_time=f"Day {day:02d}", tracker=tracker)
            run_wvs_measurement(participants, day=day, cfg=cfg, sim_time=f"Day {day:02d}", tracker=tracker)

            tracker.report(f"Day {day} Measurement")
            tracker.append_log(f"Day {day:02d} Measurement", cfg.logs_dir / "token_usage.jsonl")

            # Only participants are snapshotted, since other agents' memory
            # is unchanged since their last snapshot.
            day_snapshot_dir = cfg.results_dir / "memory_snapshots" / f"day{day:02d}"
            day_snapshot_dir.mkdir(parents=True, exist_ok=True)
            for agent in participants:
                agent.save(str(day_snapshot_dir))
    finally:
        # Full-pool snapshot if the run exits before the final day
        last_day_dir = cfg.results_dir / "memory_snapshots" / f"day{TOTAL_DAYS:02d}"
        if last_day_dir.exists():
            print(f"Run completed through Day {TOTAL_DAYS}; final snapshot skipped "
                  f"(day-folders already record what changed).")
        else:
            final_dir = cfg.results_dir / "memory_snapshots" / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            for agent in pool:
                agent.save(str(final_dir))
            print(f"Final memory snapshot (full pool) saved to: {final_dir}")


    print("\nSide-study complete.")
    print(f"Interactions logged to: {cfg.logs_dir / 'interactions.jsonl'}")
    print(f"VSM13/WVS results in: {cfg.results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Diversification side-study (standard config, OpenAI only)")
    parser.add_argument("--scenarios_file", default=DEFAULT_SCENARIOS_FILE)
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for agent sampling/pairing reproducibility")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    cfg = build_side_study_config(args)

    def embedding_fn(text: str):
        return get_embedding(text, cfg.client)

    print("Building agent pool (99 agents, single_tier standard config)...")
    pool = generate_agent_pool(
        embedding_fn=embedding_fn,
        client=cfg.client,
        model=cfg.model,
        memory_mode=cfg.memory_mode,
        logs_dir=cfg.logs_dir,
    )
    print(f"Pool built: {len(pool)} agents "
          f"({sum(1 for a in pool if a.country == 'Australia')} AU / "
          f"{sum(1 for a in pool if a.country == 'Japan')} JP / "
          f"{sum(1 for a in pool if a.country == 'Malaysia')} MY)")

    scenarios = load_side_study_scenarios(cfg.scenarios_path)

    run_side_study(cfg, pool, scenarios)


if __name__ == "__main__":
    main()