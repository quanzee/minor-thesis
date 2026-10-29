"""
VSM13 default-position baseline (supervisor comment on 4.1.6, 4.2.5, 4.4).

Administers the 24 VSM13 items to the 9 agents with the demographic summary
only: same system prompt and question wording as vsm13.py, but no belief
context (no seeds, no memories). No simulation is run.

Output mirrors simulation.run_measurement so the existing Day 0 analysis
code can read it.

Usage:
    python run_vsm13_default.py --model_key openai
    python run_vsm13_default.py --model_key deepseek
    python run_vsm13_default.py --model_key grok
    python run_vsm13_default.py --model_key all
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR / "lib"))

from config import MODEL_NAME, MODEL_NAME_3, MODEL_NAME_4, \
    client, deepseek_client, grok_client
from vsm13 import VSM13_QUESTIONS, QUESTION_DIMENSION, build_system_prompt, \
    ask_question, compute_dimensions
from kendall_tau import HUMAN_SCORES, compute_kendall_tau

CLIENT_REGISTRY = {
    "openai":   (client, MODEL_NAME),
    "deepseek": (deepseek_client, MODEL_NAME_3),
    "grok":     (grok_client, MODEL_NAME_4),
}
CULTURES = ["Australia", "Malaysia", "Japan"]
PERSONA_DIRS = ["agents_australia", "agents_malaysia", "agents_japan"]
N_RUNS = 3


def load_agents():
    """Demographics only; memory_seeds are deliberately not read."""
    agents = []
    for d in PERSONA_DIRS:
        for path in sorted((_THIS_DIR / "data" / "personas" / d).glob("*.yaml")):
            with open(path, encoding="utf-8") as f:
                a = yaml.safe_load(f)["agent"]
            agents.append(SimpleNamespace(
                name=a["name"], gender=a["gender"], age=a["age"],
                country=a["country"], education=a["education level"],
                job_type=a["job type"],
            ))
    return agents


def run_one_agent(agent, run_client, run_model):
    system_prompt = build_system_prompt(agent)
    responses, explanations = {}, {}
    for q_num, (q_type, q_text) in VSM13_QUESTIONS.items():
        score, expl = ask_question(system_prompt, q_num, q_type, q_text,
                                   belief_context="",  # no beliefs
                                   client=run_client, model=run_model)
        responses[q_num], explanations[q_num] = score, expl
        print(f"    Q{q_num:02d} [{QUESTION_DIMENSION[q_num]}]: {score}")
        time.sleep(0.3)
    return responses, explanations


def run_model(model_key):
    run_client, run_model_name = CLIENT_REGISTRY[model_key]
    agents = load_agents()
    print(f"\n=== VSM13 default baseline: {run_model_name} ({len(agents)} agents x {N_RUNS} runs) ===")

    resp_avg = {c: {} for c in CULTURES}
    all_runs = {c: {} for c in CULTURES}
    expl_last = {c: {} for c in CULTURES}
    indiv_run_dims = {c: {} for c in CULTURES}

    for agent in agents:
        print(f"  {agent.name} ({agent.country})")
        runs = []
        for r in range(N_RUNS):
            print(f"   Run {r + 1}/{N_RUNS}")
            responses, explanations = run_one_agent(agent, run_client, run_model_name)
            runs.append(responses)
        c = agent.country
        resp_avg[c][agent.name] = {q: sum(r[q] for r in runs) / N_RUNS for q in range(1, 25)}
        all_runs[c][agent.name] = {f"run_{i+1}": {str(k): v for k, v in r.items()}
                                   for i, r in enumerate(runs)}
        expl_last[c][agent.name] = {str(k): v for k, v in explanations.items()}
        indiv_run_dims[c][agent.name] = {f"run_{i+1}": compute_dimensions(r)
                                         for i, r in enumerate(runs)}

    pooled, indiv_dims, pooled_run_dims = {}, {c: {} for c in CULTURES}, {c: {} for c in CULTURES}
    for c in CULTURES:
        names = list(resp_avg[c])
        n = len(names)
        for name in names:
            indiv_dims[c][name] = compute_dimensions(resp_avg[c][name])
        pooled[c] = compute_dimensions(
            {q: sum(resp_avg[c][nm][q] for nm in names) / n for q in range(1, 25)})
        for r in range(N_RUNS):
            key = f"run_{r + 1}"
            pooled_run_dims[c][key] = compute_dimensions(
                {q: sum(all_runs[c][nm][key][str(q)] for nm in names) / n for q in range(1, 25)})

    output = {
        "condition": "default_no_beliefs",
        "model": run_model_name,
        "day": 0,
        "n_runs": N_RUNS,
        "individual_responses_averaged": {
            c: {nm: {str(k): v for k, v in r.items()} for nm, r in d.items()}
            for c, d in resp_avg.items()},
        "individual_all_runs": all_runs,
        "individual_explanations_last_run": expl_last,
        "individual_dimension_scores": indiv_dims,
        "individual_run_dimensions": indiv_run_dims,
        "pooled_dimension_scores": pooled,
        "pooled_run_dimensions": pooled_run_dims,
        "kendall_tau": compute_kendall_tau(HUMAN_SCORES, pooled),
        "kendall_tau_per_run": {
            f"run_{r+1}": compute_kendall_tau(
                HUMAN_SCORES, {c: pooled_run_dims[c][f"run_{r+1}"] for c in CULTURES})
            for r in range(N_RUNS)},
    }

    out_dir = _THIS_DIR / "results" / "vsm13_default" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "vsm13_default_day0.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Pooled default scores ({run_model_name}):")
    for c in CULTURES:
        print(f"    {c:<10} " + "  ".join(f"{d}={v:7.1f}" for d, v in pooled[c].items()))
    print(f"  Saved to {path}")


CSV_LABELS = {"openai": "v1", "deepseek": "deepseek", "grok": "grok"}  # as in RQ2_day0.csv
DIMS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]


def _fmt(x):
    return "" if x is None or x != x else str(round(x, 4))  # blank for NaN tau


def write_csv():
    """
    Writes results/vsm13_default/default_day0.csv in the RQ2_day0.csv layout,
    with one column per model whose JSON exists.
    """
    base = _THIS_DIR / "results" / "vsm13_default"
    data = {}
    for key, label in CSV_LABELS.items():
        path = base / key / "vsm13_default_day0.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data[label] = json.load(f)
    if not data:
        print("No default-baseline JSONs found; CSV not written.")
        return
    labels = list(data)

    lines = ["# SECTION 1: DEFAULT (NO BELIEFS) DAY 0 POOLED VSM13 SCORES (18 Culture-by-Dimension)",
             "Culture,Dimension," + ",".join(labels)]
    for c in CULTURES:
        for d in DIMS:
            lines.append(f"{c},{d}," + ",".join(
                _fmt(data[l]["pooled_dimension_scores"][c][d]) for l in labels))

    lines += ["", "", "# SECTION 2: OVERALL KENDALL'S TAU-B (By Dimension & Average)",
              "Metric," + ",".join(labels)]
    for d in DIMS:
        lines.append(f"{d}," + ",".join(_fmt(data[l]["kendall_tau"][d]["tau"]) for l in labels))
    lines.append("Average_Tau," + ",".join(_fmt(data[l]["kendall_tau"]["average_tau"]) for l in labels))

    runs = [f"run_{r + 1}" for r in range(N_RUNS)]
    lines += ["", "", "# SECTION 3: KENDALL'S TAU-B PER SUB-RUN FOR EACH FILE",
              "File,Dimension," + ",".join(runs)]
    for l in labels:
        per_run = data[l]["kendall_tau_per_run"]
        for d in DIMS:
            lines.append(f"{l},{d}," + ",".join(_fmt(per_run[r][d]["tau"]) for r in runs))
        lines.append(f"{l},Average_Tau," + ",".join(_fmt(per_run[r]["average_tau"]) for r in runs))

    path = base / "default_day0.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nCSV saved to {path} (models: {', '.join(labels)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", choices=list(CLIENT_REGISTRY) + ["all"], default="all")
    p.add_argument("--csv_only", action="store_true",
                   help="Rebuild the CSV from existing JSONs without making API calls")
    args = p.parse_args()
    if not args.csv_only:
        for key in (CLIENT_REGISTRY if args.model_key == "all" else [args.model_key]):
            run_model(key)
    write_csv()