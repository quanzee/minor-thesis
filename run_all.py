"""
Runs one simulation configuration end to end: the simulation, its WVS drift
computation and LLM-as-judge scoring of conversation openings. Run from the
project root:

    python run_all.py

The simulation writes to logs/ and results/ in the project root and will not
start if logs/ already holds an earlier run, so move each run's output (e.g.
into past_runs/) before running the next configuration.

The configuration below is C1 (baseline). To run another configuration,
replace the simulation flags with the flags for that condition:

    C1 (baseline) and C1-R (replicate):
        --memory_mode single_tier --no_reasoning --model_key openai
    C2 (two-tier, fixed core):
        --memory_mode two_tier --frozen_core --no_reasoning --model_key openai
    C3 (private reasoning):
        --memory_mode single_tier --use_reasoning --model_key openai
    C4 (anti-compromise instruction):
        --memory_mode single_tier --no_reasoning --anti_compromise --model_key openai
    RQ2 cross-model runs (C1 configuration):
        --memory_mode single_tier --no_reasoning --model_key deepseek
        --memory_mode single_tier --no_reasoning --model_key grok

The large-population side study is run separately:

    python simulation_architecture/run_side_study.py --seed 1
    python wvs/compute_wvs_drift.py --results_dir results/side_study
"""

import subprocess
import sys

def run_step(description: str, cmd: list[str]) -> None:
    print(f"\n{'='*60}\n{description}\n{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"FAILED: {description} (exit code {result.returncode})")
        sys.exit(1)

if __name__ == "__main__":
    run_step("Simulation (C1 baseline)", [sys.executable, "-u", "simulation_architecture/simulation.py",
                                          "--memory_mode", "single_tier", "--no_reasoning", "--model_key", "openai"])
    run_step("WVS drift computation", [sys.executable, "-u", "wvs/compute_wvs_drift.py"])
    run_step("LLM-as-judge", [sys.executable, "-u", "-m", "llm_as_judge.rescore_openings_holistic"])