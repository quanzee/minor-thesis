# Cultural Belief Drift in Multi-Agent LLM Dialogue Simulations

Code and run outputs for the RMIT minor thesis *Cultural Belief Drift in Multi-Agent LLM Dialogue Simulations*.

The study simulates a small town of LLM agents seeded with Australian, Japanese and Malaysian cultural personas, derived from Hofstede's cultural dimensions, and measures whether their stated cultural positions drift through repeated multi-agent dialogue. It addresses three research questions:

- **RQ1:** To what extent do culturally seeded LLM agents' beliefs drift from their initial cultural positions during multi-agent dialogue, and does this drift show a consistent direction (convergence, divergence or no clear pattern)?
- **RQ2:** Do the extent and direction of belief drift vary across different underlying LLMs under an identical simulation setup?
- **RQ3:** Do architectural or prompting interventions change the extent or direction of drift?

The agent architecture is adapted from Park et al. (2023), *Generative Agents: Interactive Simulacra of Human Behavior* ([joonspk-research/generative_agents](https://github.com/joonspk-research/generative_agents)), without the sandbox environment. Modules adapted from that codebase name their source files in their docstrings.

## Repository structure

```
run_all.py                      Runs one configuration end to end (simulation, WVS drift, LLM judge)
simulation_architecture/
  simulation.py                 Main simulation loop (9 agents, 6 days, 3 rounds per day)
  run_side_study.py             Large-population study (99 agents)
  run_vsm13_default.py          VSM13 without seed beliefs (Day 0 only, no simulation)
  agent.py                      Agent: planning, conversation, persona context
  simplified_associative_memory.py  Tier 2 memory stream (Park et al.)
  core_beliefs.py               Tier 1 core beliefs (two-tier memory)
  retrieve.py                   Recency/importance/relevance retrieval (Park et al.)
  reflect.py                    Periodic and post-conversation reflection (Park et al.)
  deep_reflect.py               End-of-day core belief revision (not used by any reported condition)
  town.py                       Single-location co-presence tracking
  vsm13.py                      VSM13 survey administration and dimension scoring
  kendall_tau.py                Kendall's tau-b against Hofstede's human scores
  lib/config.py                 Model clients and API retry handling
  lib/token_tracker.py          Token usage logging
  data/personas/                Persona YAMLs (3 agents per culture)
  data/scenarios_main.yaml      Main-study scenarios (one per day and dimension)
  data/scenarios_side_study.yaml  Large-population study scenarios
wvs/
  wvs.py                        WVS survey administration (WorldValuesBench items)
  compute_wvs_drift.py          Day 0 vs Day N and day-to-day WVS drift
  wvs_filtered_questions.json   The 75 WVS items used
llm_as_judge/
  rescore_openings_holistic.py  LLM-as-judge scoring of conversation openings
  opening_trajectories.py       Within-day opening-stance trajectories and summary tables
  config.py                     Judge model and day anchors
past_runs/                      Logs and results of every reported run
```

## Setup

Requires Python 3.10 or later.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with the endpoints and keys for the model deployments:

```
AZURE_ENDPOINT=
AZURE_API_KEY=
AZURE_API_VERSION=
DEEPSEEK_ENDPOINT=
DEEPSEEK_API_KEY=
GROK_ENDPOINT=
GROK_API_KEY=
```

Models used:

| Role | Model |
|---|---|
| Agents (RQ1, RQ3, RQ2 reference) | GPT-4o mini (Azure OpenAI) |
| Agents (RQ2) | DeepSeek-V4-Flash, Grok 4.1 Fast (non-reasoning) |
| LLM judge | GPT-5.4 mini (Azure OpenAI) |
| Embeddings | text-embedding-3-small (Azure OpenAI), for all agent models |

## Running a configuration

All commands are run from the project root.

```bash
python run_all.py
```

This runs C1 (baseline): the simulation, its WVS drift computation and LLM-as-judge scoring of conversation openings. To run another condition, change the simulation flags in `run_all.py`:

| Condition | Memory | Private reasoning | Anti-compromise | Flags |
|---|---|---|---|---|
| C1 (baseline) | Single-tier | Off | Off | `--memory_mode single_tier --no_reasoning --model_key openai` |
| C1-R (replicate) | Single-tier | Off | Off | Same as C1 |
| C2 | Two-tier, fixed core | Off | Off | `--memory_mode two_tier --frozen_core --no_reasoning --model_key openai` |
| C3 | Single-tier | On | Off | `--memory_mode single_tier --use_reasoning --model_key openai` |
| C4 | Single-tier | Off | On | `--memory_mode single_tier --no_reasoning --anti_compromise --model_key openai` |
| RQ2 DeepSeek | Single-tier | Off | Off | `--memory_mode single_tier --no_reasoning --model_key deepseek` |
| RQ2 Grok | Single-tier | Off | Off | `--memory_mode single_tier --no_reasoning --model_key grok` |
| C1-six (control) | Single-tier, six beliefs | Off | Off | Same as C1, after the code changes below |

C1-six is a control for C2. C2 protects the seeds and also retrieves six belief lines where the baseline retrieves three (three core beliefs plus three memory-stream beliefs). C1-six is the baseline with six beliefs retrieved and no protected seeds. There is no flag for it. Before running it with the C1 flags, change the three places where single-tier agents retrieve beliefs from 3 to 6:

1. **Conversation:** `simulation_architecture/retrieve.py`, `retrieve_current_beliefs`: `n_count=3` → `n_count=6`, and `for node in nodes[:3]` → `for node in nodes[:6]`. This sets the beliefs in each agent's persona context, through `get_persona_context` in `agent.py`.
2. **VSM13:** `simulation_architecture/simulation.py`, `run_measurement`: `retrieve_all_dimension_beliefs(agent, curr_time=sim_time)` → `retrieve_all_dimension_beliefs(agent, curr_time=sim_time, top_n=6)`
3. **WVS:** `simulation_architecture/simulation.py`, `run_wvs_measurement`: `retrieve_all_beliefs(agent, curr_time=sim_time)` → `retrieve_all_beliefs(agent, curr_time=sim_time, top_n=6)`

Leave everything else unchanged. In particular, `n_turns` in `agent.py` is the number of utterances per conversation, not a belief count, and the `n=3` / `nodes[:3]` retrievals in `plan` and utterance generation retrieve memories, not beliefs. Revert the three changes before running any other condition.

The simulation writes to `logs/` and `results/` in the project root and will not start if `logs/` already holds an earlier run, so move each run's output (e.g. into `past_runs/`) before starting the next one. An interrupted run resumes from its last completed day when restarted with the same flags.

### Opening-stance trajectories

After scoring openings for one or more runs:

```bash
python -m llm_as_judge.opening_trajectories --run openai=past_runs/C1
python -m llm_as_judge.opening_trajectories \
    --run openai=past_runs/C1 --run deepseek=past_runs/RQ2_deepseek --run grok=past_runs/RQ2_grok
```

### Large-population study

```bash
python simulation_architecture/run_side_study.py --seed 1
python wvs/compute_wvs_drift.py --results_dir results/side_study
```

The large-population study uses VSM13 and WVS only.

### VSM13 without seeds

```bash
python simulation_architecture/run_vsm13_default.py
```

The nine agents answer VSM13 with only their name, country and demographic details in the prompt (no seed beliefs), three repetitions each, on each of the three agent models. There is no simulation. This is not a condition: it shows how far the cultural cues alone (name and country) separate the cultures.

## Measures

- **VSM13** (primary): Hofstede's Values Survey Module 2013, administered to every agent at Day 0 and at the end of each day (three runs per agent, averaged). Drift is measured relative to Day 0, and rank order against Hofstede's human scores with Kendall's tau-b.
- **WVS** (supplementary): 75 items from WorldValuesBench (Zhao et al., 2024). Drift is the share of items whose majority-vote answer changed.
- **LLM judge** (supplementary): a separate model scores each conversation's opening utterance on the day's Hofstede dimension (1–100), giving within-day opening-stance trajectories.

## Past runs

| Folder | Condition |
|---|---|
| `past_runs/C1` | C1 baseline (GPT-4o mini) |
| `past_runs/C1-R` | C1 replicate |
| `past_runs/C2` | Two-tier memory, fixed core |
| `past_runs/C3` | Private reasoning |
| `past_runs/C4` | Anti-compromise instruction |
| `past_runs/RQ2_deepseek` | C1 configuration on DeepSeek-V4-Flash |
| `past_runs/RQ2_grok` | C1 configuration on Grok 4.1 Fast |
| `past_runs/C1-six` | C1 with six retrieved beliefs (control for C2) |
| `past_runs/large-population` | Large-population study |
| `past_runs/no_seed_day0` | VSM13 without seeds, Day 0 only, all three models |

Each simulation run folder contains `logs/` and `results/`. `results/run_config.json` records the flags the run was started with; for C1-six it records the C1 flags, since the six-belief change is in the code.