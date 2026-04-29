"""
VSM13 Day 0 Measurement
-----------------------
Administers all 24 VSM13 questions individually to each agent,
then computes the 6 Hofstede dimension indices from pooled responses.
"""

import yaml
import json
import time
from pathlib import Path
from config import MODEL_NAME, client

# ── Configuration ─────────────────────────────────────────────────────────────

AGENTS_DIR = Path("agents_japan")        
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = MODEL_NAME
TEMPERATURE = 0                     

client = client

# ── VSM13 Questions (24 cultural items only) ──────────────────────────────────
# Q1–Q10: ideal job importance (1=utmost, 5=very little/none)
# Q11–Q20: personal views (1=strongly agree, 5=strongly disagree)
# Q21–Q24: current life statements (scale varies per question)

VSM13_QUESTIONS = {
    1:  ("job",      "Have sufficient time for your personal or home life"),
    2:  ("job",      "Have a boss (direct superior) you can respect"),
    3:  ("job",      "Get recognition for good performance"),
    4:  ("job",      "Have security of employment"),
    5:  ("job",      "Have pleasant people to work with"),
    6:  ("job",      "Do work that is interesting"),
    7:  ("job",      "Be consulted by your boss in decisions involving your work"),
    8:  ("job",      "Live in a desirable area"),
    9:  ("job",      "Have a job respected by your family and friends"),
    10: ("job",      "Have chances for promotion"),
    11: ("private",  "Keeping time free for fun"),
    12: ("private",  "Moderation: having few desires"),
    13: ("private",  "Doing a service to a friend"),
    14: ("private",  "Thrift (not spending more than needed)"),
    15: ("solo",     "How often do you feel nervous or tense?\n1 = always\n2 = usually\n3 = sometimes\n4 = seldom\n5 = never"),
    16: ("solo",     "Are you a happy person?\n1 = always\n2 = usually\n3 = sometimes\n4 = seldom\n5 = never"),
    17: ("solo",     "Do other people or circumstances ever prevent you from doing what you really want to?\n1 = yes, always\n2 = yes, usually\n3 = sometimes\n4 = no, seldom\n5 = no, never"),
    18: ("solo",     "All in all, how would you describe your state of health these days?\n1 = very good\n2 = good\n3 = fair\n4 = poor\n5 = very poor"),
    19: ("solo",     "How proud are you to be a citizen of your country?\n1 = very proud\n2 = fairly proud\n3 = somewhat proud\n4 = not very proud\n5 = not proud at all"),
    20: ("solo",     "How often, in your experience, are subordinates afraid to contradict their boss (or students their teacher)?\n1 = never\n2 = seldom\n3 = sometimes\n4 = usually\n5 = always"),
    21: ("agree",    "One can be a good manager without having a precise answer to every question that a subordinate may raise about his or her work"),
    22: ("agree",    "Persistent efforts are the surest way to results"),
    23: ("agree",    "An organisation structure in which certain subordinates have two bosses should be avoided"),
    24: ("agree",    "A company's or organisation's rules should not be broken — not even when the employee thinks breaking the rule would be in the organisation's best interest"),
}

SECTION_INSTRUCTIONS = {
    "job": (
        "Please think of an ideal job, disregarding your present job if you have one. "
        "In choosing an ideal job, how important would it be to you to have the following?\n\n"
        "1 = of utmost importance\n"
        "2 = very important\n"
        "3 = of moderate importance\n"
        "4 = of little importance\n"
        "5 = of very little or no importance"
    ),
    "private": (
        "In your private life, how important is each of the following to you?\n\n"
        "1 = of utmost importance\n"
        "2 = very important\n"
        "3 = of moderate importance\n"
        "4 = of little importance\n"
        "5 = of very little or no importance"
    ),
    "solo": "",   # scale is embedded in each question text directly
    "agree": (
        "To what extent do you agree or disagree with each of the following statements?\n\n"
        "1 = strongly agree\n"
        "2 = agree\n"
        "3 = undecided\n"
        "4 = disagree\n"
        "5 = strongly disagree"
    ),
}

# ── Dimension index formulas (Hofstede & Minkov VSM13) ────────────────────────
# Constants C set to 0 — measuring relative drift not absolute alignment

def compute_dimensions(means: dict) -> dict:
    m = means
    PDI = 35 * (m[7]  - m[2])  + 25 * (m[20] - m[23])
    IDV = 35 * (m[4]  - m[1])  + 35 * (m[9]  - m[6])
    MAS = 35 * (m[5]  - m[3])  + 25 * (m[8]  - m[10])
    UAI = 40 * (m[18] - m[15]) + 25 * (m[21] - m[24])
    LTO = 40 * (m[13] - m[14]) + 25 * (m[19] - m[22])
    IVR = 35 * (m[12] - m[11]) + 40 * (m[17] - m[16])
    return {"PDI": PDI, "IDV": IDV, "MAS": MAS, "UAI": UAI, "LTO": LTO, "IVR": IVR}

# ── Agent loading ─────────────────────────────────────────────────────────────

def load_agent(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data["agent"]

def build_system_prompt(agent: dict) -> str:
    seeds = "\n".join(f"- {s}" for s in agent["memory_seeds"])
    return (
        f"You are {agent['name']}, a {agent['age']}-year-old {agent['gender']} "
        f"from {agent['country']}. "
        f"Your education level is {agent['education level']} and your occupation is {agent['job type']}.\n\n"
        f"The following statements reflect your genuine beliefs and values:\n{seeds}\n\n"
        f"You are now completing a survey. Answer each question as yourself — based on your own "
        f"values, beliefs, and life experience. "
        f"For each question, respond in this exact format: [number (1, 2, 3, 4, or 5)] - [explanation in 1-2 sentences]. "
        f"The number must come first. No other text."
    )

# ── Question administration ───────────────────────────────────────────────────

def ask_question(
    system_prompt: str,
    q_num: int,
    q_type: str,
    q_text: str,
) -> tuple[int, list]:
    """
    Sends one VSM13 question as the next user turn.
    Returns (response_int, updated_history).
    """
    instructions = SECTION_INSTRUCTIONS[q_type]
    if instructions:
        user_message = f"{instructions}\n\nQuestion {q_num}: {q_text}\n\nYour answer:"
    else:
        user_message = f"Question {q_num}: {q_text}\n\nYour answer:"

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_message}],
    )

    reply = response.choices[0].message.content.strip()

    # Parse number from start of reply
    part = reply.split("-")[0].strip()
    score = None
    for char in part:
        if char in "12345":
            score = int(char)
            break

    # Parse explanation as everything after the first dash
    if "-" in reply:
        explanation = reply.split("-", 1)[1].strip()
    else:
        explanation = reply  # fallback if format not followed

    if score is None:
        print(f"  WARNING: Could not parse score from '{reply}' for Q{q_num}. Defaulting to 3.")
        score = 3

    return score, explanation

# ── Per-agent run ─────────────────────────────────────────────────────────────

def run_vsm13_for_agent(agent: dict) -> dict:
    print(f"\n  Administering VSM13 to {agent['name']}...")
    system_prompt = build_system_prompt(agent)
    responses = {}
    explanations = {}

    for q_num, (q_type, q_text) in VSM13_QUESTIONS.items():
        response, explanation = ask_question(
            system_prompt, q_num, q_type, q_text
        )
        responses[q_num] = response
        explanations[q_num] = explanation
        print(f"    Q{q_num:02d}: {response} - {explanation}")
        time.sleep(0.3)  # small buffer for rate limits

    return responses, explanations

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    yaml_files = sorted(AGENTS_DIR.glob("*.yaml"))
    if not yaml_files:
        print(f"No YAML files found in '{AGENTS_DIR}/'")
        return

    print(f"Found {len(yaml_files)} agent(s): {[f.name for f in yaml_files]}")

    all_responses = {}  # agent_name -> {q_num: int}
    all_explanations = {}

    for yaml_path in yaml_files:
        agent = load_agent(yaml_path)
        print(f"\nAgent: {agent['name']} | {agent['age']}yo {agent['gender']} | "
              f"{agent['job type']} | {agent['education level']}")
        responses, explanations = run_vsm13_for_agent(agent)
        all_responses[agent["name"]] = responses
        all_explanations[agent["name"]] = explanations

# ── Pool responses and compute means ─────────────────────────────────────
    n_agents = len(all_responses)
    pooled_means = {}
    for q_num in range(1, 25):
        total = sum(all_responses[name][q_num] for name in all_responses)
        pooled_means[q_num] = total / n_agents

    print("\n\nPooled question means across all agents:")
    for q_num, mean in pooled_means.items():
        print(f"  Q{q_num:02d}: {mean:.3f}")

    # ── Compute individual dimension scores ───────────────────────────────────
    individual_dimensions = {}
    for name, responses in all_responses.items():
        individual_dimensions[name] = compute_dimensions(responses)

    print("\nIndividual VSM13 Dimension Scores — Japan")
    print("=" * 45)
    for name, dims in individual_dimensions.items():
        print(f"  {name}:")
        for dim, score in dims.items():
            print(f"    {dim:>3}: {score:+.2f}")
    print("=" * 45)

    # ── Compute pooled dimension scores ───────────────────────────────────────
    dimensions = compute_dimensions(pooled_means)

    print("\nDay 0 VSM13 Dimension Scores (Pooled) — Japan")
    print("=" * 45)
    for dim, score in dimensions.items():
        print(f"  {dim:>3}: {score:+.2f}")
    print("=" * 45)

    # ── Save to JSON ──────────────────────────────────────────────────────────
    results = {
        "day": 0,
        "culture": "Japan",
        "n_agents": n_agents,
        "individual_responses": {
            name: {str(k): v for k, v in resp.items()}
            for name, resp in all_responses.items()
        },
        "individual_explanations": {
            name: {str(k): v for k, v in expl.items()}
            for name, expl in all_explanations.items()
        },
        "individual_dimension_scores": individual_dimensions,
        "pooled_means": {str(k): v for k, v in pooled_means.items()},
        "dimension_scores": dimensions,
    }

    output_path = RESULTS_DIR / "japan_day0_vsm13_run5.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()