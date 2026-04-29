"""
VSM13 Cultural Alignment Test for LLM Personas
Based on Masoud et al. (2025) "Cultural Alignment in Large Language Models:
An Explanatory Analysis Based on Hofstede's Cultural Dimensions"

Runs the 24 VSM13 cultural questions against a persona prompt using the
Anthropic API, averages responses across N seeds, then computes all six
Hofstede dimension scores.
"""
import json
import statistics
import yaml
from config import MODEL_NAME, client

# ── Configuration ─────────────────────────────────────────────────────────────

N_SEEDS = 5          # Number of seeds to average over (paper uses 5)

# ── Persona ───────────────────────────────────────────────────────────────────

with open("persona_construction/australia_persona_v2.yaml", "r", encoding="utf-8") as f:
    persona_data = yaml.safe_load(f)

persona_seeds = "\n".join(persona_data["agent"]["memory_seeds"])
SYSTEM_PROMPT = f"""You are {persona_data['agent']['name']}, a {persona_data['agent']['gender']} from {persona_data['agent']['country']}. Answer all questions as {persona_data['agent']['name']} would, based on the following traits and values:

{persona_seeds}

You are answering a survey about your personal values and preferences.
Answer entirely as {persona_data['agent']['name']} — do not break character.
Respond with ONLY the numeric answer (1–5) for each question on its own line,
in the format: "Question N: X" where N is the question number and X is your answer.
Do not include any explanation, commentary, or preamble."""

# ── VSM13 Questions ───────────────────────────────────────────────────────────
# All 24 cultural questions from the VSM13 instrument.
# Scale: 1 = of utmost importance / strongly agree
#        5 = of very little or no importance / strongly disagree
# Questions 1–10: ideal job importance ratings
# Questions 11–24: personal values and opinions

VSM13_PREAMBLE_JOB = """Please think of an ideal job, disregarding your present job if you have one.
In choosing an ideal job, how important would it be to you to have:
(1 = of utmost importance, 2 = very important, 3 = of moderate importance,
 4 = of little importance, 5 = of very little or no importance)"""

VSM13_PREAMBLE_LIFE = """Please indicate your personal views on the following statements.
(1 = strongly agree, 2 = agree, 3 = undecided, 4 = disagree, 5 = strongly disagree)
For frequency questions: (1 = always, 2 = usually, 3 = sometimes, 4 = seldom, 5 = never)"""

VSM13_QUESTIONS = {
    # Job importance questions (1–10)
    1:  ("job", "have sufficient time for your personal or home life"),
    2:  ("job", "have a boss (direct superior) you can respect"),
    3:  ("job", "get recognition for good performance"),
    4:  ("job", "have job security"),
    5:  ("job", "work with people who cooperate well with one another"),
    6:  ("job", "be consulted by your direct superior in decisions involving your work"),
    7:  ("job", "have an element of variety and adventure in the job"),
    8:  ("job", "live in a desirable area"),
    9:  ("job", "have an interesting job (work that you actually enjoy doing)"),
    10: ("job", "work in a well-respected company or organisation"),
    # Life values and opinions (11–24)
    11: ("life", "Being thrifty (not spending more than necessary) is important to me"),
    12: ("life", "Personal stability is important to me"),
    13: ("life", "Persistence is important to me — keeping on when others have given up"),
    14: ("life", "Traditions are important to me — I try to follow them"),
    15: ("life", "How often do you feel nervous or tense at work?"),
    16: ("life", "How often is your work stressful?"),
    17: ("life", "I enjoy leisure time — time away from work for relaxation and fun"),
    18: ("life", "Other people are an important source of information about what to do"),
    19: ("life", "Rules and regulations are important — they tell me what to do"),
    20: ("life", "In my private life I aim to please myself, even if this displeases others"),
    21: ("life", "Decisions made by individuals are generally better than group decisions"),
    22: ("life", "You can be a good manager without having precise answers to every question"),
    23: ("life", "An organisation's rules should not be broken — not even when an employee believes it is in the organisation's best interest"),
    24: ("life", "Competition between employees usually does more harm than good"),
}

# ── Hofstede Dimension Formulae ───────────────────────────────────────────────
# From Equation 1 in the CAT paper.
# Constants C_X are set to 0 here — they are sample-dependent anchoring constants
# used when comparing to Hofstede's published dataset. Without the exact constants
# from Hofstede & Minkov (2010), setting them to 0 gives raw dimension scores
# that are valid for relative within-study comparison but not directly comparable
# to published country scores. See paper Section 2 for discussion.

def compute_dimensions(mu: dict) -> dict:
    PDI = 35 * (mu[7] - mu[2]) + 25 * (mu[20] - mu[23])
    IDV = 35 * (mu[4] - mu[1]) + 35 * (mu[9] - mu[6])
    MAS = 35 * (mu[5] - mu[3]) + 25 * (mu[8] - mu[10])
    UAI = 40 * (mu[18] - mu[15]) + 25 * (mu[21] - mu[24])
    LTO = 40 * (mu[13] - mu[14]) + 25 * (mu[19] - mu[22])
    IVR = 35 * (mu[12] - mu[11]) + 40 * (mu[17] - mu[16])
    return {"PDI": PDI, "IDV": IDV, "MAS": MAS, "UAI": UAI, "LTO": LTO, "IVR": IVR}

# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt() -> str:
    job_qs = "\n".join(
        f"Question {n}: {text}"
        for n, (section, text) in VSM13_QUESTIONS.items()
        if section == "job"
    )
    life_qs = "\n".join(
        f"Question {n}: {text}"
        for n, (section, text) in VSM13_QUESTIONS.items()
        if section == "life"
    )
    return f"""{VSM13_PREAMBLE_JOB}

{job_qs}

{VSM13_PREAMBLE_LIFE}

{life_qs}"""

# --- normalisation ----
def get_formula_bounds() -> dict:
    """
    Compute theoretical min/max by optimising each question's value
    independently to maximise or minimise the dimension score.
    Each question is set to 1 or 5 depending on its sign in the formula.
    """
    # (question_index, coefficient) — positive coeff: set to 5 for max, 1 for min
    # negative coeff: set to 1 for max, 5 for min
    dimension_terms = {
        "PDI": [(7, +35), (2, -35), (20, +25), (23, -25)],
        "IDV": [(4, +35), (1, -35), (9, +35), (6, -35)],
        "MAS": [(5, +35), (3, -35), (8, +25), (10, -25)],
        "UAI": [(18, +40), (15, -40), (21, +25), (24, -25)],
        "LTO": [(13, +40), (14, -40), (19, +25), (22, -25)],
        "IVR": [(12, +35), (11, -35), (17, +40), (16, -40)],
    }

    def compute_extreme(terms, maximise):
        total = 0
        for q, coeff in terms:
            if maximise:
                val = 5 if coeff > 0 else 1
            else:
                val = 1 if coeff > 0 else 5
            total += coeff * val
        return total

    bounds = {}
    for dim, terms in dimension_terms.items():
        lo = compute_extreme(terms, maximise=False)
        hi = compute_extreme(terms, maximise=True)
        bounds[dim] = (lo, hi)
        print(f"  {dim}: theoretical range [{lo}, {hi}]")
    return bounds


def normalise_dimensions(raw: dict, bounds: dict) -> dict:
    """Linear rescale each dimension score to 0-100."""
    normalised = {}
    for dim, score in raw.items():
        lo, hi = bounds[dim]
        normalised[dim] = round(100 * (score - lo) / (hi - lo), 2)
    return normalised

# ── API Call ──────────────────────────────────────────────────────────────────

def run_single_seed(prompt: str, seed: int) -> dict[int, int]:
    """Run one seed and parse the 24 responses into a dict {question_n: score}."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    scores = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept formats: "Question 1: 2", "1: 2", "1. 2", "1 2"
        for fmt in [
            lambda l: (int(l.split(":")[0].lower().replace("question","").strip()),
                       int(l.split(":")[1].strip().split()[0]))
            if ":" in l else None,
            lambda l: (int(l.split(".")[0].strip()), int(l.split(".")[1].strip().split()[0]))
            if "." in l and l.split(".")[0].strip().isdigit() else None,
        ]:
            try:
                result = fmt(line)
                if result and 1 <= result[0] <= 24 and 1 <= result[1] <= 5:
                    scores[result[0]] = result[1]
                    break
            except Exception:
                continue
    return scores

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    prompt = build_prompt()

    print(f"Running VSM13 across {N_SEEDS} seeds...\n")

    all_scores: dict[int, list[int]] = {n: [] for n in range(1, 25)}
    raw_results = []

    for seed_idx in range(1, N_SEEDS + 1):
        print(f"  Seed {seed_idx}/{N_SEEDS}...", end=" ", flush=True)
        scores = run_single_seed(prompt, seed_idx)
        parsed = len(scores)
        print(f"parsed {parsed}/24 questions")
        raw_results.append(scores)
        for q, s in scores.items():
            all_scores[q].append(s)

    # Compute means
    mu = {}
    print("\n── Per-question means ──────────────────────────────────────────")
    for q in range(1, 25):
        vals = all_scores[q]
        if vals:
            mu[q] = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  Q{q:02d}: mean={mu[q]:.2f}  sd={sd:.2f}  n={len(vals)}  raw={vals}")
        else:
            mu[q] = 3.0  # neutral fallback if question was never parsed
            print(f"  Q{q:02d}: MISSING — defaulting to 3.0")

    dims   = compute_dimensions(mu)
    bounds = get_formula_bounds()
    normed = normalise_dimensions(dims, bounds)

    print("\n── Raw Dimension Scores ────────────────────────────────────────")
    for dim, score in dims.items():
        print(f"  {dim}: {score:.2f}")

    print("\n── Normalised Scores (0-100) vs Hofstede Australia ─────────────")
    hofstede_au = {"PDI": 38, "IDV": 73, "MAS": 61, "UAI": 51, "LTO": 56, "IVR": 71}
    for dim, score in normed.items():
        gt = hofstede_au[dim]
        diff = score - gt
        print(f"  {dim}: {score:.1f}  (Hofstede AU: {gt},  diff: {diff:+.1f})")

    # Save full results
    output = {
        "model": MODEL_NAME,
        "n_seeds": N_SEEDS,
        "persona": "Oliver (Australian, male, 28)",
        "per_question_means": {str(q): round(v, 4) for q, v in mu.items()},
        "dimension_scores": {k: round(v, 4) for k, v in dims.items()},
        "raw_seed_responses": raw_results,
        "normalised scores": normed,
    }
    with open("vsm13_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nFull results saved to vsm13_results.json")

if __name__ == "__main__":
    main()