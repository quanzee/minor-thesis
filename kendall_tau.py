"""
Kendall Tau Validity Check
--------------------------
Computes Kendall Tau rank correlation between LLM-generated VSM13 dimension
scores and Hofstede human scores across Australia, Japan, and Malaysia.
Follows the methodology of Masoud et al. (CAT paper).
"""

import json
from pathlib import Path
from scipy.stats import kendalltau

# ── Configuration ─────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")

# Map culture names to their result JSON files
RESULT_FILES = {
    "Australia": "noseeds_australia_day0_vsm13_run5.json",
    "Japan":     "noseeds_japan_day0_vsm13_run5.json",
    "Malaysia":  "noseeds_malaysia_day0_vsm13_run5.json",
}

# ── Human Hofstede scores (ground truth) ─────────────────────────────────────
# Source: Hofstede Insights, 2023

HUMAN_SCORES = {
    "Australia": {"PDI": 38, "IDV": 73, "MAS": 61, "UAI": 51, "LTO": 56, "IVR": 71},
    "Japan":     {"PDI": 54, "IDV": 62, "MAS": 95, "UAI": 92, "LTO": 100,"IVR": 42},
    "Malaysia":  {"PDI": 100,"IDV": 27, "MAS": 50, "UAI": 36, "LTO": 47, "IVR": 57},
}

DIMENSIONS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]
CULTURES   = ["Australia", "Japan", "Malaysia"]


# ── Score loading ─────────────────────────────────────────────────────────────

def load_llm_scores(use_individual: bool = False) -> dict:
    """
    Loads LLM dimension scores from JSON result files.
    If use_individual=True, averages individual agent scores per culture.
    If use_individual=False, uses the pooled dimension_scores directly.
    """
    llm_scores = {}

    for culture, filename in RESULT_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {path}")

        with open(path) as f:
            data = json.load(f)

        if use_individual:
            # Average dimension scores across individual agents
            individual = data["individual_dimension_scores"]
            agents = list(individual.values())
            averaged = {}
            for dim in DIMENSIONS:
                averaged[dim] = sum(agent[dim] for agent in agents) / len(agents)
            llm_scores[culture] = averaged
        else:
            # Use the pooled dimension scores directly
            llm_scores[culture] = data["dimension_scores"]

    return llm_scores


# ── Ranking and Kendall Tau ───────────────────────────────────────────────────

def rank_cultures(scores: dict, dimension: str) -> dict:
    """
    Ranks cultures on a given dimension (highest score = rank 1).
    Returns dict mapping culture name to rank.
    """
    sorted_cultures = sorted(CULTURES, key=lambda c: scores[c][dimension], reverse=True)
    return {culture: rank + 1 for rank, culture in enumerate(sorted_cultures)}


def compute_kendall_tau(human_scores: dict, llm_scores: dict) -> dict:
    """
    Computes Kendall Tau per dimension and overall average.
    Returns dict with per-dimension tau, p-values, and average tau.
    """
    results = {}
    tau_values = []

    print(f"\n{'Dimension':<8} {'Human Ranking':<35} {'LLM Ranking':<35} {'Tau':>6} {'p-value':>10}")
    print("-" * 95)

    for dim in DIMENSIONS:
        human_ranks = rank_cultures(human_scores, dim)
        llm_ranks   = rank_cultures(llm_scores, dim)

        human_vec = [human_ranks[c] for c in CULTURES]
        llm_vec   = [llm_ranks[c]   for c in CULTURES]

        tau, p_value = kendalltau(human_vec, llm_vec)
        tau_values.append(tau)
        results[dim] = {"tau": tau, "p_value": p_value,
                        "human_ranking": human_ranks, "llm_ranking": llm_ranks}

        human_str = " > ".join(sorted(CULTURES, key=lambda c: human_ranks[c]))
        llm_str   = " > ".join(sorted(CULTURES, key=lambda c: llm_ranks[c]))

        print(f"{dim:<8} {human_str:<35} {llm_str:<35} {tau:>+6.2f} {p_value:>10.4f}")

    avg_tau = sum(tau_values) / len(tau_values)
    results["average_tau"] = avg_tau

    print("-" * 95)
    print(f"{'Average Tau':<79} {avg_tau:>+6.2f}")
    print()
    print("Note: p-values are unreliable at n=3 cultures per dimension.")
    print("Tau is interpreted as a directional indicator only, consistent")
    print("with Masoud et al. (2025) and Cao et al. (2023).")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 95)
    print("Kendall Tau: LLM Full Persona vs Hofstede Human Scores")
    print("Cultures: Australia, Japan, Malaysia | Dimensions: PDI, IDV, MAS, UAI, LTO, IVR")
    print("=" * 95)

    llm_scores = load_llm_scores(use_individual=False)
    compute_kendall_tau(HUMAN_SCORES, llm_scores)


if __name__ == "__main__":
    main()