"""
Kendall Tau Validity Check
--------------------------
Computes Kendall Tau rank correlation between LLM-generated VSM13 dimension
scores and Hofstede human scores across Australia, Japan, and Malaysia.
Follows the methodology of Masoud et al. (CAT paper).
"""

from scipy.stats import kendalltau

# ── Human Hofstede scores (ground truth) ─────────────────────────────────────
# Source: Hofstede Insights, 2023

HUMAN_SCORES = {
    "Australia": {"PDI": 38, "IDV": 73, "MAS": 61, "UAI": 51, "LTO": 56, "IVR": 71},
    "Japan":     {"PDI": 54, "IDV": 62, "MAS": 95, "UAI": 92, "LTO": 100,"IVR": 42},
    "Malaysia":  {"PDI": 100,"IDV": 27, "MAS": 50, "UAI": 36, "LTO": 47, "IVR": 57},
}

# ── LLM Full Persona scores (from your Day 0 runs) ────────────────────────────
# Replace these with your actual results if they differ

LLM_SCORES = {
    "Australia": {"PDI": 25.00, "IDV": 78.75, "MAS": -8.75, "UAI": -175.00, "LTO": -41.25, "IVR": 167.50},
    "Japan":     {"PDI": 43.2, "IDV": 26.2, "MAS": -1.2, "UAI": -58.8, "LTO": -15.0,"IVR": -14.2},
    "Malaysia":  {"PDI": 53,"IDV": -12.2, "MAS": -46.2, "UAI": -154.8, "LTO": -83.8, "IVR": 118},
}

DIMENSIONS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]
CULTURES   = ["Australia", "Japan", "Malaysia"]


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

        # Build parallel rank vectors in consistent culture order
        human_vec = [human_ranks[c] for c in CULTURES]
        llm_vec   = [llm_ranks[c]   for c in CULTURES]

        tau, p_value = kendalltau(human_vec, llm_vec)
        tau_values.append(tau)
        results[dim] = {"tau": tau, "p_value": p_value,
                        "human_ranking": human_ranks, "llm_ranking": llm_ranks}

        # Format ranking strings for display
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


def main():
    print("=" * 95)
    print("Kendall Tau: LLM Full Persona vs Hofstede Human Scores")
    print("Cultures: Australia, Japan, Malaysia | Dimensions: PDI, IDV, MAS, UAI, LTO, IVR")
    print("=" * 95)

    results = compute_kendall_tau(HUMAN_SCORES, LLM_SCORES)

    # LLM_BASE = {
    #     "Australia": {"PDI": 25.00, "IDV": 35.00, "MAS": 0.00, "UAI": -25.00, "LTO": -40.00, "IVR": 35.00},
    #     "Japan":     {"PDI": 25.00, "IDV": 35.00, "MAS": 0.00, "UAI": -25.00, "LTO": -40.00, "IVR": 35.00},
    #     "Malaysia":  {"PDI": 25.00, "IDV": 35.00, "MAS": 0.00, "UAI": -25.00, "LTO": -40.00, "IVR": 35.00},
    # }

    # print("\n" + "=" * 95)
    # print("Kendall Tau: Base LLM vs Hofstede Human Scores")
    # print("=" * 95)
    # base_results = compute_kendall_tau(HUMAN_SCORES, LLM_BASE)

    # print("\n--- Note on Base LLM ---")
    # print("Base LLM scores are identical across all cultures (no persona).")
    # print("All cultures tie on every dimension, making Tau undefined or 0.")
    # print("This confirms the base model provides no cultural differentiation.")


if __name__ == "__main__":
    main()