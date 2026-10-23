"""
Kendall's tau-b between LLM-generated VSM13 dimension scores and Hofstede
human scores for Australia, Japan and Malaysia, following Masoud et al.
(2025).
"""

import json
from pathlib import Path
from scipy.stats import kendalltau

# Hofstede human scores (Hofstede Insights, 2023)

HUMAN_SCORES = {
    "Australia": {"PDI": 38, "IDV": 73, "MAS": 61, "UAI": 51, "LTO": 56, "IVR": 71},
    "Japan":     {"PDI": 54, "IDV": 62, "MAS": 95, "UAI": 92, "LTO": 100,"IVR": 42},
    "Malaysia":  {"PDI": 100,"IDV": 27, "MAS": 50, "UAI": 36, "LTO": 47, "IVR": 57},
}

DIMENSIONS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]
CULTURES   = ["Australia", "Japan", "Malaysia"]

# Ranking and Kendall Tau
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

    print(f"\n{'Dimension':<8} {'Human Ranking':<35} {'LLM Ranking':<35} {'Tau':>6}")
    print("-" * 84)

    for dim in DIMENSIONS:
        human_ranks = rank_cultures(human_scores, dim)
        llm_ranks   = rank_cultures(llm_scores, dim)

        # Rounding removes floating-point residue so that equal scores are
        # treated as ties by tau-b.
        human_vec = [human_scores[c][dim] for c in CULTURES]
        llm_vec   = [round(llm_scores[c][dim], 6) for c in CULTURES]

        tau, _ = kendalltau(human_vec, llm_vec)
        tau_values.append(tau)
        results[dim] = {"tau": tau,
                        "human_ranking": human_ranks, "llm_ranking": llm_ranks}

        human_str = " > ".join(sorted(CULTURES, key=lambda c: human_ranks[c]))
        llm_str   = " > ".join(sorted(CULTURES, key=lambda c: llm_ranks[c]))

        print(f"{dim:<8} {human_str:<35} {llm_str:<35} {tau:>+6.2f}")

    avg_tau = sum(tau_values) / len(tau_values)
    results["average_tau"] = avg_tau

    print("-" * 95)
    print(f"{'Average Tau':<79} {avg_tau:>+6.2f}")
    print()
    print("Note: with only n=3 cultures per dimension, tau is interpreted")
    print("as a descriptive/directional indicator only, not as a basis for")
    print("significance testing, consistent with Masoud et al. (2025) and")
    print("Cao et al. (2023).")

    return results

