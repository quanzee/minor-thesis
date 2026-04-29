import yaml
import json
import time
from pathlib import Path
from config import MODEL_NAME, client
from vsm13 import VSM13_QUESTIONS, SECTION_INSTRUCTIONS, compute_dimensions, ask_question

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL = MODEL_NAME
TEMPERATURE = 0                      # deterministic, per supervisor instruction

client = client

SYSTEMPROMPT = "For each question, respond in this exact format: [number (1, 2, 3, 4, or 5)] - [explanation in 1-2 sentences]. The number must come first. No other text."

def run_vsm13() -> dict:
    print(f"\n  Administering VSM13...")
    system_prompt = SYSTEMPROMPT
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
    responses, explanations = run_vsm13()

    scores = {}
    for qnum, response in responses.items():
        scores[qnum] = response

    # ── Compute dimension scores ──────────────────────────────────────────────
    dimensions = compute_dimensions(scores)

    print("\nDay 0 VSM13 Dimension Scores — Base LLM")
    print("=" * 35)
    for dim, score in dimensions.items():
        print(f"  {dim:>3}: {score:+.2f}")
    print("=" * 35)

    # ── Save to JSON ──────────────────────────────────────────────────────────
    results = {
        "individual_responses": scores,
        "dimension_scores": dimensions,
    }

    output_path = RESULTS_DIR / "base_day0_vsm13_run5.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()