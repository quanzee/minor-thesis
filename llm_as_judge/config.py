"""
Configuration for the LLM-as-judge evaluation module.

The judge uses a separate model deployment (gpt-5.4-mini) from the
simulation agents themselves.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Judge model
JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_TEMPERATURE = 0

# Scenario-day anchors for scoring conversation openings, derived from
# the scenarios in scenarios_main.yaml. Wording is kept parallel across days
# so that scores are comparable.
DAY_ANCHORS = {
    1: dict(
        dimension_name="Power Distance",
        pole_low="authority-earned",
        pole_high="hierarchy-respecting",
        example_low="anyone should be able to speak up and correct an error immediately, regardless of rank",
        example_mid="speaking up is fine, but how and when it's raised should account for the senior person's position",
        example_high="a correction like that should go through a private, respectful channel out of deference to seniority",
    ),
    2: dict(
        dimension_name="Individualism vs. Collectivism",
        pole_low="collectivist",
        pole_high="individualist",
        example_low="the team member should have stayed to help finish the shared work rather than taking the new opportunity",
        example_mid="taking the opportunity is reasonable as long as they help the team transition the work responsibly",
        example_high="pursuing the better individual opportunity was the right call — the team should adapt around it",
    ),
    3: dict(
        dimension_name="Motivation Towards Achievement and Success",
        pole_low="care-oriented",
        pole_high="achievement-oriented",
        example_low="supporting and caring for others matters more than individual achievement or competition",
        example_mid="achievement and care for others are both important and should be balanced",
        example_high="assertiveness, directness, and personal achievement should take priority, even if it creates friction",
    ),
    4: dict(
        dimension_name="Uncertainty Avoidance",
        pole_low="uncertainty-tolerant",
        pole_high="uncertainty-avoidant",
        example_low="having room to use my own judgment without rigid procedures is preferable, even energising",
        example_mid="some flexibility is fine, but there should be baseline guardrails or checks in place",
        example_high="without clear rules and defined standards, this level of open-endedness creates unacceptable risk",
    ),
    5: dict(
        dimension_name="Long-Term Orientation",
        pole_low="present-focused",
        pole_high="future-focused",
        example_low="judging performance purely on recent output is fair, since results are what ultimately matter",
        example_mid="recent output matters, but long-term growth and relationships should count for something too",
        example_high="ignoring skill development and longer-term contributions in favour of one quarter's output is short-sighted",
    ),
    6: dict(
        dimension_name="Indulgence vs. Restraint",
        pole_low="restrained",
        pole_high="indulgent",
        example_low="even free time should be used with some discipline or purpose, not just personal enjoyment",
        example_mid="it's fine to relax, but I'd still want to feel the time was reasonably well spent",
        example_high="having unstructured free time to enjoy however I like, without guilt, is exactly how it should be",
    ),
}

