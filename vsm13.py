"""
File: vsm13.py
Description: VSM13 administration module.
Can be run standalone for T0 measurement or called from simulation.py
for longitudinal drift measurement.
"""

import json
import time
from pathlib import Path
from config import MODEL_NAME, client

MODEL = MODEL_NAME
TEMPERATURE = 0
RESULTS_DIR = Path("results")

VSM13_QUESTIONS = {
    1:  ("job",     "Have sufficient time for your personal or home life"),
    2:  ("job",     "Have a boss (direct superior) you can respect"),
    3:  ("job",     "Get recognition for good performance"),
    4:  ("job",     "Have security of employment"),
    5:  ("job",     "Have pleasant people to work with"),
    6:  ("job",     "Do work that is interesting"),
    7:  ("job",     "Be consulted by your boss in decisions involving your work"),
    8:  ("job",     "Live in a desirable area"),
    9:  ("job",     "Have a job respected by your family and friends"),
    10: ("job",     "Have chances for promotion"),
    11: ("private", "Keeping time free for fun"),
    12: ("private", "Moderation: having few desires"),
    13: ("private", "Doing a service to a friend"),
    14: ("private", "Thrift (not spending more than needed)"),
    15: ("solo",    "How often do you feel nervous or tense?\n1 = always\n2 = usually\n3 = sometimes\n4 = seldom\n5 = never"),
    16: ("solo",    "Are you a happy person?\n1 = always\n2 = usually\n3 = sometimes\n4 = seldom\n5 = never"),
    17: ("solo",    "Do other people or circumstances ever prevent you from doing what you really want to?\n1 = yes, always\n2 = yes, usually\n3 = sometimes\n4 = no, seldom\n5 = no, never"),
    18: ("solo",    "All in all, how would you describe your state of health these days?\n1 = very good\n2 = good\n3 = fair\n4 = poor\n5 = very poor"),
    19: ("solo",    "How proud are you to be a citizen of your country?\n1 = very proud\n2 = fairly proud\n3 = somewhat proud\n4 = not very proud\n5 = not proud at all"),
    20: ("solo",    "How often, in your experience, are subordinates afraid to contradict their boss (or students their teacher)?\n1 = never\n2 = seldom\n3 = sometimes\n4 = usually\n5 = always"),
    21: ("agree",   "One can be a good manager without having a precise answer to every question that a subordinate may raise about his or her work"),
    22: ("agree",   "Persistent efforts are the surest way to results"),
    23: ("agree",   "An organisation structure in which certain subordinates have two bosses should be avoided"),
    24: ("agree",   "A company's or organisation's rules should not be broken — not even when the employee thinks breaking the rule would be in the organisation's best interest"),
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
    "solo": "",
    "agree": (
        "To what extent do you agree or disagree with each of the following statements?\n\n"
        "1 = strongly agree\n"
        "2 = agree\n"
        "3 = undecided\n"
        "4 = disagree\n"
        "5 = strongly disagree"
    ),
}

DIMENSIONS = ["PDI", "IDV", "MAS", "UAI", "LTO", "IVR"]


def compute_dimensions(means: dict) -> dict:
    m = means
    PDI = 35 * (m[7]  - m[2])  + 25 * (m[20] - m[23])
    IDV = 35 * (m[4]  - m[1])  + 35 * (m[9]  - m[6])
    MAS = 35 * (m[5]  - m[3])  + 25 * (m[8]  - m[10])
    UAI = 40 * (m[18] - m[15]) + 25 * (m[21] - m[24])
    LTO = 40 * (m[13] - m[14]) + 25 * (m[19] - m[22])
    IVR = 35 * (m[12] - m[11]) + 40 * (m[17] - m[16])
    return {"PDI": PDI, "IDV": IDV, "MAS": MAS, "UAI": UAI, "LTO": LTO, "IVR": IVR}


def build_system_prompt_from_agent(agent, curr_time=None) -> str:
    identity = (
        f"You are {agent.name}, a {agent.age}-year-old {agent.gender} "
        f"from {agent.country}. "
        f"Your education level is {agent.education} and "
        f"your occupation is {agent.job_type}."
    )

    # check if there are any non-seed memories
    # seed memories are created at "Day 00, Initialisation"
    non_seed_memories = [
        n for n in agent.memory.seq_event + agent.memory.seq_thought
        if n.created != "Day 00, Initialisation"
    ]

    if curr_time and len(non_seed_memories) > 0:
        # simulation is running — retrieve relevant memories
        retrieved = agent.retrieve(
            ["personal values and beliefs",
             "attitudes toward work and authority",
             "relationships with others"],
            curr_time
        )
        memory_context = ""
        seen_ids = set()
        for nodes in retrieved.values():
            for node in nodes[:3]:
                if node.node_id not in seen_ids:
                    memory_context += f"- {node.description}\n"
                    seen_ids.add(node.node_id)
    else:
        # T0 or no simulation memories yet — use all seed memories directly
        memory_context = "\n".join(
            f"- {node.description}"
            for node in reversed(agent.memory.seq_event)
        ) if agent.memory.seq_event else ""

    return (
        f"{identity}\n\n"
        f"The following reflects your genuine beliefs and values:\n"
        f"{memory_context}\n\n"
        f"You are now completing a survey. Answer each question as yourself — "
        f"based on your own values, beliefs, and life experience. "
        f"For each question, respond in this exact format: "
        f"[number (1, 2, 3, 4, or 5)] - [explanation in 1-2 sentences]. "
        f"The number must come first. No other text."
    )


def ask_question(system_prompt: str, q_num: int,
                 q_type: str, q_text: str, tracker=None) -> tuple:
    """
    Administers a single VSM13 question and returns (score, explanation).
    """
    instructions = SECTION_INSTRUCTIONS[q_type]
    if instructions:
        user_message = f"{instructions}\n\nQuestion {q_num}: {q_text}\n\nYour answer:"
    else:
        user_message = f"Question {q_num}: {q_text}\n\nYour answer:"

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
    )
    if tracker:
        tracker.add(response.usage)

    reply = response.choices[0].message.content
    if reply is None:
        print(f"  WARNING: Empty response for Q{q_num}. Defaulting to 3.")
        return 3, "No response"
    reply = reply.strip()

    part = reply.split("-")[0].strip()
    score = None
    for char in part:
        if char in "12345":
            score = int(char)
            break

    explanation = reply.split("-", 1)[1].strip() if "-" in reply else reply

    if score is None:
        print(f"  WARNING: Could not parse score from '{reply}' for Q{q_num}. Defaulting to 3.")
        score = 3

    return score, explanation


def run_vsm13(agent, curr_time=None, tracker=None) -> dict:
    """
    Administers all 24 VSM13 questions to an Agent object.
    Returns dict mapping question number to response score.
    Used by simulation.py for longitudinal measurement.
    """
    system_prompt = build_system_prompt_from_agent(agent, curr_time)
    responses = {}
    explanations = {}

    for q_num, (q_type, q_text) in VSM13_QUESTIONS.items():
        score, explanation = ask_question(
            system_prompt, q_num, q_type, q_text, tracker=tracker
        )
        responses[q_num] = score
        explanations[q_num] = explanation
        print(f"    Q{q_num:02d}: {score} - {explanation}")
        time.sleep(0.3)

    return responses, explanations