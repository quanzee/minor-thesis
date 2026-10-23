"""
VSM13 administration, called from simulation.py at T0 and at the end of
each simulation day.

Each question is answered with belief context retrieved for its Hofstede
dimension. The source depends on agent.memory_mode:
    two_tier: CoreBeliefs (Tier 1). With agent.frozen_core, the top Tier 2
        thoughts for the same focal point are appended.
    single_tier: AssociativeMemory (Tier 2) thought nodes, which include
        the persona seeds.
"""

import json
import time
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).parent
_LIB_DIR = Path(__file__).parent.parent / "lib"

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_LIB_DIR))

from config import api_call_with_retry

TEMPERATURE = 0
SEED_CREATED = "Day 00, Initialisation"

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

QUESTION_DIMENSION = {
    1: "IDV", 2: "PDI", 3: "MAS", 4: "IDV", 5: "MAS",
    6: "IDV", 7: "PDI", 8: "MAS", 9: "IDV", 10: "MAS",
    11: "IVR", 12: "IVR", 13: "LTO", 14: "LTO",
    15: "UAI", 16: "IVR", 17: "IVR", 18: "UAI",
    19: "LTO", 20: "PDI", 21: "UAI", 22: "LTO",
    23: "PDI", 24: "UAI"
}

DIMENSION_FOCAL_POINTS = {
    "PDI": "personal values and beliefs related to authority, hierarchy, and deference to those in power",
    "IDV": "personal values and beliefs related to individual goals versus group loyalty and collective identity",
    "MAS": "personal values and beliefs related to achievement, ambition, competition, and assertiveness versus cooperation",
    "UAI": "personal values and beliefs related to uncertainty, rules, structure, and tolerance for ambiguity",
    "LTO": "personal values and beliefs related to tradition, long-term planning, thrift, and perseverance",
    "IVR": "personal values and beliefs related to enjoyment, leisure, fun, and personal freedom",
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
    MAS = 35 * (m[5]  - m[3])  + 35 * (m[8]  - m[10])
    UAI = 40 * (m[18] - m[15]) + 25 * (m[21] - m[24])
    LTO = 40 * (m[13] - m[14]) + 25 * (m[19] - m[22])
    IVR = 35 * (m[12] - m[11]) + 40 * (m[17] - m[16])
    return {"PDI": PDI, "IDV": IDV, "MAS": MAS, "UAI": UAI, "LTO": LTO, "IVR": IVR}


def build_system_prompt(agent) -> str:
    """
    Builds the system prompt with the agent's identity and the survey
    instructions. Beliefs are added per question in ask_question.
    """
    return (
        f"You are {agent.name}, a {agent.age}-year-old {agent.gender} "
        f"from {agent.country}. "
        f"Your education level is {agent.education} and "
        f"your occupation is {agent.job_type}.\n\n"
        f"You are now completing a survey. Answer each question as yourself — "
        f"based on your own values, beliefs, and life experience. "
        f"For each question, respond in this exact format: "
        f"[number (1, 2, 3, 4, or 5)] - [explanation in 1-2 sentences]. "
        f"The number must come first. No other text."
    )


def _retrieve_two_tier(agent, dimension, curr_time, top_n) -> tuple:
    """
    Retrieves active CoreBeliefs (Tier 1) by semantic relevance.

    With agent.frozen_core, the top Tier 2 thoughts for the same focal
    point are appended, retrieved as in single_tier. Log entries are
    tagged "tier": 1 for core beliefs and "tier": 2 for thoughts.
    """
    focal_point = DIMENSION_FOCAL_POINTS[dimension]
    query_embedding = agent.embedding_fn(focal_point)
    retrieved_beliefs = agent.core.retrieve(query_embedding, top_n=top_n)

    belief_lines = [f"- {b.description}" for b in retrieved_beliefs]
    retrieved_nodes_log = [
        {
            "tier": 1,
            "node_id": b.node_id,
            "description": b.description,
            "created": b.created,
            "is_active": b.is_active
        }
        for b in retrieved_beliefs
    ]

    if getattr(agent, "frozen_core", False):
        thought_context, thought_log = _retrieve_single_tier(agent, dimension, curr_time, top_n)
        if thought_context:
            belief_lines.append(thought_context)
        retrieved_nodes_log.extend({"tier": 2, **entry} for entry in thought_log)

    return "\n".join(belief_lines), retrieved_nodes_log


def _retrieve_single_tier(agent, dimension, curr_time, top_n) -> tuple:
    """
    Retrieves AssociativeMemory (Tier 2) thought nodes for the dimension's
    focal point, following Park et al. (2023). last_accessed is not
    updated, so the survey does not alter the agent's memory.
    """
    focal_point = DIMENSION_FOCAL_POINTS[dimension]
    retrieved = agent.retrieve([focal_point], curr_time, n=top_n, node_types=["thought"], update_access=False)

    belief_lines = []
    retrieved_nodes_log = []

    seen_ids = set()
    for nodes in retrieved.values():
        for node in nodes[:top_n]:
            if node.node_id not in seen_ids and node.node_type == "thought":
                belief_lines.append(f"- {node.description}")
                seen_ids.add(node.node_id)
                retrieved_nodes_log.append({
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "description": node.description,
                    "created": node.created,
                    "poignancy": node.poignancy,
                    "is_seed": node.created == SEED_CREATED,
                    "recency": round(getattr(node, "recency", None) or 0.0, 4),
                    "importance": round(getattr(node, "importance", None) or 0.0, 4),
                    "relevance": round(getattr(node, "relevance", None) or 0.0, 4),
                })
    return "\n".join(belief_lines), retrieved_nodes_log


def retrieve_dimension_beliefs(agent, dimension, curr_time, top_n=3) -> tuple:
    """
    Retrieves beliefs relevant to a Hofstede dimension from the source
    given by agent.memory_mode. Returns (belief_context, nodes_log).
    """
    if agent.memory_mode == "two_tier":
        return _retrieve_two_tier(agent, dimension, curr_time, top_n)
    else:
        return _retrieve_single_tier(agent, dimension, curr_time, top_n)


def ask_question(system_prompt: str, q_num: int, q_type: str, q_text: str,
                 belief_context: str, client, model, tracker=None) -> tuple:
    """
    Administers one VSM13 question with its belief context.
    Returns (score, explanation).
    """
    instructions = SECTION_INSTRUCTIONS[q_type]

    belief_block = (
        f"Your relevant values and beliefs for this question:\n{belief_context}\n\n"
        if belief_context else ""
    )

    if instructions:
        user_message = f"{belief_block}{instructions}\n\nQuestion {q_num}: {q_text}\n\nYour answer:"
    else:
        user_message = f"{belief_block}Question {q_num}: {q_text}\n\nYour answer:"

    response = api_call_with_retry(lambda: client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
    ))
    if response is None:
        print(f"  WARNING: Content filter triggered for Q{q_num}. Defaulting to 3.")
        return 3, "Content filter triggered"

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


def retrieve_all_dimension_beliefs(agent, curr_time=None, top_n=3) -> dict:
    """
    Retrieves belief context for each of the six Hofstede dimensions.
    Returns {dimension: (belief_context, nodes_log)}.

    Called once per agent per day, with the result reused across the
    day's repeated runs, so that the runs differ only in generation and
    not in retrieved context.
    """
    return {
        dimension: retrieve_dimension_beliefs(agent, dimension, curr_time, top_n=top_n)
        for dimension in DIMENSIONS
    }


def log_vsm13_retrieval(agent, curr_time, belief_contexts: dict):
    """
    Writes one retrieval log entry for a belief_contexts dict, so that
    retrieval reused across runs is logged once.
    """
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    with open(agent.logs_dir / "vsm13_retrieval.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "agent": agent.name,
            "culture": agent.country,
            "curr_time": curr_time,
            "retrieval_by_dimension": {
                dim: {"n_retrieved": len(nodes_log), "nodes": nodes_log}
                for dim, (_, nodes_log) in belief_contexts.items()
            }
        }, ensure_ascii=False) + "\n")


def run_vsm13(agent, curr_time=None, tracker=None, top_n=3, belief_contexts=None) -> tuple:
    """
    Administers all 24 VSM13 questions to an agent.
    Returns (responses, explanations).

    belief_contexts: optional output of retrieve_all_dimension_beliefs(),
        reused across repeated runs on the same day. If omitted, beliefs
        are retrieved and logged within this call.
    """
    system_prompt = build_system_prompt(agent)
    responses = {}
    explanations = {}

    own_retrieval = belief_contexts is None
    if own_retrieval:
        belief_contexts = retrieve_all_dimension_beliefs(agent, curr_time=curr_time, top_n=top_n)

    for q_num, (q_type, q_text) in VSM13_QUESTIONS.items():
        dimension = QUESTION_DIMENSION[q_num]
        belief_context, _ = belief_contexts[dimension]

        score, explanation = ask_question(
            system_prompt, q_num, q_type, q_text,
            belief_context, agent.client, agent.model, tracker=tracker
        )
        responses[q_num] = score
        explanations[q_num] = explanation
        print(f"    Q{q_num:02d} [{dimension}]: {score} - {explanation}")
        time.sleep(0.3)

    if own_retrieval:
        log_vsm13_retrieval(agent, curr_time, belief_contexts)

    return responses, explanations