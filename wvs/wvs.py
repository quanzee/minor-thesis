"""
World Values Survey (WVS) administration, used alongside VSM13 as a
supplementary convergent-validity measure. Uses the paraphrased question
bank from WorldValuesBench (Zhao et al., 2024), filtered to three
categories: Social Values, Norms and Stereotypes; Social Capital, Trust
and Organisational Membership; and Economic Values.

Because WVS items do not map onto a single Hofstede dimension, beliefs are
retrieved using each question's own text as the focal point. The source
depends on agent.memory_mode, as in vsm13.py:
    two_tier: CoreBeliefs (Tier 1). With agent.frozen_core, the top Tier 2
        thoughts for the same question are appended.
    single_tier: AssociativeMemory (Tier 2) thought nodes.
"""

import json
import time
from pathlib import Path
import sys

_WVS_DIR = Path(__file__).parent
_LIB_DIR = Path(__file__).parent.parent / "simulation_architecture" / "lib"

sys.path.insert(0, str(_WVS_DIR))
sys.path.insert(0, str(_LIB_DIR))

from config import api_call_with_retry

TEMPERATURE = 0
SEED_CREATED = "Day 00, Initialisation"

# Filtered WVS question bank (75 questions, 3 categories)
WVS_QUESTIONS_PATH = WVS_QUESTIONS_PATH = Path(__file__).parent / "wvs_filtered_questions.json"

def load_wvs_questions() -> dict:
    """
    Loads the filtered WVS question bank, keyed by question ID (e.g. "Q1").
    Each entry has question, category, use_case, answer_data_type,
    answer_scale_min and answer_scale_max.
    """
    with open(WVS_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


WVS_QUESTIONS = load_wvs_questions()
_QUESTION_EMBEDDING_CACHE = {}

CATEGORIES = [
    "Social Values, Norms, Stereotypes",
    "Social Capital, Trust and Organizational Membership",
    "Economic Values",
]


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
        f"based on your own values, beliefs, and life experience, exactly as "
        f"you see things right now. "
        f"For each question, respond in this exact format: "
        f"[number] - [explanation in 1-2 sentences]. "
        f"The number must come first, and must be an integer within the "
        f"scale given for that question. No other text."
    )


def _retrieve_two_tier(agent, q_id, question_text, curr_time, top_n) -> tuple:
    """
    Retrieves active CoreBeliefs (Tier 1) by semantic relevance. Question
    embeddings are cached, since question text does not change.

    With agent.frozen_core, the top Tier 2 thoughts for the same question
    are appended, retrieved as in single_tier. Log entries are tagged
    "tier": 1 for core beliefs and "tier": 2 for thoughts.
    """
    if q_id not in _QUESTION_EMBEDDING_CACHE:
        _QUESTION_EMBEDDING_CACHE[q_id] = agent.embedding_fn(question_text)

    query_embedding = _QUESTION_EMBEDDING_CACHE[q_id]
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
        thought_context, thought_log = _retrieve_single_tier(agent, q_id, question_text, curr_time, top_n)
        if thought_context:
            belief_lines.append(thought_context)
        retrieved_nodes_log.extend({"tier": 2, **entry} for entry in thought_log)

    return "\n".join(belief_lines), retrieved_nodes_log


def _retrieve_single_tier(agent, q_id, question_text, curr_time, top_n) -> tuple:
    """
    Retrieves AssociativeMemory (Tier 2) thought nodes using the question
    text as the focal point, following Park et al. (2023). last_accessed
    is not updated, so the survey does not alter the agent's memory.
    """
    retrieved = agent.retrieve([question_text], curr_time, n=top_n, node_types=["thought"], update_access=False)

    belief_lines = []
    retrieved_nodes_log = []
    seen_ids = set()
    for nodes in retrieved.values():
        for node in nodes[:top_n]:
            if node.node_id not in seen_ids:
                belief_lines.append(f"- {node.description}")
                seen_ids.add(node.node_id)
                retrieved_nodes_log.append({
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "description": node.description,
                    "created": node.created,
                    "poignancy": node.poignancy,
                    "is_seed": node.created == SEED_CREATED,
                })

    return "\n".join(belief_lines), retrieved_nodes_log


def retrieve_question_beliefs(agent, q_id, question_text, curr_time=None, top_n=3) -> tuple:
    """
    Retrieves beliefs relevant to a WVS question from the source given by
    agent.memory_mode. Returns (belief_context, nodes_log).
    """
    if agent.memory_mode == "two_tier":
        return _retrieve_two_tier(agent, q_id, question_text, curr_time, top_n)
    else:
        return _retrieve_single_tier(agent, q_id, question_text, curr_time, top_n)


def ask_question(system_prompt: str, q_id: str, q_text: str, scale_min: int,
                 scale_max: int, belief_context: str, client, model,
                 tracker=None) -> tuple:
    """
    Administers one WVS question with its belief context.
    Returns (score, explanation).
    """
    belief_block = (
        f"Your relevant values and beliefs for this question:\n{belief_context}\n\n"
        if belief_context else ""
    )

    scale_note = f"(Answer with an integer from {scale_min} to {scale_max}.)"
    user_message = f"{belief_block}{q_text}\n{scale_note}\n\nYour answer:"

    response = api_call_with_retry(lambda: client.chat.completions.create(
        model=model,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message}
        ],
    ))
    if response is None:
        default = round((scale_min + scale_max) / 2)
        print(f"  WARNING: Content filter triggered for {q_id}. Defaulting to {default}.")
        return default, "Content filter triggered"

    if tracker:
        tracker.add(response.usage)

    reply = response.choices[0].message.content
    if reply is None:
        default = round((scale_min + scale_max) / 2)
        print(f"  WARNING: Empty response for {q_id}. Defaulting to {default}.")
        return default, "No response"
    reply = reply.strip()

    part = reply.split("-")[0].strip()
    # leading integer, which may have two digits on 10-point scales
    digits = ""
    for char in part:
        if char.isdigit():
            digits += char
        elif digits:
            break
    score = int(digits) if digits else None

    explanation = reply.split("-", 1)[1].strip() if "-" in reply else reply

    if score is None or not (scale_min <= score <= scale_max):
        default = round((scale_min + scale_max) / 2)
        print(f"  WARNING: Could not parse valid score from '{reply}' for {q_id} "
              f"(expected {scale_min}-{scale_max}). Defaulting to {default}.")
        score = default

    return score, explanation


def retrieve_all_beliefs(agent, curr_time=None, top_n=3) -> dict:
    """
    Retrieves belief context for every WVS question.
    Returns {q_id: (belief_context, nodes_log)}.

    Called once per agent per day, with the result reused across the
    day's repeated runs, so that the runs differ only in generation and
    not in retrieved context.
    """
    return {
        q_id: retrieve_question_beliefs(agent, q_id, meta["question"], curr_time=curr_time, top_n=top_n)
        for q_id, meta in WVS_QUESTIONS.items()
    }


def log_wvs_retrieval(agent, curr_time, belief_contexts: dict):
    """
    Writes one retrieval log entry for a belief_contexts dict, so that
    retrieval reused across runs is logged once.
    """
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    with open(agent.logs_dir / "wvs_retrieval.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "agent": agent.name,
            "culture": agent.country,
            "curr_time": curr_time,
            "retrieval_by_question": {
                q_id: {"n_retrieved": len(nodes_log), "nodes": nodes_log}
                for q_id, (_, nodes_log) in belief_contexts.items()
            }
        }, ensure_ascii=False) + "\n")


def run_wvs(agent, curr_time=None, tracker=None, top_n=3, belief_contexts=None) -> tuple:
    """
    Administers all filtered WVS questions to an agent.
    Returns (responses, explanations).

    belief_contexts: optional output of retrieve_all_beliefs(), reused
        across repeated runs on the same day. If omitted, beliefs are
        retrieved and logged within this call.
    """
    system_prompt = build_system_prompt(agent)
    responses = {}
    explanations = {}

    own_retrieval = belief_contexts is None
    if own_retrieval:
        belief_contexts = retrieve_all_beliefs(agent, curr_time=curr_time, top_n=top_n)

    for q_id, meta in WVS_QUESTIONS.items():
        q_text = meta["question"]
        scale_min = int(meta["answer_scale_min"])
        scale_max = int(meta["answer_scale_max"])
        category = meta["category"]

        belief_context, _ = belief_contexts[q_id]

        score, explanation = ask_question(
            system_prompt, q_id, q_text, scale_min, scale_max,
            belief_context, agent.client, agent.model, tracker=tracker
        )
        responses[q_id] = score
        explanations[q_id] = explanation

        print(f"    {q_id} [{category}]: {score} - {explanation}")
        time.sleep(0.3)

    if own_retrieval:
        log_wvs_retrieval(agent, curr_time, belief_contexts)

    return responses, explanations


def compute_drift(pre_responses: dict, post_responses: dict) -> dict:
    """
    Compares two sets of canonical (majority-vote) responses for the same
    agent and records whether each question's answer changed.

    WVS items use 2, 4, 5 and 10-point scales, so differences in magnitude
    are not comparable across items. Drift is therefore measured as
    changed versus unchanged per question.

    Returns:
        per_question: {q_id: True if the answer changed}
        flip_rate_by_scale: share of changed items per scale length
        overall_flip_rate: share of changed items across all questions
    """
    from collections import defaultdict

    per_question = {}
    flips_by_scale = defaultdict(list)  # scale_size -> list of bool

    for q_id, meta in WVS_QUESTIONS.items():
        if q_id not in pre_responses or q_id not in post_responses:
            continue

        scale_min = int(meta["answer_scale_min"])
        scale_max = int(meta["answer_scale_max"])
        scale_size = scale_max - scale_min + 1

        changed = post_responses[q_id] != pre_responses[q_id]
        per_question[q_id] = changed
        flips_by_scale[scale_size].append(changed)

    flip_rate_by_scale = {
        f"{scale_size}_point": sum(flips) / len(flips)
        for scale_size, flips in flips_by_scale.items()
        if flips
    }


    all_flips = [f for flips in flips_by_scale.values() for f in flips]
    overall_flip_rate = (
        sum(all_flips) / len(all_flips) if all_flips else None
    )

    return {
        "per_question": per_question,
        "flip_rate_by_scale": flip_rate_by_scale,
        "overall_flip_rate": overall_flip_rate,
    }