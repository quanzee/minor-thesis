"""
End-of-day deep reflection for two_tier agents.

The agent is shown its Tier 2 (AssociativeMemory) thoughts from a rolling
window alongside its most relevant active Tier 1 (CoreBeliefs) beliefs, and
is asked whether any belief should be revised or a new belief added. The
update decision is a single LLM judgement rather than a similarity
threshold on thought embeddings.

Runs once per agent at the end of each simulation day, for two_tier runs
without --frozen_core. None of the conditions in this study use it: the
two-tier condition (C2) uses a fixed core, so deep reflection is skipped.

Parameters:
    MIN_THOUGHT_COUNT   : minimum Tier 2 thoughts in the window before
                          reflection is attempted
    ROLLING_WINDOW_DAYS : number of days of Tier 2 thoughts considered
    TOP_N_BELIEFS_SHOWN : number of active Tier 1 beliefs shown as revision
                          candidates, ranked by semantic similarity
"""

import sys
from pathlib import Path
_LIB_DIR = Path(__file__).parent / "lib"
sys.path.insert(0, str(_LIB_DIR))

from config import api_call_with_retry
import re

FIELD_PATTERN = re.compile(
    r"UPDATE_TYPE:\s*(?P<update_type>.*?)\s*\n"
    r"TARGET_BELIEF_NUMBER:\s*(?P<target>.*?)\s*\n"
    r"UPDATED_TEXT:\s*(?P<updated_text>.*?)\s*\n"
    r"DIMENSION:\s*(?P<dimension>.*)",
    re.DOTALL
)

BELIEF_LINE_PATTERN = re.compile(r"BELIEF_(\d+):\s*(.*)")

# The model states the dimension of each revised or new belief rather than
# inheriting the tag of the belief it replaces, because candidate beliefs
# are selected by similarity, not by dimension.
DIMENSION_CODES = {"PDI", "IDV", "MAS", "UAI", "LTO", "IVR"}

# Parameters
MIN_THOUGHT_COUNT    = 3
ROLLING_WINDOW_DAYS  = 1
TOP_N_BELIEFS_SHOWN  = 5


def get_recent_thoughts(agent, current_day: int) -> list:
    """
    Returns the agent's Tier 2 thought nodes created within the last
    ROLLING_WINDOW_DAYS days.
    """
    cutoff_day = max(1, current_day - ROLLING_WINDOW_DAYS + 1)
    recent = []
    for node in agent.memory.seq_thought:
        # created is formatted "Day XX, Round"
        try:
            day_str = node.created.split(",")[0].strip()
            day_num = int(day_str.split()[1])
            if day_num >= cutoff_day:
                recent.append(node)
        except (IndexError, ValueError):
            continue
    return recent


def get_candidate_beliefs(agent, thoughts: list) -> list:
    """
    Returns the active Tier 1 beliefs most similar to the centroid of the
    window's thought embeddings, most relevant first.
    """
    centroid = [0.0] * len(thoughts[0].embedding)
    for node in thoughts:
        for i, val in enumerate(node.embedding):
            centroid[i] += val
    centroid = [v / len(thoughts) for v in centroid]

    return agent.core.retrieve(centroid, top_n=TOP_N_BELIEFS_SHOWN)


def judge_belief_update(agent, thoughts: list, candidate_beliefs: list,
                        client, model, tracker=None) -> dict:
    """
    Asks the agent whether any candidate belief should change, given the
    day's thoughts. The prompt requires one line per belief before the
    verdict so that each belief is considered individually.

    Returns a dict with update_type ("none", "revision" or "new_belief"),
    target_belief_index, updated_text, dimension and reasoning (the
    per-belief lines, for logging).
    """
    thoughts_text = "\n".join([f"- {n.description}" for n in thoughts])

    fallback = {
        "update_type": "none",
        "target_belief_index": None,
        "updated_text": None,
        "dimension": None,
        "reasoning": ""
    }

    if not candidate_beliefs:
        return fallback

    beliefs_text = "\n".join(
        [f"{i+1}. {b.description}" for i, b in enumerate(candidate_beliefs)]
    )

    belief_line_template = "\n".join(
        f"BELIEF_{i+1}: <one line — does it hold, or has it shifted, and how>"
        for i in range(len(candidate_beliefs))
    )

    prompt = f"""You are {agent.name}, a {agent.age}-year-old {agent.gender} from {agent.country} working as {agent.job_type}.

The day has just ended. Now is a time to reflect on all the thoughts you've accumulated today, to see whether any of your core beliefs have shifted.

Today, you had the following thoughts:
{thoughts_text}

Here are some of your current core beliefs that are most relevant to these thoughts:
{beliefs_text}

A revision doesn't have to be a full contradiction. Maybe you now see one of these beliefs a bit differently. Maybe the strength of how much you feel it — more or less than before — has changed. That counts as a real revision, not just a flat reversal.

If you do revise a belief, write a completely fresh statement of what you now believe — a single core idea in one main clause, matching the style of your existing beliefs. Do not just append a qualification to the old wording.

If you are revising or adding a belief, also state which single cultural dimension it belongs to, from this list: PDI (power distance), IDV (individualism vs. collectivism), MAS (achievement vs. care), UAI (uncertainty avoidance), LTO (long-term vs. short-term orientation), IVR (indulgence vs. restraint). Judge this from what the belief is actually about — it does not have to match the dimension of the belief it is replacing.

Respond in this exact format. You must write one line for every belief listed above before giving your verdict — do not skip any, and do not summarize multiple beliefs into one line:

{belief_line_template}
UPDATE_TYPE: none, revision, or new_belief
TARGET_BELIEF_NUMBER: <the number of the belief being revised, or none>
UPDATED_TEXT: <the revised or new belief, freshly written, one clause, or none>
DIMENSION: <PDI, IDV, MAS, UAI, LTO, or IVR — or none if UPDATE_TYPE is none>"""

    response = api_call_with_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    ))

    if response is None:
        print(f"      Content filter triggered in deep reflection for {agent.name}, skipping.")
        return fallback

    if tracker:
        tracker.add(response.usage)

    content = response.choices[0].message.content.strip()

    try:
        # per-belief lines, kept for logging
        belief_lines = BELIEF_LINE_PATTERN.findall(content)
        reasoning = "; ".join(f"#{n}: {text.strip()}" for n, text in belief_lines)

        match = FIELD_PATTERN.search(content)
        if not match:
            raise ValueError("could not locate verdict fields")

        update_type = match.group("update_type").strip().lower()
        if update_type not in ("none", "revision", "new_belief"):
            update_type = "none"

        target_raw = match.group("target").strip().lower()
        target_index = None
        if update_type == "revision":
            if target_raw.isdigit():
                idx = int(target_raw) - 1
                if 0 <= idx < len(candidate_beliefs):
                    target_index = idx
                else:
                    update_type = "none"
            else:
                update_type = "none"

        updated_text = match.group("updated_text").strip()
        if updated_text.lower() == "none":
            updated_text = None

        if update_type in ("revision", "new_belief") and not updated_text:
            update_type = "none"

        dimension = match.group("dimension").strip().upper()
        if dimension not in DIMENSION_CODES:
            dimension = None

        # An update without a valid dimension tag is discarded, since
        # downstream drift scoring relies on the tag.
        if update_type in ("revision", "new_belief") and dimension is None:
            update_type = "none"

        print(f"      Deep reflect [{agent.name}]: {update_type} — {reasoning[:200]}")

        return {
            "update_type": update_type,
            "target_belief_index": target_index,
            "updated_text": updated_text,
            "dimension": dimension,
            "reasoning": reasoning
        }

    except Exception:
        print(f"      Deep reflect [{agent.name}]: response parse failed, defaulting to no update")
        return fallback


def run_deep_reflect(agent, current_day: int, curr_time: str,
                     client, model, tracker=None) -> dict:
    """
    Runs the end-of-day deep reflection for one agent: collects recent
    Tier 2 thoughts, selects candidate Tier 1 beliefs, asks the agent
    whether any should change, and applies the update to CoreBeliefs.

    Args:
        agent: Agent with agent.memory and agent.core.
        current_day: simulation day.
        curr_time: simulation time, e.g. "Day 05, Evening".
        client, model: LLM client and model name.
        tracker: TokenTracker or None.

    Returns:
        A dict for logging with the update outcome, the candidate beliefs
        shown, the old and new belief text, the assigned dimension,
        whether that dimension differs from the superseded belief's tag
        (dimension_mismatch), the reasoning, and the thoughts used.
    """
    result = {
        "updated": False,
        "agent": agent.name,
        "culture": agent.country,
        "day": current_day,
        "thought_count": 0,
        "update_type": "none",
        "candidate_beliefs_shown": [],
        "challenged_belief": None,
        "new_belief": None,
        "dimension": None,
        "dimension_mismatch": False,
        "reasoning": "",
        "rolling_window_thoughts": []
    }

    recent_thoughts = get_recent_thoughts(agent, current_day)
    result["thought_count"] = len(recent_thoughts)
    print(f"      Deep reflect [{agent.name}]: {len(recent_thoughts)} thoughts in rolling window")

    if len(recent_thoughts) < MIN_THOUGHT_COUNT:
        print(f"      Deep reflect [{agent.name}]: below minimum thought count "
              f"({len(recent_thoughts)} < {MIN_THOUGHT_COUNT}), skipping")
        return result

    result["rolling_window_thoughts"] = [n.description for n in recent_thoughts]

    candidate_beliefs = get_candidate_beliefs(agent, recent_thoughts)
    result["candidate_beliefs_shown"] = [
        {"node_id": b.node_id, "description": b.description}
        for b in candidate_beliefs
    ]

    judgment = judge_belief_update(
        agent, recent_thoughts, candidate_beliefs, client, model, tracker=tracker
    )

    result["update_type"] = judgment["update_type"]
    result["reasoning"] = judgment["reasoning"]

    if judgment["update_type"] == "none":
        return result

    challenged_belief = None
    if judgment["update_type"] == "revision" and judgment["target_belief_index"] is not None:
        challenged_belief = candidate_beliefs[judgment["target_belief_index"]]
        result["challenged_belief"] = challenged_belief.description

        # A dimension that differs from the superseded belief's tag is
        # logged but not overridden; the model's stated dimension is kept.
        existing_dimension = getattr(challenged_belief, "dimension", None)
        if existing_dimension is not None and existing_dimension != judgment["dimension"]:
            result["dimension_mismatch"] = True
            print(f"      Deep reflect [{agent.name}]: dimension mismatch — "
                  f"superseded belief tagged {existing_dimension}, "
                  f"model reports {judgment['dimension']}")

    new_belief_text = judgment["updated_text"]
    new_embedding = agent.embedding_fn(new_belief_text)

    agent.core.update_belief(
        new_description=new_belief_text,
        new_embedding=new_embedding,
        curr_time=curr_time,
        dimension=judgment["dimension"],
        supersedes_node_id=challenged_belief.node_id if challenged_belief else None,
        trigger_evidence=[n.description for n in recent_thoughts]
    )

    result["updated"] = True
    result["new_belief"] = new_belief_text
    result["dimension"] = judgment["dimension"]

    print(f"      Deep reflect [{agent.name}]: Tier 1 updated — "
          f"{'revision' if challenged_belief else 'new belief added'}")

    return result