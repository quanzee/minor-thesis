"""
Reflection module adapted from Park et al. (2023), without the
Smallville-specific dependencies.

Provides periodic reflection, triggered by accumulated poignancy, and
post-conversation reflection. Both store their output as thought nodes in
the agent's AssociativeMemory.

Based on reverie/backend_server/persona/cognitive_modules/reflect.py in the
generative_agents repository (generate_focal_points,
generate_insights_and_evidence, run_reflect, reflection_trigger, reflect),
with prompts adapted from persona/prompt_template/:
    generate_focal_pt_v1.txt        -> generate_focal_points
    insight_and_evidence_v1.txt     -> generate_insights
    planning_thought_on_convo_v1.txt and memo_on_convo_v1.txt
                                    -> generate_conversation_reflection
"""
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
_LIB_DIR = Path(__file__).parent / "lib"

sys.path.insert(0, str(_MODULE_DIR))
sys.path.insert(0, str(_LIB_DIR))

from retrieve import new_retrieve
from config import api_call_with_retry

import re

from simplified_associative_memory import time_key

_NUMBERING = re.compile(r'^\s*(?:[-*•]+\s*)?(?:\d+\s*[\).:]\s*)*')

def _clean_line(line):
    return _NUMBERING.sub('', line).strip().strip('*"').strip()

def generate_focal_points(agent, client, model, curr_time, n=3, tracker=None):
    """
    Generates focal points from the agent's memories since the last
    reflection. Focal points are the questions that guide reflection
    retrieval. Prompt adapted from Park et al. (2023) generate_focal_pt_v1.txt.
    """
    # memories accumulated since the last reflection
    recent = agent.memories_since_last_reflect

    if not recent:
        # fall back to the most recently accessed memories
        recent = sorted(
            agent.memory.seq_event + agent.memory.seq_thought,
            key=lambda x: time_key(x.last_accessed)
        )[-5:]

    statements = "\n".join([n.description for n in recent])

    prompt = f"""Recent observations and experiences:
{statements}

Given the above, what are {n} most salient high-level questions that {agent.name} would reflect on?
1)"""

    response = api_call_with_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    ))

    if response is None:
        print(f"      Content filter triggered for focal points, skipping reflection.")
        return []

    if tracker:
        tracker.add(response.usage)

    lines = [_clean_line(l) for l in response.choices[0].message.content.strip().split("\n")]
    lines = [l for l in lines if l]
    questions = [l for l in lines if "?" in l]
    return (questions or lines)[:n]


def generate_insights(nodes, agent, curr_time, focal_pt, client, model, n=1, tracker=None):
    """
    Generates high-level insights from a set of retrieved memory nodes.
    Returns a dict mapping insight string to list of source node IDs.
    Prompt adapted from Park et al. (2023) insight_and_evidence_v1.txt.
    """

    statements = "\n".join(
        [f"{i}. {node.description}" for i, node in enumerate(nodes)]
    )

    prompt = f"""Based on the following observations and experiences:
{statements}

What {n} high-level insight would {agent.name} draw from the above? (example format: insight (because of 1, 5, 3))
Respond with exactly {n} line. Each line must contain exactly one insight and exactly one (because of ...) citation.
Write each insight as a plain declarative sentence in third person with no numbering, bullets, or markdown formatting.
1."""

    response = api_call_with_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    ))

    if response is None:
        print(f"      Content filter triggered for insights, skipping.")
        return {}

    if tracker:
        tracker.add(response.usage)

    insights = {}
    raw = response.choices[0].message.content.strip()

    for line in raw.split("\n"):
        content = _clean_line(line)
        if not content:
            continue

        pattern = re.compile(r'(.*?)\bbecause of\s+([\d\s,]+)', re.IGNORECASE)
        matches = list(pattern.finditer(content))

        if matches:
            for m in matches:
                insight_text = re.sub(r'^[\s.,)]+', '', m.group(1))
                insight_text = re.sub(r'[\s(]+$', '', insight_text).strip()
                evidence_nums = [int(x) for x in re.findall(r'\d+', m.group(2))]
                evidence_node_ids = [
                    nodes[i].node_id
                    for i in evidence_nums
                    if i < len(nodes)
                ]
                if insight_text:
                    insights[insight_text] = evidence_node_ids

            remainder = content[matches[-1].end():]
            remainder = re.sub(r'^[\s.,)]+', '', remainder).strip()
            if remainder:
                insights[remainder] = []
        else:
            insight_text = content.strip()
            if insight_text:
                insights[insight_text] = []

    return insights


def generate_conversation_reflection(agent, conversation_turns, curr_time,
                                     client, model, tracker=None):
    """
    Generates a single post-conversation thought, merging Park et al.'s
    (2023) two post-conversation prompts (planning_thought_on_convo_v1.txt
    and memo_on_convo_v1.txt) into one. Returns the thought as a string.
    """
    dialogue = "\n".join(
        [f"{speaker}: {utterance}"
         for speaker, utterance in conversation_turns]
    )

    other_name = next(
        speaker for speaker, _ in conversation_turns
        if speaker != agent.name
    )

    planning_prompt = f"""[Conversation]
{dialogue}

Write one sentence about what {agent.name} found notable, wants to remember, or would act on after this conversation.

"""

    planning_response = api_call_with_retry(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": planning_prompt}],
        temperature=0.7
    ))

    if planning_response is None:
        print(f"      Content filter triggered for conversation reflection, skipping.")
        return None

    if tracker:
        tracker.add(planning_response.usage)

    raw_thought = planning_response.choices[0].message.content.strip().strip('"')

    # avoid repeating the name if the response already starts with it
    if raw_thought.lower().startswith(agent.name.lower()):
        planning_thought = raw_thought
    else:
        planning_thought = f"{agent.name} {raw_thought}"

    return planning_thought


def reflection_trigger(agent):
    """
    Returns True if accumulated poignancy since last reflection
    exceeds the threshold.
    """
    return agent.importance_since_last_reflect >= agent.reflection_threshold


def run_reflect(agent, client, model, curr_time, embedding_fn, tracker=None):
    """
    Runs the full reflection cycle for an agent:
    1. Generate focal points from recent memories
    2. Retrieve relevant nodes per focal point
    3. Generate insights and store as thought nodes
    """
    focal_points = generate_focal_points(
        agent, client, model, curr_time, n=1, tracker=tracker
    )

    retrieved = new_retrieve(
        agent_memory=agent.memory,
        focal_points=focal_points,
        embedding_fn=embedding_fn,
        n_count=30,
        curr_time=curr_time
    )

    all_insights = []

    for focal_pt, nodes in retrieved.items():
        if not nodes:
            continue

        insights = generate_insights(
            nodes, agent, curr_time, focal_pt, client, model, n=3, tracker=tracker
        )

        for thought_text, evidence_ids in insights.items():
            poignancy = agent._score_poignancy(thought_text, memory_type="thought", tracker=tracker)
            embedding = embedding_fn(thought_text)

            agent.memory.add_thought(
                description=thought_text,
                poignancy=poignancy,
                embedding=embedding,
                created=curr_time,
                source_node_ids=evidence_ids,
                reflection_type="periodic"
            )
            all_insights.append(thought_text)

    agent.importance_since_last_reflect = 0
    agent.memories_since_last_reflect = []

    return focal_points, all_insights


def post_conversation_reflect(agent, conversation_turns,
                               chat_node_id, client, model,
                               embedding_fn, curr_time, tracker=None):
    """
    Runs reflection immediately after a conversation ends.
    Stores planning thought as a thought node linked to the chat.
    """
    planning_thought = generate_conversation_reflection(
        agent, conversation_turns, curr_time, client, model, tracker=tracker
    )

    if planning_thought:
        poignancy = agent._score_poignancy(planning_thought, memory_type="thought", tracker=tracker)
        embedding = embedding_fn(planning_thought)
        agent.memory.add_thought(
            description=planning_thought,
            poignancy=poignancy,
            embedding=embedding,
            created=curr_time,
            source_node_ids=[chat_node_id],
            reflection_type="post_conversation"
        )


def reflect(agent, client, model, embedding_fn, curr_time, tracker=None):
    """
    Runs periodic reflection if the trigger threshold has been reached.
    Returns (reflected, focal_points, insights).
    """
    if reflection_trigger(agent):
        focal_points, all_insights = run_reflect(agent, client, model, curr_time, embedding_fn, tracker=tracker)
        return True, focal_points, all_insights
    return False, [], []