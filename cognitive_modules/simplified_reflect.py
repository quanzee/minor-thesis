"""
File: reflect.py
Description: Simplified reflection module for generative agents,
adapted from Park et al. (2023). Strips Smallville-specific dependencies
while preserving the core reflect architecture.
"""

import datetime
from simplified_retrieve import new_retrieve

REFLECTION_THRESHOLD = 150  # accumulated poignancy threshold, following Park et al.


def generate_focal_points(agent_memory, embedding_fn, client, model, n=3):
    """
    Generates focal points from the agent's most recent and important memories.
    Focal points are the questions that guide reflection retrieval.
    """
    # gather recent events and thoughts, excluding idle nodes
    nodes = agent_memory.seq_event + agent_memory.seq_thought
    nodes = [n for n in nodes if "idle" not in n.description]
    nodes = sorted(nodes, key=lambda x: x.last_accessed)

    # take the most recent N nodes as statements
    recent = nodes[-10:] if len(nodes) >= 10 else nodes
    statements = "\n".join([n.description for n in recent])

    prompt = f"""Here are some recent observations and thoughts:
{statements}

Given these, what are the {n} most important questions we can answer about this person's experiences and feelings? 
Return exactly {n} questions, one per line, no numbering."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    focal_points = [
        line.strip()
        for line in response.choices[0].message.content.strip().split("\n")
        if line.strip()
    ][:n]

    return focal_points


def generate_insights(nodes, agent_name, client, model, n=5):
    """
    Generates high-level insights from a set of retrieved memory nodes.
    Returns a dict mapping insight string to list of source node IDs.
    """
    statements = "\n".join(
        [f"{i}. {node.description}" for i, node in enumerate(nodes)]
    )

    prompt = f"""Here are some memories belonging to {agent_name}:
{statements}

Based on these memories, what are {n} high-level insights or conclusions you can infer about {agent_name}?
For each insight, indicate which memory numbers (0-indexed) support it.

Respond in this exact format, one insight per line:
INSIGHT: <insight text> | EVIDENCE: <comma-separated memory numbers>"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    insights = {}
    for line in response.choices[0].message.content.strip().split("\n"):
        if "INSIGHT:" in line and "EVIDENCE:" in line:
            try:
                insight_part, evidence_part = line.split("|")
                insight_text = insight_part.replace("INSIGHT:", "").strip()
                evidence_nums = [
                    int(x.strip())
                    for x in evidence_part.replace("EVIDENCE:", "").split(",")
                    if x.strip().isdigit()
                ]
                evidence_node_ids = [
                    nodes[i].node_id
                    for i in evidence_nums
                    if i < len(nodes)
                ]
                insights[insight_text] = evidence_node_ids
            except (ValueError, IndexError):
                continue

    return insights


def generate_poignancy(description, client, model):
    """
    Scores a memory's poignancy on a 1-10 scale using the LLM.
    Returns an integer score.
    """
    if "idle" in description:
        return 1

    prompt = f"""On a scale of 1 to 10, how emotionally significant or important is this memory?
Memory: "{description}"
Respond with a single integer only."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return int(response.choices[0].message.content.strip())
    except ValueError:
        return 5


def generate_conversation_reflection(agent_name, conversation_turns,
                                     client, model):
    """
    Generates a planning thought and memo after a conversation ends.
    Returns (planning_thought, memo_thought) as strings.
    """
    dialogue = "\n".join(
        [f"{speaker}: {utterance}"
         for speaker, utterance in conversation_turns]
    )

    planning_prompt = f"""Here is a conversation {agent_name} just had:
{dialogue}

What should {agent_name} keep in mind for future planning based on this conversation?
Respond in one sentence."""

    memo_prompt = f"""Here is a conversation {agent_name} just had:
{dialogue}

What is one key thing {agent_name} will remember about this conversation?
Respond in one sentence."""

    planning_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": planning_prompt}],
        temperature=0.7
    )

    memo_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": memo_prompt}],
        temperature=0.7
    )

    planning_thought = planning_response.choices[0].message.content.strip()
    memo_thought = memo_response.choices[0].message.content.strip()

    return planning_thought, memo_thought


def reflection_trigger(agent, threshold=REFLECTION_THRESHOLD):
    """
    Returns True if accumulated poignancy since last reflection
    exceeds the threshold.
    """
    return agent["importance_since_last_reflect"] >= threshold


def run_reflect(agent, client, model, embedding_fn, curr_time):
    """
    Runs the full reflection cycle for an agent:
    1. Generate focal points from recent memories
    2. Retrieve relevant nodes per focal point
    3. Generate insights and store as thought nodes
    """
    focal_points = generate_focal_points(
        agent["memory"], embedding_fn, client, model, n=3
    )

    retrieved = new_retrieve(
        agent_memory=agent["memory"],
        focal_points=focal_points,
        embedding_fn=embedding_fn,
        n_count=30,
        curr_time=curr_time
    )

    for focal_pt, nodes in retrieved.items():
        if not nodes:
            continue

        insights = generate_insights(
            nodes, agent["name"], client, model, n=5
        )

        for thought_text, evidence_ids in insights.items():
            poignancy = generate_poignancy(thought_text, client, model)
            embedding = embedding_fn(thought_text)

            agent["memory"].add_thought(
                description=thought_text,
                poignancy=poignancy,
                embedding=embedding,
                created=curr_time,
                source_node_ids=evidence_ids
            )

    # reset counter
    agent["importance_since_last_reflect"] = 0


def post_conversation_reflect(agent, conversation_turns,
                               chat_node_id, client, model,
                               embedding_fn, curr_time):
    """
    Runs reflection immediately after a conversation ends.
    Stores planning thought and memo as thought nodes linked to the chat.
    """
    planning_thought, memo_thought = generate_conversation_reflection(
        agent["name"], conversation_turns, client, model
    )

    for thought_text in [planning_thought, memo_thought]:
        poignancy = generate_poignancy(thought_text, client, model)
        embedding = embedding_fn(thought_text)

        agent["memory"].add_thought(
            description=thought_text,
            poignancy=poignancy,
            embedding=embedding,
            created=curr_time,
            source_node_ids=[chat_node_id]
        )


def reflect(agent, client, model, embedding_fn, curr_time):
    """
    Main reflection entry point. Checks trigger condition and
    runs reflection if threshold is met.
    """
    if reflection_trigger(agent):
        run_reflect(agent, client, model, embedding_fn, curr_time)