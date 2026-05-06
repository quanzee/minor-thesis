"""
File: reflect.py
Description: Simplified reflection module for generative agents,
adapted from Park et al. (2023). Strips Smallville-specific dependencies
while preserving the core reflect architecture.
"""

from cognitive_modules.simplified_retrieve import new_retrieve

def generate_focal_points(agent, client, model, n=3, tracker=None):
    """
    Generates focal points from the agent's most recent and important memories.
    Focal points are the questions that guide reflection retrieval.
    """
    # take the nodes accummulated since the last reflection as statements
    recent = agent.memories_since_last_reflect

    if not recent:
    # fallback if no new memories since last reflection
        recent = sorted(
            agent.memory.seq_event + agent.memory.seq_thought,
            key=lambda x: x.last_accessed
        )[-5:]
    
    statements = "\n".join([n.description for n in recent])

    prompt = f"""{statements}
Given only the information above, what are {n} most salient high-level questions we can answer about the subjects grounded in the statements?
1)"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    if tracker:
        tracker.add(response.usage)

    focal_points = []
    for line in response.choices[0].message.content.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # remove the numbering e.g. "1) " or "1. "
            cleaned = line.split(")", 1)[-1].split(".", 1)[-1].strip()
            if cleaned:
                focal_points.append(cleaned)

    return focal_points[:n]


def generate_insights(nodes, agent_name, client, model, n=5, tracker=None):
    """
    Generates high-level insights from a set of retrieved memory nodes.
    Returns a dict mapping insight string to list of source node IDs.
    Prompt adapted from Park et al. (2023) insight_and_evidence_v1.txt.
    """
    statements = "\n".join(
        [f"{i}. {node.description}" for i, node in enumerate(nodes)]
    )

    prompt = f"""{statements}
What {n} high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))
1."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    if tracker:
        tracker.add(response.usage)

    insights = {}
    raw = "1." + response.choices[0].message.content.strip()

    for line in raw.split("\n"):
        line = line.strip()
        if not line or not line[0].isdigit():
            continue

        # remove numbering
        content = line.split(".", 1)[-1].strip()
        if not content:
            continue

        # split insight text from evidence e.g. "insight (because of 1, 5, 3)"
        if "(because of" in content:
            insight_text, evidence_raw = content.rsplit("(because of", 1)
            insight_text = insight_text.strip()
            evidence_raw = evidence_raw.replace(")", "").strip()
            evidence_nums = [
                int(x.strip())
                for x in evidence_raw.split(",")
                if x.strip().isdigit()
            ]
            evidence_node_ids = [
                nodes[i].node_id
                for i in evidence_nums
                if i < len(nodes)
            ]
        else:
            insight_text = content.strip()
            evidence_node_ids = []

        if insight_text:
            insights[insight_text] = evidence_node_ids

    return insights

def generate_conversation_reflection(agent_name, conversation_turns,
                                     client, model, tracker=None):
    """
    Generates a planning thought and memo after a conversation ends.
    Returns (planning_thought, memo_thought) as strings.
    Prompts adapted from Park et al. (2023):
    - planning_thought_on_convo_v1.txt
    - convo_to_thoughts_v1.txt
    """
    dialogue = "\n".join(
        [f"{speaker}: {utterance}"
         for speaker, utterance in conversation_turns]
    )

    # identify the other agent in the conversation
    other_name = next(
        speaker for speaker, _ in conversation_turns
        if speaker != agent_name
    )

    planning_prompt = f"""[Conversation]
{dialogue}

Write down if there is anything from the conversation that {agent_name} needs to remember for their planning, from {agent_name}'s perspective, in a full sentence.

\"{agent_name}"""

    memo_prompt = f"""Here is the conversation that happened between {agent_name} and {other_name}.

{dialogue}

Summarize what {agent_name} thought about {other_name} in one short sentence. The sentence needs to be in third person:"""

    planning_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": planning_prompt}],
        temperature=0.7
    )
    if tracker:
        tracker.add(planning_response.usage)

    memo_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": memo_prompt}],
        temperature=0.7
    )
    if tracker:
        tracker.add(memo_response.usage)

    planning_thought = (
        f"{agent_name} " +
        planning_response.choices[0].message.content.strip().strip('"')
    )
    memo_thought = memo_response.choices[0].message.content.strip()

    return planning_thought, memo_thought


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
        agent, client, model, n=3, tracker=tracker
    )

    retrieved = new_retrieve(
        agent_memory=agent.memory,
        focal_points=focal_points,
        embedding_fn=embedding_fn,
        n_count=30,
        curr_time=curr_time
    )

    for focal_pt, nodes in retrieved.items():
        if not nodes:
            continue

        insights = generate_insights(
            nodes, agent.name, client, model, n=5, tracker=tracker
        )

        for thought_text, evidence_ids in insights.items():
            poignancy = agent._score_poignancy(thought_text, memory_type="thought")
            embedding = embedding_fn(thought_text)

            agent.memory.add_thought(
                description=thought_text,
                poignancy=poignancy,
                embedding=embedding,
                created=curr_time,
                source_node_ids=evidence_ids
            )

    # reset counter
    agent.importance_since_last_reflect = 0
    agent.memories_since_last_reflect = []


def post_conversation_reflect(agent, conversation_turns,
                               chat_node_id, client, model,
                               embedding_fn, curr_time, tracker=None):
    """
    Runs reflection immediately after a conversation ends.
    Stores planning thought and memo as thought nodes linked to the chat.
    """
    planning_thought, memo_thought = generate_conversation_reflection(
        agent.name, conversation_turns, client, model
    )

    for thought_text in [planning_thought, memo_thought]:
        poignancy = agent._score_poignancy(thought_text, memory_type="thought", tracker=tracker)
        embedding = embedding_fn(thought_text)

        agent.memory.add_thought(
            description=thought_text,
            poignancy=poignancy,
            embedding=embedding,
            created=curr_time,
            source_node_ids=[chat_node_id]
        )


def reflect(agent, client, model, embedding_fn, curr_time, tracker=None):
    """
    Main reflection entry point. Checks trigger condition and
    runs reflection if threshold is met.
    """
    if reflection_trigger(agent):
        run_reflect(agent, client, model, curr_time, embedding_fn, tracker=tracker)