"""
File: reflect.py
Description: Simplified reflection module for generative agents,
adapted from Park et al. (2023). Strips Smallville-specific dependencies
while preserving the core reflect architecture.
"""

from cognitive_modules.simplified_retrieve import new_retrieve

def generate_focal_points(agent, client, model, curr_time, n=3, tracker=None):
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

    # retrieve current beliefs relevant to recent memories for persona context
    persona_context, belief_ids = agent.get_persona_context(
        agent.embedding_fn, curr_time,
        context=None
    )

    statements = "\n".join([
    n.description for n in recent 
    if n.node_id not in belief_ids
    ])

    prompt = f"""{persona_context}

Recent observations and experiences:
{statements}

Given the above, what are {n} most salient high-level questions that {agent.name} would reflect on, consistent with their background and values?
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


def generate_insights(nodes, agent, curr_time, focal_pt, client, model, n=3, tracker=None):
    """
    Generates high-level insights from a set of retrieved memory nodes.
    Returns a dict mapping insight string to list of source node IDs.
    Prompt adapted from Park et al. (2023) insight_and_evidence_v1.txt.
    """
    persona_context, belief_ids = agent.get_persona_context(
    agent.embedding_fn, curr_time,
    context=[focal_pt]
    )

    # rebuild statements excluding belief nodes
    filtered_nodes = [n for n in nodes if n.node_id not in belief_ids]
    statements = "\n".join(
        [f"{i}. {node.description}" for i, node in enumerate(filtered_nodes)]
    )

    prompt = f"""{persona_context}

Based on the following observations and experiences:
{statements}

What {n} high-level insights would {agent.name} draw from the above, consistent with their values and cultural background? (example format: insight (because of 1, 5, 3))
Write each insight as a plain declarative sentence in third person with no numbering, bullets, or markdown formatting.
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
                filtered_nodes[i].node_id
                for i in evidence_nums
                if i < len(filtered_nodes)
            ]
        else:
            insight_text = content.strip()
            evidence_node_ids = []

        if insight_text:
            insights[insight_text] = evidence_node_ids

    return insights

def generate_conversation_reflection(agent, conversation_turns, curr_time,
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
        if speaker != agent.name
    )

    # retrieve current beliefs relevant to the conversation
    persona_context, _ = agent.get_persona_context(
        agent.embedding_fn, curr_time,
        context=[f"conversation with {other_name}"]
    )

    planning_prompt = f"""{persona_context}

[Conversation]
{dialogue}

Write down if there is anything from the conversation that {agent.name} needs to remember for their planning, consistent with their values and cultural background, in a full sentence.

\"{agent.name}"""

    memo_prompt = f"""{persona_context}
    
Here is the conversation that happened between {agent.name} and {other_name}.

{dialogue}

Summarize what {agent.name} genuinely thought about {other_name} and the conversation, consistent with {agent.name}'s cultural background. One sentence, third person:"""

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
        f"{agent.name} " +
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
        agent, client, model, curr_time, n=3, tracker=tracker
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
            poignancy = agent._score_poignancy(thought_text, memory_type="thought")
            embedding = embedding_fn(thought_text)

            agent.memory.add_thought(
                description=thought_text,
                poignancy=poignancy,
                embedding=embedding,
                created=curr_time,
                source_node_ids=evidence_ids
            )
            all_insights.append(thought_text)

    # reset counter
    agent.importance_since_last_reflect = 0
    agent.memories_since_last_reflect = []

    return focal_points, all_insights


def post_conversation_reflect(agent, conversation_turns,
                               chat_node_id, client, model,
                               embedding_fn, curr_time, tracker=None):
    """
    Runs reflection immediately after a conversation ends.
    Stores planning thought and memo as thought nodes linked to the chat.
    """
    planning_thought, memo_thought = generate_conversation_reflection(
        agent, conversation_turns, curr_time, client, model, tracker=tracker
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
        focal_points, all_insights = run_reflect(agent, client, model, curr_time, embedding_fn, tracker=tracker)
        return True, focal_points, all_insights
    return False, [], []

