"""
Memory retrieval adapted from Park et al. (2023), scoring Tier 2
(AssociativeMemory) nodes by recency, importance and relevance.

Based on reverie/backend_server/persona/cognitive_modules/retrieve.py in the
generative_agents repository (new_retrieve, extract_recency,
extract_importance, extract_relevance, normalize_dict_floats,
top_highest_x_values, cos_sim). The weights [0.5, 3, 2] are taken from
new_retrieve and the recency decay of 0.99 from
persona/memory_structures/scratch.py.

Each retrieval is logged to logs/tier2_retrieval.jsonl with the agent's
memory_mode.
"""

from numpy import dot
from numpy.linalg import norm
import json
from pathlib import Path
_BASE_DIR = Path(__file__).parent.parent
_LOGS_DIR = _BASE_DIR / "logs"
RETRIEVAL_LOG = _LOGS_DIR / "tier2_retrieval.jsonl"

def cos_sim(a, b):
    """
    Computes cosine similarity between two vectors.
    Returns a scalar between -1 and 1.
    """
    return dot(a, b) / (norm(a) * norm(b))


def normalize_dict_floats(d, target_min, target_max):
    """
    Normalises the float values of a dictionary to a target range
    using min-max scaling.
    """
    if not d:
        return d

    min_val = min(d.values())
    max_val = max(d.values())
    range_val = max_val - min_val

    if range_val == 0:
        for key in d:
            d[key] = (target_max - target_min) / 2
    else:
        for key in d:
            d[key] = ((d[key] - min_val) * (target_max - target_min)
                      / range_val + target_min)
    return d


def top_highest_x_values(d, x):
    """
    Returns a new dictionary with the top x key-value pairs
    by highest value.
    """
    return dict(sorted(d.items(),
                       key=lambda item: item[1],
                       reverse=True)[:x])


def extract_recency(nodes, decay=0.99):
    """
    Computes recency scores by exponential decay over node position.
    Nodes are sorted by last_accessed in ascending order, so the most
    recently accessed node scores decay**1. The default decay of 0.99
    follows Park et al. (2023).

    Returns a dict mapping node_id to recency score.
    """
    recency_out = {}
    for i, node in enumerate(nodes):
        position_from_end = len(nodes) - i
        recency_out[node.node_id] = decay ** position_from_end
    return recency_out


def extract_importance(nodes):
    """
    Returns a dict mapping node_id to poignancy (1-10, assigned when the
    memory is created).
    """
    importance_out = {}
    for node in nodes:
        importance_out[node.node_id] = node.poignancy
    return importance_out


def extract_relevance(nodes, focal_pt_embedding):
    """
    Returns a dict mapping node_id to the cosine similarity between the
    node's embedding and the focal point embedding.
    """
    relevance_out = {}
    for node in nodes:
        relevance_out[node.node_id] = cos_sim(node.embedding,
                                               focal_pt_embedding)
    return relevance_out


def new_retrieve(agent_memory, focal_points, embedding_fn,
                 n_count=30,
                 gw=None, curr_time=None, node_types=None, update_access=True):
    """
    Retrieves the top n_count most relevant memory nodes for each
    focal point using a weighted combination of recency, importance,
    and relevance scores, following Park et al. (2023).

    Args:
        agent_memory: AssociativeMemory object.
        focal_points: list of focal point strings.
        embedding_fn: callable mapping text to an embedding vector.
        n_count: number of memories to retrieve per focal point.
        gw: weights [recency, relevance, importance]; defaults to
            [0.5, 3, 2].
        curr_time: simulation time used to update last_accessed.
        node_types: optional list of node types to restrict retrieval to.
        update_access: whether to update last_accessed on retrieved nodes.

    Returns:
        A dict mapping each focal point to a list of MemoryNode objects.
    """
    if gw is None:
        gw = [0.5, 3, 2]

    if curr_time is None:
        curr_time = "Day 00, Initialisation"

    retrieved = {}

    for focal_pt in focal_points:
        # nodes sorted by last_accessed, oldest first
        nodes = agent_memory.get_all_nodes()

        if node_types is not None:
            nodes = [n for n in nodes if n.node_type in node_types]

        if not nodes:
            retrieved[focal_pt] = []
            continue

        recency_out = extract_recency(nodes)
        recency_out = normalize_dict_floats(recency_out, 0, 1)

        importance_out = extract_importance(nodes)
        importance_out = normalize_dict_floats(importance_out, 0, 1)

        focal_embedding = embedding_fn(focal_pt)
        relevance_out = extract_relevance(nodes, focal_embedding)
        relevance_out = normalize_dict_floats(relevance_out, 0, 1)

        master_out = {}
        for node in nodes:
            key = node.node_id
            master_out[key] = (recency_out[key] * gw[0]
                               + relevance_out[key] * gw[1]
                               + importance_out[key] * gw[2])

        master_out = top_highest_x_values(master_out, n_count)
        master_nodes = [agent_memory.id_to_node[key]
                        for key in master_out.keys()]

        # Survey retrieval passes update_access=False so that it does not
        # alter memory.
        if update_access:
            for node in master_nodes:
                node.last_accessed = curr_time

        log_entry = {
            "agent": agent_memory.agent_name if hasattr(agent_memory, 'agent_name') else "unknown",
            "memory_mode": getattr(agent_memory, "memory_mode", "unknown"),
            "focal_point": focal_pt,
            "curr_time": curr_time,
            "retrieved": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "description": node.description[:100],
                    "poignancy": node.poignancy,
                    "recency": round(recency_out[node.node_id], 4),
                    "importance": round(importance_out[node.node_id], 4),
                    "relevance": round(relevance_out[node.node_id], 4),
                    "master_score": round(master_out[node.node_id], 4)
                }
                for node in master_nodes
            ]
        }
        _LOGS_DIR.mkdir(exist_ok=True)
        with open(RETRIEVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        retrieved[focal_pt] = master_nodes

    return retrieved

import time

def get_embedding(text, client, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small-1"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < retries - 1:
                print(f"      Embedding call failed (attempt {attempt+1}), retrying in {delay}s: {e}")
                time.sleep(delay)
            else:
                raise

def retrieve_current_beliefs(agent, embedding_fn, curr_time, context=None):
    """
    Retrieves the agent's most salient beliefs from Tier 2 thought nodes.
    Early in the simulation these are the persona seeds; later they
    include thoughts formed through interaction and reflection.

    Used for single_tier agents and for the Tier 2 component of
    frozen_core agents. Returns (belief_text, retrieved_node_ids).
    """
    if context:
        focal_point = ["personal values and beliefs related to " + ", ".join(context)]
    else:
        focal_point = ["personal values and beliefs"]

    belief_retrieved = new_retrieve(
        agent_memory=agent.memory,
        focal_points=focal_point,
        embedding_fn=embedding_fn,
        n_count=3,
        curr_time=curr_time,
        node_types=["thought"]
    )

    seen_ids = set()
    belief_lines = []
    for nodes in belief_retrieved.values():
        for node in nodes[:3]:
            if node.node_id not in seen_ids:
                belief_lines.append(f"- {node.description}")
                seen_ids.add(node.node_id)

    return "\n".join(belief_lines), seen_ids