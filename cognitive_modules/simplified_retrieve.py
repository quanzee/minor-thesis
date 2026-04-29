"""
File: retrieve.py
Description: Simplified retrieval module for generative agents,
adapted from Park et al. (2023). Implements the new_retrieve function
using recency, importance, and relevance scoring.
"""

import datetime
from numpy import dot
from numpy.linalg import norm


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
    Computes recency scores for each node using exponential decay
    based on position in the chronologically sorted node list.
    
    Nodes are assumed to be sorted by last_accessed in ascending order
    (oldest first), so the most recently accessed node gets decay^1
    and older nodes get progressively smaller scores.

    INPUT:
        nodes: list of MemoryNode objects sorted by last_accessed ascending
        decay: decay factor, default 0.99 following Park et al.
    OUTPUT:
        recency_out: dict mapping node_id to recency score
    """
    recency_out = {}
    for i, node in enumerate(nodes):
        # nodes[-1] is most recent, nodes[0] is oldest
        # position from the end determines recency
        position_from_end = len(nodes) - i
        recency_out[node.node_id] = decay ** position_from_end
    return recency_out


def extract_importance(nodes):
    """
    Extracts importance scores from each node's poignancy value.
    Poignancy is assigned when the memory is created (1-10 scale).

    INPUT:
        nodes: list of MemoryNode objects
    OUTPUT:
        importance_out: dict mapping node_id to poignancy score
    """
    importance_out = {}
    for node in nodes:
        importance_out[node.node_id] = node.poignancy
    return importance_out


def extract_relevance(nodes, focal_pt_embedding):
    """
    Computes relevance scores as cosine similarity between each
    node's embedding and the focal point embedding.

    INPUT:
        nodes: list of MemoryNode objects
        focal_pt_embedding: embedding vector of the focal point string
    OUTPUT:
        relevance_out: dict mapping node_id to cosine similarity score
    """
    relevance_out = {}
    for node in nodes:
        relevance_out[node.node_id] = cos_sim(node.embedding,
                                               focal_pt_embedding)
    return relevance_out


def new_retrieve(agent_memory, focal_points, embedding_fn,
                 n_count=30,
                 gw=None, curr_time=None):
    """
    Retrieves the top n_count most relevant memory nodes for each
    focal point using a weighted combination of recency, importance,
    and relevance scores, following Park et al. (2023).

    INPUT:
        agent_memory: AssociativeMemory object
        focal_points: list of strings describing current focus
        embedding_fn: callable that takes a string and returns an embedding
        n_count: number of top memories to retrieve per focal point
        gw: global weights list [recency_gw, relevance_gw, importance_gw]
            defaults to [0.5, 3, 2] following Park et al.
        curr_time: current simulation datetime for updating last_accessed
    OUTPUT:
        retrieved: dict mapping focal point string to list of MemoryNode objects
    """
    if gw is None:
        gw = [1, 1, 1]

    if curr_time is None:
        curr_time = datetime.datetime.now()

    retrieved = {} #initialise output dictionary

    for focal_pt in focal_points:
        # Get all non-idle nodes sorted by last_accessed ascending
        nodes = agent_memory.get_all_nodes()

        if not nodes:
            retrieved[focal_pt] = []
            continue

        # Compute and normalise the three scoring components
        recency_out = extract_recency(nodes)
        recency_out = normalize_dict_floats(recency_out, 0, 1)

        importance_out = extract_importance(nodes)
        importance_out = normalize_dict_floats(importance_out, 0, 1)

        focal_embedding = embedding_fn(focal_pt)
        relevance_out = extract_relevance(nodes, focal_embedding)
        relevance_out = normalize_dict_floats(relevance_out, 0, 1)

        # Compute weighted master score for each node
        master_out = {}
        for node in nodes:
            key = node.node_id
            master_out[key] = (recency_out[key] * gw[0]
                               + relevance_out[key] * gw[1]
                               + importance_out[key] * gw[2])

        # Extract top n_count nodes
        master_out = top_highest_x_values(master_out, n_count)
        master_nodes = [agent_memory.id_to_node[key]
                        for key in master_out.keys()]

        # Update last_accessed timestamp for retrieved nodes
        for node in master_nodes:
            node.last_accessed = curr_time

        retrieved[focal_pt] = master_nodes

    return retrieved