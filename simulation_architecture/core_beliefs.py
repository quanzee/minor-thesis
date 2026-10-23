"""
Tier 1 core belief memory for two_tier agents.

Stores the Hofstede-derived persona seeds as core beliefs, retrieved by
semantic relevance only (no recency or importance scoring). Tier 2 episodic
memory (chats and thoughts) is held separately in AssociativeMemory.

Revision through end-of-day deep reflection (update_belief) is supported.
In this study the two-tier condition (C2) uses a fixed core
(--frozen_core), so beliefs are seeded once and never revised.
"""

import json
from numpy import dot
from numpy.linalg import norm
from dataclasses import dataclass
from typing import Optional


@dataclass
class CoreBeliefNode:
    node_id: str
    description: str                    # natural language belief string
    embedding: list                     # semantic vector for relevance retrieval
    created: str                        # simulation time when seeded
    dimension: Optional[str] = None     # Hofstede code: PDI/IDV/MAS/UAI/LTO/IVR
    is_active: bool = True              # False if superseded by an update
    superseded_by: Optional[str] = None        # node_id of the replacing node
    superseded_at: Optional[str] = None        # simulation time of supersession


class CoreBeliefs:
    """
    Tier 1 memory of core cultural beliefs.

    Beliefs are seeded from the persona YAML, retrieved by cosine
    similarity and, when the core is not frozen, updated through deep
    reflection. Superseded
    beliefs are kept for drift analysis but excluded from retrieval, and
    every update is recorded in update_log.
    """

    def __init__(self):
        self.id_to_node = dict()        # node_id -> CoreBeliefNode
        self.beliefs = []               # all nodes in insertion order
        self.node_count = 0
        self.agent_name = None
        self.update_log = []            # audit trail of all Tier 1 updates

    # ID generation
    def _generate_node_id(self):
        self.node_count += 1
        return f"core_{self.node_count}"

    # Seeding
    def seed_beliefs(self, seed_descriptions: list, embedding_fn,
                     created: str = "Day 00, Initialisation"):
        """
        Loads persona seeds into Tier 1, one CoreBeliefNode per seed.

        Args:
            seed_descriptions: list of dicts with "description" and
                "dimension" (PDI, IDV, MAS, UAI, LTO or IVR).
            embedding_fn: callable mapping text to an embedding vector.
            created: simulation timestamp.
        """
        for seed in seed_descriptions:
            description = seed["description"]
            dimension = seed.get("dimension")
            embedding = embedding_fn(description)
            node_id = self._generate_node_id()
            node = CoreBeliefNode(
                node_id=node_id,
                description=description,
                embedding=embedding,
                created=created,
                dimension=dimension,
                is_active=True
            )
            self.beliefs.append(node)
            self.id_to_node[node_id] = node

    # Retrieval
    def retrieve(self, query_embedding: list, top_n: int = 5) -> list:
        """
        Returns up to top_n active beliefs, sorted by cosine similarity to
        query_embedding.
        """
        active_nodes = [n for n in self.beliefs if n.is_active]
        if not active_nodes:
            return []

        scored = []
        for node in active_nodes:
            sim = self._cosine_similarity(query_embedding, node.embedding)
            scored.append((sim, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_n]]

    @staticmethod
    def _cosine_similarity(vec_a: list, vec_b: list) -> float:
        """
        Computes the cosine similarity of two vectors.
        """
        return dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

    # Update mechanism
    def update_belief(self, new_description: str, new_embedding: list,
                      curr_time: str, dimension: Optional[str] = None,
                      supersedes_node_id: Optional[str] = None,
                      trigger_evidence: Optional[list] = None):
        """
        Adds a core belief, optionally superseding an existing one. Called
        by the end-of-day deep reflection, so not used when the core is
        frozen (C2). A superseded node is marked inactive but retained for
        drift analysis.

        Args:
            new_description: text of the new belief.
            new_embedding: embedding of the new belief.
            curr_time: simulation timestamp of the update.
            dimension: Hofstede code assigned to the new belief. It is set
                by the caller rather than inherited from the superseded
                belief.
            supersedes_node_id: node_id of the belief being replaced, or
                None for a new belief.
            trigger_evidence: Tier 2 descriptions that prompted the update.

        Returns:
            The new CoreBeliefNode.
        """
        # mark old node as superseded
        old_description = None
        old_dimension = None
        if supersedes_node_id and supersedes_node_id in self.id_to_node:
            old_node = self.id_to_node[supersedes_node_id]
            old_node.is_active = False
            old_node.superseded_by = self._peek_next_id()
            old_node.superseded_at = curr_time
            old_description = old_node.description
            old_dimension = old_node.dimension

        # create new node
        node_id = self._generate_node_id()
        node = CoreBeliefNode(
            node_id=node_id,
            description=new_description,
            embedding=new_embedding,
            created=curr_time,
            dimension=dimension,
            is_active=True
        )
        self.beliefs.append(node)
        self.id_to_node[node_id] = node

        # log the update event
        self.update_log.append({
            "curr_time": curr_time,
            "event": "supersession" if supersedes_node_id else "new_belief",
            "new_node_id": node_id,
            "new_description": new_description,
            "new_dimension": dimension,
            "superseded_node_id": supersedes_node_id,
            "old_description": old_description,
            "old_dimension": old_dimension,
            "trigger_evidence": trigger_evidence or []
        })

        return node

    def _peek_next_id(self) -> str:
        """Returns what the next node_id will be without incrementing."""
        return f"core_{self.node_count + 1}"

    # Inspection helpers
    def get_active_beliefs(self) -> list:
        """Returns all currently active CoreBeliefNodes."""
        return [n for n in self.beliefs if n.is_active]

    def get_all_beliefs(self) -> list:
        """Returns all nodes including superseded ones. Used for drift analysis."""
        return self.beliefs

    def get_update_log(self) -> list:
        """Returns the full audit trail of Tier 1 update events."""
        return self.update_log

    # Persistence
    def save(self, filepath: str):
        """
        Saves CoreBeliefs to a JSON file.
        Includes both active and superseded nodes and the full update log.
        """
        data = {
            "agent_name": self.agent_name,
            "node_count": self.node_count,
            "update_log": self.update_log,
            "nodes": {}
        }
        for node in self.beliefs:
            data["nodes"][node.node_id] = {
                "description": node.description,
                "embedding": node.embedding if isinstance(node.embedding, list)
                             else node.embedding.tolist(),
                "created": node.created,
                "dimension": node.dimension,
                "is_active": node.is_active,
                "superseded_by": node.superseded_by,
                "superseded_at": node.superseded_at
            }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str):
        """
        Loads a CoreBeliefs instance from a JSON file.
        Reconstructs all nodes including superseded ones.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        core = cls()
        core.agent_name = data.get("agent_name")
        core.node_count = data["node_count"]
        core.update_log = data.get("update_log", [])

        for node_id, nd in data["nodes"].items():
            node = CoreBeliefNode(
                node_id=node_id,
                description=nd["description"],
                embedding=nd["embedding"],
                created=nd["created"],
                dimension=nd.get("dimension"),
                is_active=nd["is_active"],
                superseded_by=nd.get("superseded_by"),
                superseded_at=nd.get("superseded_at")
            )
            core.beliefs.append(node)
            core.id_to_node[node_id] = node

        return core