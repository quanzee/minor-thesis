"""
Long-term memory stream adapted from Park et al. (2023), without the
Smallville-specific dependencies.

Used as Tier 2 (AssociativeMemory) in both memory modes. single_tier agents
are seeded here through seed_memories(); two_tier agents are seeded in
CoreBeliefs instead.

Based on reverie/backend_server/persona/memory_structures/associative_memory.py
in the generative_agents repository (ConceptNode and AssociativeMemory).
seed_memories() follows load_history_via_whisper in
persona/cognitive_modules/converse.py.
"""

import json

# Main study rounds and side study timepoints, in within-day order
_ROUND_ORDER = {"Initialisation": 0, "Morning": 1, "Afternoon": 2, "Evening": 3,
                "Timepoint 1": 1, "Timepoint 2": 2, "Timepoint 3": 3, "Timepoint 4": 4}

def time_key(t):
    day_part, _, round_part = t.partition(",")
    return (int(day_part.replace("Day", "").strip()), _ROUND_ORDER.get(round_part.strip(), 0))

class MemoryNode:
    def __init__(self, node_id, node_type, description,
                 poignancy, embedding, created, depth=0, filling=None, reflection_type = None):
        self.node_id = node_id
        self.node_type = node_type      # "event", "thought" or "chat"
        self.description = description
        self.poignancy = poignancy      # importance score, 1-10
        self.embedding = embedding
        self.created = created          # simulation time, e.g. "Day 01, Morning"
        self.last_accessed = created    # simulation time, used for recency
        self.depth = depth              # 0 for events/chats, 1+ for thoughts
        self.filling = filling or []    # dialogue turns (chats) or source node IDs (thoughts)
        self.reflection_type = reflection_type  # "seed", "periodic", "post_conversation" or None

class AssociativeMemory:
    def __init__(self, memory_mode=None):
        self.memory_mode = memory_mode  # "single_tier" or "two_tier"
        self.id_to_node = dict()        # node_id -> MemoryNode
        self.seq_event = []             # events, newest first (not populated in this study)
        self.seq_thought = []           # thoughts, newest first
        self.seq_chat = []              # chats, newest first
        self.node_count = 0
        self.agent_name = None

    def _generate_node_id(self):
        self.node_count += 1
        return f"node_{self.node_count}"

    def add_event(self, description, poignancy, embedding, created):
        """
        Adds an event memory to the memory stream. Retained from Park et
        al.'s three node types; the single-location design of this study
        does not generate event observations.
        """
        node_id = self._generate_node_id()
        node = MemoryNode(
            node_id=node_id,
            node_type="event",
            description=description,
            poignancy=poignancy,
            embedding=embedding,
            created=created,
            depth=0
        )
        self.seq_event.insert(0, node)
        self.id_to_node[node_id] = node
        return node

    def add_thought(self, description, poignancy, embedding,
                    created, source_node_ids=None, reflection_type=None):
        """
        Adds a thought memory to the memory stream. Depth is one more than
        the maximum depth of its source nodes.
        """
        node_id = self._generate_node_id()
        depth = 1
        if source_node_ids:
            try:
                depth += max(self.id_to_node[i].depth
                            for i in source_node_ids
                            if i in self.id_to_node)
            except ValueError:
                pass

        node = MemoryNode(
            node_id=node_id,
            node_type="thought",
            description=description,
            poignancy=poignancy,
            embedding=embedding,
            created=created,
            depth=depth,
            filling=source_node_ids or [],
            reflection_type=reflection_type
        )
        self.seq_thought.insert(0, node)
        self.id_to_node[node_id] = node
        return node

    def add_chat(self, description, poignancy, embedding,
                 created, dialogue_turns):
        """
        Adds a chat memory to the memory stream.
        dialogue_turns is a list of (speaker, utterance) tuples.
        """
        node_id = self._generate_node_id()
        node = MemoryNode(
            node_id=node_id,
            node_type="chat",
            description=description,
            poignancy=poignancy,
            embedding=embedding,
            created=created,
            depth=0,
            filling=dialogue_turns
        )
        self.seq_chat.insert(0, node)
        self.id_to_node[node_id] = node
        return node

    def get_all_nodes(self):
        """
        Returns all nodes sorted by last_accessed time (oldest first), as
        input to new_retrieve.
        """
        all_nodes = self.seq_event + self.seq_thought + self.seq_chat
        all_nodes = sorted(all_nodes, key=lambda x: (time_key(x.last_accessed), int(x.node_id.split("_")[1])))
        return all_nodes

    def save(self, filepath):
        """
        Saves the memory stream to a JSON file, with embeddings stored as
        lists.
        """
        data = {}
        for node_id, node in self.id_to_node.items():
            data[node_id] = {
                "node_type": node.node_type,
                "description": node.description,
                "poignancy": node.poignancy,
                "embedding": node.embedding if isinstance(node.embedding, list)
                             else node.embedding.tolist(),
                "created": node.created,
                "last_accessed": node.last_accessed,
                "depth": node.depth,
                "filling": node.filling,
                "reflection_type": node.reflection_type
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"node_count": self.node_count, "nodes": data}, f, indent=2)

    @classmethod
    def load(cls, filepath, memory_mode=None):
        """
        Loads an AssociativeMemory from a JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory = cls(memory_mode=memory_mode)
        memory.node_count = data["node_count"]

        for node_id, node_data in data["nodes"].items():
            node = MemoryNode(
                node_id=node_id,
                node_type=node_data["node_type"],
                description=node_data["description"],
                poignancy=node_data["poignancy"],
                embedding=node_data["embedding"],
                created=node_data["created"],
                depth=node_data["depth"],
                filling=node_data["filling"],
                reflection_type=node_data.get("reflection_type")
            )
            node.last_accessed = node_data["last_accessed"]

            memory.id_to_node[node_id] = node

            if node.node_type == "event":
                memory.seq_event.append(node)
            elif node.node_type == "thought":
                memory.seq_thought.append(node)
            elif node.node_type == "chat":
                memory.seq_chat.append(node)

        memory.seq_event.reverse()
        memory.seq_thought.reverse()
        memory.seq_chat.reverse()

        return memory

    def seed_memories(self, seed_descriptions, poignancy,
                      embedding_fn, created):
        """
        Seeds the memory stream with persona beliefs (single_tier only).
        Each seed is stored as a thought node with a fixed poignancy.

        Seeds are stored as thoughts, following Park et al.'s seeding
        mechanism (load_history_via_whisper), which uses add_thought. A
        fixed poignancy is used instead of LLM-scored poignancy, and the
        paraphrasing and expiration steps are omitted, to give a fixed,
        measurable baseline.
        """
        for seed in seed_descriptions:
            description = seed["description"]
            embedding = embedding_fn(description)
            self.add_thought(
                description=description,
                poignancy=poignancy,
                embedding=embedding,
                created=created,
                reflection_type="seed"
            )