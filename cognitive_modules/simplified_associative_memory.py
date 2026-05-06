"""
File: associative_memory.py
Description: Simplified long-term memory module for generative agents,
adapted from Park et al. (2023). Strips Smallville-specific dependencies
while preserving the core memory stream architecture.
"""

import json

class MemoryNode:
    def __init__(self, node_id, node_type, description,
                 poignancy, embedding, created, depth=0, filling=None):
        self.node_id = node_id
        self.node_type = node_type      # "event", "thought", "chat"
        self.description = description  #the natural langauge memory string
        self.poignancy = poignancy      # importance score 1-10
        self.embedding = embedding      # vector stored directly
        self.created = created          # string object for when the memory was created
        self.last_accessed = created    # datetime object for when the memory was last accessed (for retrieval sorting)
        self.depth = depth              # 0 for events/chats, 1+ for thoughts
        self.filling = filling or []    # dialogue turns for chats


class AssociativeMemory:
    def __init__(self):
        self.id_to_node = dict()        # node_id -> MemoryNode
        self.seq_event = []             # chronological list of events
        self.seq_thought = []           # chronological list of thoughts
        self.seq_chat = []              # chronological list of chats
        self.node_count = 0             # total node counter for generating IDs

    def _generate_node_id(self):
        self.node_count += 1
        return f"node_{self.node_count}"

    def add_event(self, description, poignancy, embedding, created):
        """
        Adds an event memory to the memory stream.
        Events are direct observations of things that happened.
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
                    created, source_node_ids=None):
        """
        Adds a reflection/thought memory to the memory stream.
        Thoughts are higher-level insights synthesised from events.
        Depth is 1 + max depth of source nodes, reflecting how abstract
        the reflection is.
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
            filling=source_node_ids or []
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
        Returns all nodes (events, thoughts, chats) sorted by 
        last_accessed time, excluding idle nodes.
        Used as input to new_retrieve.
        """
        all_nodes = self.seq_event + self.seq_thought + self.seq_chat
        all_nodes = [n for n in all_nodes if "idle" not in n.description]
        all_nodes = sorted(all_nodes, key=lambda x: x.last_accessed)
        return all_nodes

    def save(self, filepath):
        """
        Saves the memory stream to a JSON file.
        Embeddings are stored as lists since JSON doesn't support numpy arrays.
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
                "filling": node.filling
            }

        with open(filepath, "w") as f:
            json.dump({"node_count": self.node_count, "nodes": data}, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """
        Loads a memory stream from a JSON file and reconstructs
        the AssociativeMemory object.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        memory = cls()
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
                filling=node_data["filling"]
            )
            node.last_accessed = node_data["last_accessed"]

            memory.id_to_node[node_id] = node

            if node.node_type == "event":
                memory.seq_event.append(node)
            elif node.node_type == "thought":
                memory.seq_thought.append(node)
            elif node.node_type == "chat":
                memory.seq_chat.append(node)

        return memory

    def seed_memories(self, seed_descriptions, poignancy, 
                      embedding_fn, created):
        """
        Initialises the memory stream with persona seed memories.
        Each seed is added as an event node with a fixed poignancy score.
        embedding_fn is a callable that takes a string and returns a vector.
        """
        for description in seed_descriptions:
            embedding = embedding_fn(description)
            self.add_event(
                description=description,
                poignancy=poignancy,
                embedding=embedding,
                created=created
            )