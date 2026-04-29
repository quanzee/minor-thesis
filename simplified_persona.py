"""
File: agent.py
Description: Simplified Agent class for cultural drift simulation,
adapted from Park et al. (2023). Removes Smallville-specific dependencies
(spatial memory, maze, tiles, emoji) while preserving the core
memory-reflection-planning architecture.
"""

import datetime
import yaml
from pathlib import Path

from simplified_associative_memory import AssociativeMemory
from simplified_retrieve import new_retrieve
from reflect import reflect, post_conversation_reflect
from config import client, MODEL_NAME


class Agent:
    def __init__(self, name, gender, age, country, education,
                 job_type, memory_seeds, embedding_fn):
        # ── Identity ──────────────────────────────────────────────────────
        self.name = name
        self.gender = gender
        self.age = age
        self.country = country
        self.education = education
        self.job_type = job_type

        # ── Cognitive infrastructure ───────────────────────────────────────
        self.memory = AssociativeMemory()
        self.embedding_fn = embedding_fn
        self.client = client
        self.model = MODEL_NAME

        # ── Simulation state ───────────────────────────────────────────────
        self.location = None          # current location label e.g. "university"
        self.curr_time = None         # current simulation datetime
        self.chat_partner = None      # name of agent currently conversing with

        # ── Reflection trigger counter ─────────────────────────────────────
        # Accumulates poignancy of new memories since last reflection.
        # Reflection fires when this exceeds the threshold, following
        # Park et al. (2023).
        self.importance_since_last_reflect = 0
        self.reflection_threshold = 150

        # ── Seed memories ──────────────────────────────────────────────────
        created = datetime.datetime.now()
        self.memory.seed_memories(
            seed_descriptions=memory_seeds,
            poignancy=5, #need to think about this value
            embedding_fn=embedding_fn,
            created=created
        )

    @classmethod
    def from_yaml(cls, yaml_path, embedding_fn):
        """
        Loads an agent directly from a YAML file.
        """
        with open(yaml_path) as f:
            data = yaml.safe_load(f)["agent"]
        return cls(
            name=data["name"],
            gender=data["gender"],
            age=data["age"],
            country=data["country"],
            education=data["education level"],
            job_type=data["job type"],
            memory_seeds=data["memory_seeds"],
            embedding_fn=embedding_fn
        )

    def perceive(self, observation: str, curr_time: datetime.datetime):
        """
        Adds an observation to the agent's memory stream.
        Replaces the maze-based perceive in Park et al. — observations
        are passed directly as strings rather than inferred from tile proximity.

        INPUT:
            observation: natural language description of what the agent observes
            curr_time: current simulation datetime
        OUTPUT:
            the created MemoryNode
        """
        poignancy = self._score_poignancy(observation)
        embedding = self.embedding_fn(observation)
        node = self.memory.add_event(
            description=observation,
            poignancy=poignancy,
            embedding=embedding,
            created=curr_time
        )
        self.importance_since_last_reflect += poignancy
        return node

    def retrieve(self, focal_points: list, curr_time: datetime.datetime):
        """
        Retrieves relevant memories for a list of focal points.

        INPUT:
            focal_points: list of strings describing current focus
            curr_time: current simulation datetime
        OUTPUT:
            dict mapping focal point string to list of MemoryNodes
        """
        return new_retrieve(
            agent_memory=self.memory,
            focal_points=focal_points,
            embedding_fn=self.embedding_fn,
            n_count=30,
            curr_time=curr_time
        )

    def plan(self, other_agents: list, curr_time: datetime.datetime):
        """
        Decides whether to initiate a conversation and with whom,
        based on current memory state and who is co-present.
        """
        if not other_agents:
            return None

        # Generate focal points based on co-present agents and current location
        focal_points = [
            f"relationship and experiences with people at {self.location}",
            f"my thoughts and feelings about my current situation"
        ] + [f"experiences with {a.name}" for a in other_agents]

        # Retrieve relevant memories using the proper retrieval mechanism
        retrieved = self.retrieve(focal_points, curr_time) #30 nodes

        # Flatten retrieved nodes into a deduplicated context string
        seen_ids = set()
        memory_context = ""
        for nodes in retrieved.values():
            for node in nodes[:3]: #takes only the top 3 from each focal point's results
                if node.node_id not in seen_ids:
                    memory_context += f"- {node.description}\n"
                    seen_ids.add(node.node_id)

        # List of available agents
        agent_list = "\n".join([
            f"- {a.name} ({a.age}-year-old {a.gender} from {a.country}, {a.education}, {a.job_type})"
            for a in other_agents
        ])

        prompt = f"""You are {self.name}, a {self.age}-year-old {self.gender} from {self.country}. Your education level is {self.education} and you work as {self.job_type}.

        Your relevant memories and thoughts:
        {memory_context if memory_context else "No relevant memories yet."}

        The following people are currently in the same location as you:
        {agent_list}

        Would you like to start a conversation with anyone? If yes, who and why?
        If no, explain why not.

        Respond in this exact format:
        DECISION: yes or no
        TARGET: <name of person> or none
        REASON: <one sentence reason>"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 #justification?
        )

        content = response.choices[0].message.content.strip()

        try:
            lines = {
                line.split(":")[0].strip(): line.split(":", 1)[1].strip()
                for line in content.split("\n")
                if ":" in line
            }
            if lines.get("DECISION", "no").lower() == "yes":
                target_name = lines.get("TARGET", "none").strip()
                target = next(
                    (a for a in other_agents if a.name == target_name), None
                )
                return target
        except (ValueError, StopIteration):
            pass

        return None

    def reflect(self, curr_time: datetime.datetime):
        """
        Checks reflection trigger and runs reflection if threshold is met.
        Wraps the reflect function from reflect.py.
        """
        reflect(self, self.client, self.model,
                self.embedding_fn, curr_time)

    def converse(self, other_agent, curr_time: datetime.datetime,
                 n_turns: int = 6): #
        """
        Conducts a multi-turn conversation between this agent and another.
        Each agent generates their utterance in turn based on their memory
        and persona. Replaces open_convo_session from Park et al.

        INPUT:
            other_agent: the Agent to converse with
            curr_time: current simulation datetime
            n_turns: total number of utterances across both agents
        OUTPUT:
            list of (speaker_name, utterance) tuples
        """
        # retrieve once at the start for both agents
        self_context = self.retrieve(
            [f"relationship with {other_agent.name}",
            f"experiences at {self.location}"],
            curr_time
        )
        other_context = other_agent.retrieve(
            [f"relationship with {self.name}",
            f"experiences at {self.location}"],
            curr_time
        )

        conversation = []
        current_speaker = self
        other_speaker = other_agent
        current_context = self_context
        other_context_swap = other_context

        for _ in range(n_turns):
            utterance = current_speaker._generate_utterance(
                other_speaker, conversation, current_context, curr_time
            )
            conversation.append((current_speaker.name, utterance))
            current_speaker, other_speaker = other_speaker, current_speaker
            current_context, other_context_swap = other_context_swap, current_context

        # Store conversation in both agents' memories
        description = (f"{self.name} and {other_agent.name} "
                       f"had a conversation at {self.location}")

        for agent in [self, other_agent]:
            poignancy = agent._score_poignancy(description)
            embedding = agent.embedding_fn(description)
            chat_node = agent.memory.add_chat(
                description=description,
                poignancy=poignancy,
                embedding=embedding,
                created=curr_time,
                dialogue_turns=conversation
            )
            agent.importance_since_last_reflect += poignancy

            # post-conversation reflection
            post_conversation_reflect(
                agent=agent,
                conversation_turns=conversation,
                chat_node_id=chat_node.node_id,
                client=agent.client,
                model=agent.model,
                embedding_fn=agent.embedding_fn,
                curr_time=curr_time
            )

        return conversation

    def save(self, save_dir: str):
        """
        Saves the agent's memory state to disk.
        """
        path = Path(save_dir) / f"{self.name}_memory.json"
        self.memory.save(str(path))

    def load_memory(self, save_dir: str):
        """
        Loads the agent's memory state from disk.
        """
        path = Path(save_dir) / f"{self.name}_memory.json"
        self.memory = AssociativeMemory.load(str(path))

    def _generate_utterance(self, other_agent, conversation_so_far,
                             retrieved_context, curr_time: datetime.datetime):
        """
        Generates a single utterance in a conversation, grounded in
        the agent's memory and persona.
        """
        memory_context = ""
        for nodes in retrieved_context.values():
            for node in nodes[:3]:
                memory_context += f"- {node.description}\n"

        # format conversation so far
        convo_so_far = "\n".join(
            [f"{speaker}: {utterance}"
             for speaker, utterance in conversation_so_far]
        ) if conversation_so_far else "This is the start of the conversation."

        prompt = f"""You are {self.name}, a {self.age}-year-old {self.gender} from {self.country}.
        Your education level is {self.education} and you work as {self.job_type}.

        Relevant memories:
        {memory_context if memory_context else "No prior memories of this person."}

        You are speaking with {other_agent.name}, a {other_agent.age}-year-old {other_agent.gender} from {other_agent.country}. Their education level is {other_agent.education} and they work as {other_agent.job_type}.

        Conversation so far:
        {convo_so_far}

        Respond naturally as {self.name} in one or two sentences. Stay true to your 
        background and values."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def _score_poignancy(self, description: str):
        """
        Scores the emotional significance of a memory on a 1-10 scale.
        """
        prompt = f"""On a scale of 1 to 10, how emotionally significant is this event for a person?
        Event: "{description}"
        Respond with a single integer only."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        try:
            return int(response.choices[0].message.content.strip())
        except ValueError:
            return 5