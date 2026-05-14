"""
File: agent.py
Description: Simplified Agent class for cultural drift simulation,
adapted from Park et al. (2023). Removes Smallville-specific dependencies
(spatial memory, maze, tiles, emoji) while preserving the core
memory-reflection-planning architecture.
"""

import yaml
from pathlib import Path

from cognitive_modules.simplified_associative_memory import AssociativeMemory
from cognitive_modules.simplified_retrieve import new_retrieve
from cognitive_modules.simplified_reflect import reflect, post_conversation_reflect
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
        self.primary_location = self._assign_primary_location()
        self.curr_time = None         # current simulation datetime
        self.chat_partner = None      # name of agent currently conversing with

        # ── Reflection trigger counter ─────────────────────────────────────
        # Accumulates poignancy of new memories since last reflection.
        # Reflection fires when this exceeds the threshold, following
        # Park et al. (2023).
        self.importance_since_last_reflect = 0
        self.reflection_threshold = 75
        self.memories_since_last_reflect = []

        # ── Seed memories ──────────────────────────────────────────────────
        self.memory.seed_memories(
            seed_descriptions=memory_seeds,
            poignancy=8, #need to think about this value
            embedding_fn=embedding_fn,
            created="Day 00, Initialisation"
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

    def perceive(self, observation: str, curr_time: str, tracker=None, poignancy_override=None):
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
        poignancy = poignancy_override if poignancy_override is not None else self._score_poignancy(observation, memory_type="event", tracker=tracker)
        embedding = self.embedding_fn(observation)
        node = self.memory.add_event(
            description=observation,
            poignancy=poignancy,
            embedding=embedding,
            created=curr_time
        )
        self.importance_since_last_reflect += poignancy
        self.memories_since_last_reflect.append(node)
        return node

    def retrieve(self, focal_points: list, curr_time: str):
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

    def plan(self, other_agents: list, curr_time: str, tracker=None):
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
        if tracker:
            tracker.add(response.usage)

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

    def reflect(self, curr_time: str, tracker=None):
        """
        Checks reflection trigger and runs reflection if threshold is met.
        Wraps the reflect function from reflect.py.
        """
        reflect(self, self.client, self.model,
                self.embedding_fn, curr_time, tracker=tracker)

    def converse(self, other_agent, curr_time: str,
                 n_turns: int = 6, tracker=None, scenario_focal: str = None): #
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
        # get most recent high-importance event as scenario anchor
        focal_points = [
            f"relationship with {other_agent.name}",
            f"experiences at {self.location}"
        ]
        if scenario_focal:
            focal_points.append(scenario_focal)

        self_context = self.retrieve(focal_points, curr_time)
        other_context = other_agent.retrieve(
            [f"relationship with {self.name}",
             f"experiences at {self.location}",
             *([ scenario_focal] if scenario_focal else [])],
            curr_time
        )

        conversation = []
        current_speaker = self
        other_speaker = other_agent
        current_context = self_context
        other_context_swap = other_context

        for _ in range(n_turns):
            utterance = current_speaker._generate_utterance(
                other_speaker, conversation, current_context, other_context_swap,curr_time
            )
            conversation.append((current_speaker.name, utterance))
            current_speaker, other_speaker = other_speaker, current_speaker
            current_context, other_context_swap = other_context_swap, current_context

        # Store conversation in both agents' memories
        description = (f"{self.name} and {other_agent.name} "
                       f"had a conversation at {self.location}")

        for agent in [self, other_agent]:
            poignancy = agent._score_poignancy(description, memory_type="chat", tracker=tracker)
            embedding = agent.embedding_fn(description)
            chat_node = agent.memory.add_chat(
                description=description,
                poignancy=poignancy,
                embedding=embedding,
                created=curr_time,
                dialogue_turns=conversation
            )
            agent.importance_since_last_reflect += poignancy
            agent.memories_since_last_reflect.append(chat_node)

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
                            retrieved_context, other_retrieved_context, curr_time, tracker=None):
        """
        Generates a single utterance in a conversation.
        Prompt adapted from Park et al. (2023) create_conversation_v2.txt.
        """
        # build agent summary strings
        self_summary = (
            f"{self.name} is a {self.age}-year-old {self.gender} "
            f"from {self.country}. Their education level is {self.education} "
            f"and they work as {self.job_type}."
        )
        other_summary = (
            f"{other_agent.name} is a {other_agent.age}-year-old "
            f"{other_agent.gender} from {other_agent.country}. "
            f"Their education level is {other_agent.education} "
            f"and they work as {other_agent.job_type}."
        )

        # what self thinks about other_agent — from retrieved context
        self_thoughts = ""
        for nodes in retrieved_context.values():
            for node in nodes[:3]:
                self_thoughts += f"- {node.description}\n"
        if not self_thoughts:
            self_thoughts = f"{self.name} does not have any prior thoughts about {other_agent.name}."

        # what other_agent thinks about self — retrieve separately
        other_thoughts = ""
        for nodes in other_retrieved_context.values():
            for node in nodes[:3]:
                other_thoughts += f"- {node.description}\n"
        if not other_thoughts:
            other_thoughts = f"{other_agent.name} does not have any prior thoughts about {self.name}."

        # format conversation so far
        if conversation_so_far:
            previous_convo = "\n".join(
                [f"{speaker}: {utterance}"
                for speaker, utterance in conversation_so_far]
            )
            previous_convo = f"Here is the conversation so far:\n{previous_convo}"
        else:
            previous_convo = ""

        prompt = f"""We have two characters.

        Character 1.
        {self_summary}

        Character 2.
        {other_summary}
        ---
        Context:
        Here is what {self.name} thinks about {other_agent.name}:
        {self_thoughts}
        Here is what {other_agent.name} thinks about {self.name}:
        {other_thoughts}
        Currently, it is {curr_time}
        -- {self.name} is at {self.location}
        -- {other_agent.name} is at {other_agent.location}
        {previous_convo}

        {self.name} and {other_agent.name} are in {self.location}. What would they talk about now?
        
        {self.name}: \""""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stop=[f"\n{other_agent.name}:"]
        )
        if tracker:
            tracker.add(response.usage)

        utterance = response.choices[0].message.content.strip().strip('"')
        return utterance

    def _score_poignancy(self, description: str, memory_type: str = "event", tracker=None):
        """
        Scores the poignancy of a memory on a 1-10 scale.
        Uses different prompt calibration per memory type,
        adapted from Park et al. (2023).
        
        memory_type: "event", "thought", or "chat"
        """
        if "idle" in description:
            return 1

        agent_summary = (
            f"{self.name} is a {self.age}-year-old {self.gender} "
            f"from {self.country}. Their education level is {self.education} "
            f"and they work as {self.job_type}."
        )

        if memory_type == "thought":
            prompt = f"""Here is a brief description of {self.name}.
            {agent_summary}

            On the scale of 1 to 10, where 1 is purely mundane (e.g., I need to do the dishes, I need to walk the dog) and 10 is extremely significant (e.g., I wish to become a professor, I love Elie), rate the likely significance of the following thought for {self.name}.

            Thought: {description}
            Rate (return a number between 1 to 10):
            Respond with a single integer only. Do not explain your reasoning."""

        elif memory_type == "chat":
            prompt = f"""Here is a brief description of {self.name}.
            {agent_summary}

            On the scale of 1 to 10, where 1 is purely mundane (e.g., routine morning greetings) and 10 is extremely poignant (e.g., a conversation about breaking up, a fight), rate the likely poignancy of the following conversation for {self.name}.

            Conversation:
            {description}
            Rate (return a number between 1 to 10):
            Respond with a single integer only. Do not explain your reasoning."""

        else:  # event
            prompt = f"""Here is a brief description of {self.name}.
            {agent_summary}

            On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following event for {self.name}.

            Event: {description}
            Rate (return a number between 1 to 10):
            Respond with a single integer only. Do not explain your reasoning."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        if tracker:
            tracker.add(response.usage)

        try:
            raw = response.choices[0].message.content.strip()
            import re
            match = re.search(r'\b(\d+)\b', raw)
            score = int(match.group(1)) if match else 5
            print(f"      Poignancy [{memory_type}]: '{raw}' → {score}")
            return score
        except (ValueError, AttributeError):
            print(f"      Poignancy parse failed, defaulting to 5")
            return 5

        # try:
        #     return int(response.choices[0].message.content.strip())
        # except ValueError:
        #     return 5
        
    def plan_location(self, curr_time, tracker=None):
        """
        Decides which location to go to this round.
        Defaults to primary institution unless memories
        suggest a strong reason to go elsewhere.
        """
        try:
            retrieved = self.retrieve(
                [f"plans and intentions for today",
                f"people I want to talk to"],
                curr_time
            )

            memory_context = ""
            seen_ids = set()
            for nodes in retrieved.values():
                for node in nodes[:3]:
                    if node.node_id not in seen_ids:
                        memory_context += f"- {node.description}\n"
                        seen_ids.add(node.node_id)

            prompt = f"""You are {self.name}, a {self.age}-year-old {self.gender} from {self.country}. Your education level is {self.education} and you work as {self.job_type}.

            Your primary location is {self.primary_location}.

            The available locations are:
            - University
            - Workplace
            - Communal Space

            Your relevant memories:
            {memory_context if memory_context else "No relevant memories yet."}

            Where will you go this round? You should go to your primary location unless you have a specific reason based on your memories to go elsewhere.

            Respond in this exact format:
            LOCATION: <location name>
            REASON: <one sentence reason>"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            if tracker:
                tracker.add(response.usage)

            content = response.choices[0].message.content.strip()

            try:
                lines = {
                    line.split(":")[0].strip(): line.split(":", 1)[1].strip()
                    for line in content.split("\n")
                    if ":" in line
                }
                location = lines.get("LOCATION", self.primary_location).strip()
                if location not in ["University", "Workplace", "Communal Space"]:
                    return self.primary_location
                return location
            except:
                return self.primary_location
        except Exception as e:
            print(f"      WARNING: plan_location failed for {self.name}: {e}")
            print(f"      Defaulting to primary location: {self.primary_location}")
            return self.primary_location
        
    def _assign_primary_location(self) -> str:
        """
        Assigns the agent's primary institution based on job type.
        """
        if self.job_type == "Full-time student":
            return "University"
        else:
            return "Workplace"