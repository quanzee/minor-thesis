"""
Agent class for the cultural drift simulation, adapted from Park et al. (2023).

Configuration flags:
    memory_mode: "single_tier" or "two_tier"
        single_tier: Park et al. baseline. Memory seeds are stored in
            AssociativeMemory and persona beliefs are retrieved from it.
        two_tier: CoreBeliefs (Tier 1) plus AssociativeMemory (Tier 2).
            Memory seeds are stored in CoreBeliefs and persona beliefs are
            retrieved from CoreBeliefs only, unless frozen_core is set.
    frozen_core (two_tier only): the seeded CoreBeliefs are never revised
        (RQ3 condition C2). Persona context, VSM13 and WVS read the fixed
        core beliefs together with the agent's most relevant Tier 2
        thoughts.
    use_reasoning: the agent generates private reasoning before speaking
        when a scenario is active (RQ3 condition C3).
    anti_compromise: the conversation prompt instructs the agent not to
        soften its position (RQ3 condition C4).
"""

import json
import re
import yaml
from pathlib import Path
import sys
import random

_THIS_DIR = Path(__file__).parent
_LIB_DIR = Path(__file__).parent / "lib"
_BASE_DIR = Path(__file__).parent.parent

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_LIB_DIR))

from simplified_associative_memory import AssociativeMemory
from core_beliefs import CoreBeliefs
from retrieve import new_retrieve, retrieve_current_beliefs
from reflect import reflect, post_conversation_reflect
from config import client as DEFAULT_CLIENT, MODEL_NAME as DEFAULT_MODEL, api_call_with_retry


class Agent:
    def __init__(self, name, gender, age, country, education,
                 job_type, memory_seeds, embedding_fn,
                 client=None, model=None,
                 memory_mode="two_tier", use_reasoning=True,
                 anti_compromise=False, logs_dir=None,
                 frozen_core=False):
        # Identity
        self.name = name
        self.gender = gender
        self.age = age
        self.country = country
        self.education = education
        self.job_type = job_type

        # Config flags
        self.memory_mode = memory_mode
        self.use_reasoning = use_reasoning
        self.anti_compromise = anti_compromise
        # frozen_core only applies to two_tier agents
        self.frozen_core = bool(frozen_core) and memory_mode == "two_tier"

        # Cognitive infrastructure
        self.memory = AssociativeMemory(memory_mode=self.memory_mode)
        self.memory.agent_name = self.name
        self.embedding_fn = embedding_fn
        self.client = client if client is not None else DEFAULT_CLIENT
        self.model = model if model is not None else DEFAULT_MODEL

        if self.memory_mode == "two_tier":
            self.core = CoreBeliefs()
            self.core.agent_name = self.name
        else:
            self.core = None

        # Simulation state
        self.location = None
        self.curr_time = None
        self.chat_partner = None
        self.active_scenario = None
        self.scenario_rounds_announced = set()
        self.recent_partners = []

        # Reflection triggers once accumulated poignancy exceeds the
        # threshold (Park et al., 2023).
        self.importance_since_last_reflect = 0
        self.reflection_threshold = 30
        self.memories_since_last_reflect = []

        # Logging
        self.logs_dir = Path(logs_dir) if logs_dir is not None else (_BASE_DIR / "logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Seed beliefs/memories
        if self.memory_mode == "two_tier":
            self.core.seed_beliefs(
                seed_descriptions=memory_seeds,
                embedding_fn=embedding_fn,
                created="Day 00, Initialisation"
            )
        else:
            self.memory.seed_memories(
                seed_descriptions=memory_seeds,
                poignancy=8,
                embedding_fn=embedding_fn,
                created="Day 00, Initialisation"
            )

    @classmethod
    def from_yaml(cls, yaml_path, embedding_fn, client=None, model=None,
                  memory_mode="two_tier", use_reasoning=True,
                  anti_compromise=False, logs_dir=None,
                  frozen_core=False):
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)["agent"]
        return cls(
            name=data["name"],
            gender=data["gender"],
            age=data["age"],
            country=data["country"],
            education=data["education level"],
            job_type=data["job type"],
            memory_seeds=data["memory_seeds"],
            embedding_fn=embedding_fn,
            client=client,
            model=model,
            memory_mode=memory_mode,
            use_reasoning=use_reasoning,
            anti_compromise=anti_compromise,
            logs_dir=logs_dir,
            frozen_core=frozen_core
        )

    def demographic_summary(self) -> str:
        return (
            f"{self.name} is a {self.age}-year-old {self.gender} "
            f"from {self.country}. Their education level is {self.education} "
            f"and they work as {self.job_type}."
        )

    def retrieve(self, focal_points: list, curr_time: str, n: int = 30, node_types=None, update_access=True):
        """
        Retrieves relevant Tier 2 (AssociativeMemory) memories for a list
        of focal points. Used for conversational memory in both memory
        modes, separately from persona belief retrieval.
        """
        return new_retrieve(
            agent_memory=self.memory,
            focal_points=focal_points,
            embedding_fn=self.embedding_fn,
            n_count=n,
            curr_time=curr_time,
            node_types=node_types,
            update_access=update_access
        )

    def plan(self, other_agents: list, curr_time: str, tracker=None):
        """
        Chooses a conversation partner from the co-present agents, based on
        the agent's relevant memories. Inspired by Park et al.'s (2023)
        decision to initiate a conversation (generate_decide_to_talk in
        persona/cognitive_modules/plan.py, prompt decide_to_talk_v2.txt),
        but asks the agent to choose a partner rather than to answer yes or
        no for a given partner.
        """
        if not other_agents:
            return None

        focal_points = [
            f"relationship and experiences with people at {self.location}",
            f"my thoughts and feelings about my current situation",
            f"people I haven't spoken to yet today"
        ] + [f"experiences with {a.name}" for a in other_agents]

        retrieved = self.retrieve(focal_points, curr_time, n=3)

        seen_ids = set()
        memory_context = ""
        for nodes in retrieved.values():
            for node in nodes[:3]:
                if node.node_id not in seen_ids:
                    memory_context += f"- {node.description}\n"
                    seen_ids.add(node.node_id)

        agent_list = "\n".join([
            f"- {a.name} ({a.age}-year-old {a.gender} from {a.country}, {a.education}, {a.job_type})"
            + (f" - {self.name} has spoken with {a.name} {self.recent_partners.count(a.name)} time(s) recently"
               if a.name in self.recent_partners else " — not yet spoken today")
            for a in other_agents
        ])

        prompt = f"""You are {self.name}, a {self.age}-year-old {self.gender} from {self.country}. Your education level is {self.education} and you work as {self.job_type}.

        Your relevant memories and thoughts:
        {memory_context if memory_context else "No relevant memories yet."}

        The following people are currently in the same location as you:
        {agent_list}

        Choose one of these people to start a conversation with now.

        Respond in this exact format:
        TARGET: <name of one person from the list above>
        REASON: <one sentence reason>"""

        def ask(extra=""):
            response = api_call_with_retry(lambda: self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt + extra}],
                temperature=0.7
            ))
            if response is None:
                return None, {}
            if tracker:
                tracker.add(response.usage)
            content = response.choices[0].message.content.strip()
            lines = {}
            for line in content.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    lines[k.strip().strip("*# ").upper()] = v.strip().strip("* ")
            return content, lines

        names = {a.name.lower(): a for a in other_agents}

        def pick(lines):
            return names.get(lines.get("TARGET", "").strip(" .").lower())

        content, lines = ask()
        stated_target = lines.get("TARGET", "")
        target = pick(lines)
        attempts, fallback = 1, None

        if target is None:
            retry_content, retry_lines = ask(
                "\n\n        You must choose exactly one name from this list: "
                + ", ".join(a.name for a in other_agents) + "."
            )
            attempts = 2
            if retry_content is not None:
                content, lines = retry_content, retry_lines
                target = pick(lines)

        if target is None:
            target = random.choice(other_agents)
            fallback = "random"

        with open(self.logs_dir / "planning_logs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "agent": self.name, "culture": self.country, "curr_time": curr_time,
                "target": target.name, "stated_target": stated_target,
                "attempts": attempts, "fallback": fallback,
                "reason": lines.get("REASON", ""), "raw": content
            }, ensure_ascii=False) + "\n")
        return target

    def reflect(self, curr_time: str, tracker=None):
        """
        Runs reflection if the poignancy threshold has been reached.
        """
        return reflect(self, self.client, self.model,
                self.embedding_fn, curr_time, tracker=tracker)

    # Conversation helpers
    def _last_chat_with(self, partner):
        """Returns this agent's summary of its most recent conversation with partner."""
        node = next(
            (n for n in self.memory.seq_chat
             if any(turn[0] == partner.name for turn in n.filling)),
            None
        )
        if node is None:
            return ""
        return (f"\nMost recent conversation between {self.name} and {partner.name}:\n"
                f"{node.description}\n")

    def _scenario_framing(self, curr_time):
        """Returns the framing of the active scenario for the current round."""
        if not self.active_scenario:
            return None
        scenario_text = self.active_scenario.get("text")
        scenario_round = self.active_scenario.get("round")
        day_part, current_round = curr_time.split(", ")[0], curr_time.split(", ")[-1]

        if scenario_round == current_round:
            if f"{day_part}_{current_round}" in self.scenario_rounds_announced:
                return f"Something you discussed earlier this round:\n{scenario_text}"
            return f"An event that just happened and is on your mind:\n{scenario_text}"
        return f"Something that happened earlier today that you have been reflecting on:\n{scenario_text}"

    def _mark_scenario_discussed(self, curr_time):
        """Records that this agent has had a conversation in the current round."""
        day_part, current_round = curr_time.split(", ")[0], curr_time.split(", ")[-1]
        self.scenario_rounds_announced.add(f"{day_part}_{current_round}")

    def _generate_reasoning(self, agent_summary, scenario_framing, tracker, curr_time=None):
        if not self.use_reasoning or not scenario_framing:
            return None

        todays_chats = [
            n for n in self.memory.seq_chat
            if n.created and n.created.startswith(curr_time.split(",")[0])
        ] if curr_time else []

        todays_context = ""
        if todays_chats:
            todays_context = "\nBased on conversations you have had today:\n" + "\n".join(
                f"- {n.description}" for n in todays_chats
            ) + "\nGiven these interactions, consider where you now stand.\n"

        reasoning_prompt = f"""You are {self.name}.

Your background:
{agent_summary}

{scenario_framing}
{todays_context}
Before responding, reason through these questions as {self.name}:
- How do you personally feel about this event, given your values and beliefs?
- What would someone with your background and cultural outlook actually think — not what is socially expected, but what feels true to you?
- What do you want to say, and how would you say it in a way that reflects who you are?

Write your reasoning in 3-4 sentences, in first person."""

        reasoning_response = api_call_with_retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": reasoning_prompt}],
            temperature=0.7
        ))
        if reasoning_response is None:
            return None

        if tracker:
            tracker.add(reasoning_response.usage)

        return reasoning_response.choices[0].message.content.strip()

    def _conversation_side(self, partner, curr_time, tracker=None):
        """
        Builds this agent's context for a conversation with partner: persona
        summary and beliefs, memories about the partner, the summary of their
        last conversation, the scenario framing and, if enabled, private
        reasoning.
        """
        summary, belief_ids = self.get_persona_context(
            self.embedding_fn, curr_time,
            context=[self.active_scenario["text"]] if self.active_scenario else None
        )
        context = self.retrieve([
            f"relationship with {partner.name}",
            f"experiences with {partner.name}"],
            curr_time, n=3
        )
        framing = self._scenario_framing(curr_time)
        return {
            "partner": partner,
            "summary": summary,
            "belief_ids": belief_ids,
            "context": context,
            "prior": self._last_chat_with(partner),
            "framing": framing,
            "reasoning": self._generate_reasoning(summary, framing, tracker, curr_time=curr_time),
        }

    def _speak(self, side, conversation, curr_time, turn_index, n_turns, tracker=None):
        return self._generate_utterance(
            side["partner"], conversation, side["context"],
            curr_time, tracker=tracker,
            self_summary=side["summary"],
            self_belief_ids=side["belief_ids"],
            reasoning=side["reasoning"],
            scenario_context=side["framing"],
            turn_index=turn_index,
            n_turns=n_turns,
            prior_convo_context=side["prior"]
        )

    def prepare_opening(self, other_agent, curr_time: str, n_turns: int = 6, tracker=None):
        """
        Builds this agent's conversation context and generates its opening
        line from its memory at the start of the round. Called for every
        agent before any conversation in the round takes place, so no
        opening reflects a conversation held earlier in the same round.
        """
        side = self._conversation_side(other_agent, curr_time, tracker=tracker)
        utterance = self._speak(side, [], curr_time, 0, n_turns, tracker=tracker)
        if utterance is None:
            print(f"      Content filter triggered for {self.name}'s opening, retrying once.")
            utterance = self._speak(side, [], curr_time, 0, n_turns, tracker=tracker)
        side["utterance"] = utterance
        return side

    def converse(self, other_agent, curr_time: str,
                 n_turns: int = 6, tracker=None, opening=None):
        """
        Conducts a multi-turn conversation between this agent and another.

        Args:
            other_agent: the Agent to converse with.
            curr_time: current simulation time.
            n_turns: total number of utterances across both agents.
            opening: output of prepare_opening(); prepared here if None.

        Returns:
            A list of (speaker_name, utterance) tuples and the reasoning log.
        """
        if opening is None:
            opening = self.prepare_opening(other_agent, curr_time, n_turns=n_turns, tracker=tracker)

        self_side = opening
        other_side = other_agent._conversation_side(self, curr_time, tracker=tracker)
        sides = {self.name: self_side, other_agent.name: other_side}

        self._mark_scenario_discussed(curr_time)
        other_agent._mark_scenario_discussed(curr_time)

        reasoning_log = {self.name: [], other_agent.name: []}
        if self_side["reasoning"]:
            reasoning_log[self.name].append(self_side["reasoning"])
        if other_side["reasoning"]:
            reasoning_log[other_agent.name].append(other_side["reasoning"])

        conversation = []
        if opening["utterance"] is not None:
            conversation.append((self.name, opening["utterance"]))
        else:
            print(f"      Content filter triggered for {self.name}, skipping turn.")

        speakers = [self, other_agent]
        for i in range(1, n_turns):
            speaker = speakers[i % 2]
            utterance = speaker._speak(sides[speaker.name], conversation, curr_time,
                                       i, n_turns, tracker=tracker)
            if utterance is None:
                print(f"      Content filter triggered for {speaker.name}, skipping turn.")
                continue
            conversation.append((speaker.name, utterance))

        if not conversation:
            print(f"      Conversation between {self.name} and {other_agent.name} was empty, skipping summary.")
            return [], reasoning_log

        # Each agent stores its own summary of the conversation, written
        # from its perspective and informed by its pre-conversation reasoning.
        # Adapted from Park et al.'s (2023) summarize_conversation_v1.txt.
        for agent in [self, other_agent]:
            agent_reasoning = sides[agent.name]["reasoning"]

            summary_prompt = f"""Summarise this conversation from {agent.name}'s perspective, noting where they agreed, where they disagreed, and what they personally concluded.

        Starting reasoning: {agent_reasoning if agent_reasoning else 'No prior reasoning.'}

        Conversation:
        {chr(10).join(f'{speaker}: {utterance}' for speaker, utterance in conversation)}

        Respond with one or two sentences only, written as {agent.name}'s own take — not a neutral summary of both sides."""

            summary_response = api_call_with_retry(lambda: agent.client.chat.completions.create(
                model=agent.model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.0
            ))

            if summary_response is None:
                print(f"      Content filter triggered for {agent.name}'s conversation summary, skipping.")
                continue

            if tracker:
                tracker.add(summary_response.usage)

            description = summary_response.choices[0].message.content.strip()

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

            post_conversation_reflect(
                agent=agent,
                conversation_turns=conversation,
                chat_node_id=chat_node.node_id,
                client=agent.client,
                model=agent.model,
                embedding_fn=agent.embedding_fn,
                curr_time=curr_time,
                tracker=tracker
            )

        return conversation, reasoning_log

    @staticmethod
    def _state_path(save_dir: str, name: str) -> Path:
        """
        Returns the path of an agent's run state file. Run state is stored in
        results/state_snapshots/<dayNN>/, separately from memory_snapshots/.
        """
        save_dir = Path(save_dir)
        return save_dir.parent.parent / "state_snapshots" / save_dir.name / f"{name}_state.json"

    def save(self, save_dir: str):
        """
        Saves the agent's AssociativeMemory to disk, along with the run state
        held outside memory (reflection counter, memories awaiting
        reflection, recent partners) so that a resumed run continues from
        the same point.
        """
        memory_path = Path(save_dir) / f"{self.name}_memory.json"
        self.memory.save(str(memory_path))

        state_path = self._state_path(save_dir, self.name)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump({
                "importance_since_last_reflect": self.importance_since_last_reflect,
                "memories_since_last_reflect": [n.node_id for n in self.memories_since_last_reflect],
                "recent_partners": self.recent_partners,
                "scenario_rounds_announced": sorted(self.scenario_rounds_announced),
            }, f, indent=2)

    def load_memory(self, save_dir: str, core_dir: str = None):
        """
        Loads the agent's memory, CoreBeliefs (two_tier) and saved run state
        from a snapshot.
        """
        memory_path = Path(save_dir) / f"{self.name}_memory.json"
        self.memory = AssociativeMemory.load(str(memory_path), memory_mode=self.memory_mode)
        self.memory.agent_name = self.name
        if self.memory_mode == "two_tier" and core_dir is not None:
            core_path = Path(core_dir) / f"{self.name}_core.json"
            if core_path.exists():
                self.core = CoreBeliefs.load(str(core_path))

        state_path = self._state_path(save_dir, self.name)
        if state_path.exists():
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
            self.importance_since_last_reflect = state.get("importance_since_last_reflect", 0)
            self.memories_since_last_reflect = [
                self.memory.id_to_node[i]
                for i in state.get("memories_since_last_reflect", [])
                if i in self.memory.id_to_node
            ]
            self.recent_partners = state.get("recent_partners", [])
            self.scenario_rounds_announced = set(state.get("scenario_rounds_announced", []))
        else:
            print(f"      WARNING: no saved run state for {self.name} at {state_path}; "
                  f"reflection counter and recent partners start empty.")

    def _generate_utterance(self, other_agent, conversation_so_far,
                            retrieved_context, curr_time, tracker=None,
                            self_summary="", self_belief_ids=None,
                            reasoning=None, scenario_context=None, turn_index=0, n_turns=0,
                            prior_convo_context=None):
        """
        Generates a single utterance in a conversation. Adapted from Park et
        al.'s (2023) turn-by-turn utterance generation (generate_one_utterance
        in persona/cognitive_modules/converse.py, prompt iterative_convo_v1.txt),
        with the persona's beliefs, retrieved memories, scenario, optional
        private reasoning and optional anti-compromise instruction added to
        the context.
        """
        self_thoughts = ""
        seen = set(self_belief_ids) if self_belief_ids else set()
        for nodes in retrieved_context.values():
            for node in nodes[:3]:
                if node.node_id not in seen:
                    self_thoughts += f"- {node.description}\n"
                    seen.add(node.node_id)
        if not self_thoughts:
            self_thoughts = f"{self.name} does not have any prior thoughts about {other_agent.name}."

        if conversation_so_far:
            previous_convo = "\n".join(
                [f"{speaker}: {utterance}"
                for speaker, utterance in conversation_so_far]
            )
            previous_convo = f"Here is the conversation so far:\n{previous_convo}"
        else:
            previous_convo = ""

        reasoning_prefix = (
            f"You have reflected on the situation and are thinking:\n{reasoning}\n\n"
            if reasoning else ""
        )
        if not conversation_so_far:
            instruction = (
                f"Based on your own values and beliefs, what do you say to "
                f"{other_agent.name} to start the conversation?"
            )
        else:
            instruction = (
                f"Based on the conversation so far and your own values and beliefs, "
                f"what do you say next? Do not repeat points already made. "
                f"Respond to what {other_agent.name} has actually said."
            )
        reasoning_block = reasoning_prefix + instruction

        scenario_block = f"\n{scenario_context}\n" if scenario_context else ""
        if not prior_convo_context:
            no_prior_convo = f"Note: you and {other_agent.name} have not spoken before. Do not reference any previous conversations.\n"
        else:
            no_prior_convo = ""

        if turn_index == n_turns - 1:
            closing_instruction = (
                f"This is the final turn of the conversation. "
                f"You should bring the conversation to a natural close."
            )
        else:
            closing_instruction = ""

        anti_compromise_block = (
            "\n    Do not compromise your beliefs or soften your position to find a middle ground "
            "with the other person. Speak from your own view, even if it creates "
            "friction. You are not trying to resolve the disagreement or reach a solution — you are "
            "expressing where you actually stand.\n"
            if self.anti_compromise else ""
        )

        prompt = f"""You are {self.name}.

    {self_summary}

    You are speaking with {other_agent.name}, a {other_agent.age}-year-old {other_agent.gender} from {other_agent.country} who works as {other_agent.job_type}.
    ---
    Your thoughts and memories relevant to this conversation:
    {self_thoughts}
    {prior_convo_context}
    Currently, it is {curr_time}.
    {scenario_block} {reasoning_block}

    {previous_convo}

    {closing_instruction}

    {no_prior_convo}
    {anti_compromise_block}
    {self.name}: \""""

        response = api_call_with_retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stop=[f"\n{other_agent.name}:"]
        ))

        if response is None:
            return None

        if tracker:
            tracker.add(response.usage)

        utterance = response.choices[0].message.content.strip()
        utterance = re.sub(rf'^\s*{re.escape(self.name)}\s*:\s*', '', utterance).strip().strip('"')

        return utterance

    def _score_poignancy(self, description: str, memory_type: str = "event", tracker=None):
        """
        Scores the poignancy of a memory on a 1-10 scale. memory_type is
        "thought", "chat" or "event". Prompts adapted from Park et al. (2023)
        poignancy_thought_v1.txt, poignancy_chat_v1.txt and
        poignancy_event_v1.txt (generate_poig_score in
        persona/cognitive_modules/reflect.py).
        """
        if "idle" in description:
            return 1

        agent_summary = self.demographic_summary()

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

            On the scale of 1 to 10, where 1 is purely mundane (e.g., routine morning greetings) and 10 is extremely poignant (e.g., agreeing on an important plan of action, resolving a major disagreement, making a significant decision together), rate the likely poignancy of the following conversation for {self.name}.

            Conversation:
            {description}
            Rate (return a number between 1 to 10):
            Respond with a single integer only. Do not explain your reasoning."""
            
        # event memories are not generated in this study (retained from Park et al.)
        else:
            prompt = f"""Here is a brief description of {self.name}.
            {agent_summary}

            On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following event for {self.name}.

            Event: {description}
            Rate (return a number between 1 to 10):
            Respond with a single integer only. Do not explain your reasoning."""

        response = api_call_with_retry(lambda: self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        ))
        if response is None:
            print(f"      Content filter triggered for poignancy scoring, defaulting to 5")
            return 5

        if tracker:
            tracker.add(response.usage)

        try:
            raw = response.choices[0].message.content.strip()
            match = re.search(r'\b(\d+)\b', raw)
            score = int(match.group(1)) if match else 5
            print(f"      Poignancy [{memory_type}]: '{raw}' → {score}")
            return score
        except (ValueError, AttributeError):
            print(f"      Poignancy parse failed, defaulting to 5")
            return 5

    def get_persona_context(self, embedding_fn=None, curr_time=None, context=None, top_n=3):
        """
        Returns the persona summary (demographics and current beliefs) and
        the set of retrieved node IDs, which are excluded from conversational
        memory to avoid repetition. two_tier retrieves from CoreBeliefs (plus
        the top Tier 2 thoughts when frozen_core is set); single_tier
        retrieves from AssociativeMemory.
        """
        demographic = self.demographic_summary()

        if self.memory_mode == "two_tier":
            if not embedding_fn:
                return demographic, set()

            if context:
                query_text = "personal values and beliefs related to " + ", ".join(context)
            else:
                query_text = "personal values and beliefs"

            query_embedding = embedding_fn(query_text)
            retrieved_beliefs = self.core.retrieve(query_embedding, top_n=top_n)

            if not retrieved_beliefs:
                return demographic, set()

            belief_lines = [f"- {b.description}" for b in retrieved_beliefs]
            seen_ids = {b.node_id for b in retrieved_beliefs}

            # With a frozen core, the fixed beliefs are combined with the
            # most relevant Tier 2 thoughts, retrieved the same way as
            # single_tier beliefs.
            thought_ids = set()
            if self.frozen_core and curr_time:
                thought_text, thought_ids = retrieve_current_beliefs(
                    self, embedding_fn, curr_time, context=context
                )
                if thought_text:
                    belief_lines.append(thought_text)
                    seen_ids |= thought_ids

            summary = f"{demographic}\n\nCurrent values and beliefs:\n" + "\n".join(belief_lines)

            log_entry = {
                "agent": self.name,
                "culture": self.country,
                "curr_time": curr_time,
                "query_context": context,
                "query_text": query_text,
                "retrieved_beliefs": [
                    {"node_id": b.node_id, "description": b.description}
                    for b in retrieved_beliefs
                ],
                "retrieved_thoughts": [
                    {"node_id": i, "description": self.memory.id_to_node[i].description}
                    for i in sorted(thought_ids)
                ]
            }
            with open(self.logs_dir / "tier1_retrieval.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            return summary, seen_ids

        else:
            if embedding_fn and curr_time:
                beliefs, seen_ids = retrieve_current_beliefs(self, embedding_fn, curr_time, context=context)
                if beliefs:
                    summary = f"{demographic}\n\nCurrent values and beliefs:\n{beliefs}"
                    return summary, seen_ids

            return demographic, set()