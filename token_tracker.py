"""
File: token_tracker.py
Description: Tracks token usage across API calls during simulation.
"""


class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def add(self, usage):
        """Add token usage from an API response."""
        if usage:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens

    def report(self, label: str):
        """Print a token usage summary."""
        print(f"\n  Token Usage — {label}")
        print(f"    Prompt tokens:     {self.prompt_tokens:,}")
        print(f"    Completion tokens: {self.completion_tokens:,}")
        print(f"    Total tokens:      {self.total_tokens:,}")

    def reset(self):
        """Reset counters for a new day."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def append_log(self, label: str, filepath):
        """Appends current token usage with a label to a running log file."""
        import json
        from pathlib import Path
        entry = {
            "label": label,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")