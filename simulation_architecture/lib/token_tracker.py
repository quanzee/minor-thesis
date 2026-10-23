"""
Token usage tracking across API calls.
"""
from datetime import datetime
import json

class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def add(self, usage):
        """Adds token usage from an API response."""
        if usage:
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens
            self.total_tokens += usage.total_tokens

    def report(self, label: str):
        """Prints a token usage summary."""
        print(f"\n  Token Usage — {label}")
        print(f"    Prompt tokens:     {self.prompt_tokens:,}")
        print(f"    Completion tokens: {self.completion_tokens:,}")
        print(f"    Total tokens:      {self.total_tokens:,}")

    def reset(self):
        """Resets the counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def append_log(self, label: str, filepath):
        """Appends the current token usage to a log file."""
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

    def start_run(self, filepath, run_label: str = None):
        entry = {
            "event": "run_start",
            "run_label": run_label,
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def end_run(self, filepath):
        from datetime import datetime
        import json
        entry = {
            "event": "run_end",
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")