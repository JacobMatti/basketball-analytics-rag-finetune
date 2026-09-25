"""Step-by-step tracing for the answer pipeline.

Every question gets a Trace. Each step records an event (stage, message, data).
Events are printed as they happen and saved in full to a JSON Lines log, so a run
can be studied afterwards: prompts, raw model replies, the SQL, the rows, every
check that passed or failed, and why the pipeline answered or abstained.

Console detail is set with RAG_DEBUG:
    RAG_DEBUG=0   only each question and its final result
    RAG_DEBUG=1   every step, with long values shortened (default)
    RAG_DEBUG=2   every step in full, including complete prompts
The log file always gets everything in full.
"""
import json
import os
import time

DEBUG_LEVEL = int(os.getenv("RAG_DEBUG", "1") or 0)
PREVIEW_CHARS = 600


def _shorten(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [{len(text) - limit} more chars]"


class Trace:
    """Collects every step of answering one question."""

    def __init__(self, label: str, question: str):
        self.label = label
        self.question = question
        self.events: list[dict] = []
        self.started = time.time()
        self.ended = None
        self.outcome = None
        self.reason = None
        if DEBUG_LEVEL >= 0:
            print(f"\n{'=' * 100}\n{label}: {question}\n{'=' * 100}")

    def log(self, stage: str, message: str, data=None, full: bool = False) -> None:
        """Record one step.

        :param stage: short step name, e.g. "sql.reply"
        :param message: one-line description
        :param data: any JSON-serializable detail (prompt, rows, reply, ...)
        :param full: long data (like a whole prompt) that the console shows only at
            RAG_DEBUG=2; it is always saved to the log file
        """
        self.events.append({
            "t": round(time.time() - self.started, 2),
            "stage": stage,
            "message": message,
            "data": data,
        })
        if DEBUG_LEVEL < 1:
            return
        print(f"  [{stage}] {message}")
        if data is None:
            return
        if full and DEBUG_LEVEL < 2:
            print(f"      (full text in log file; {len(_shorten(data, 10**9))} chars)")
            return
        limit = 10**9 if DEBUG_LEVEL >= 2 else PREVIEW_CHARS
        for line in _shorten(data, limit).splitlines():
            print(f"      {line}")

    def abstain(self, reason: str) -> None:
        """Mark the question as abstained, with the reason."""
        self.outcome, self.reason = "abstained", reason
        self.ended = time.time()
        self.log("result", f"ABSTAIN: {reason}")

    def answered(self, result: dict) -> None:
        """Mark the question as answered."""
        self.outcome = "answered"
        self.ended = time.time()
        self.log("result", "ANSWERED", result)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "question": self.question,
            "outcome": self.outcome,
            "reason": self.reason,
            "seconds": round((self.ended or time.time()) - self.started, 1),
            "events": self.events,
        }


def write_log(path: str, traces: list[Trace]) -> None:
    """Write one JSON object per question (JSON Lines), overwriting the file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict(), default=str) + "\n")


def append_log(path: str, trace: Trace) -> None:
    """Append one trace to a JSON Lines file (used by the chat server)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace.to_dict(), default=str) + "\n")


def print_summary(traces: list[Trace]) -> None:
    """Print a one-line-per-question table at the end of a run."""
    print(f"\n{'=' * 100}\nSUMMARY\n{'=' * 100}")
    for t in traces:
        detail = "" if t.outcome == "answered" else f"  ({t.reason})"
        print(f"  {t.label:<6} {t.outcome or '?':<10} {t.to_dict()['seconds']:>6}s{detail}")
    answered = sum(t.outcome == "answered" for t in traces)
    print(f"  answered {answered} of {len(traces)}")
