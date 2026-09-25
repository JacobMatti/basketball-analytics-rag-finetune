"""FastAPI frontend endpoint using the same JSON-plan pipeline as Part 1."""
import os
import traceback

import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text

from backend.config import DB_DSN, LLM_MODEL
from backend.debug import Trace, append_log
from backend.planner import answer_question
from backend.common import Context
from backend.utils import ollama_generate_full

MAX_EVIDENCE_SHOWN = 25
CHAT_LOG_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "part1", "debug", "chat_log.jsonl"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
eng = sa.create_engine(DB_DSN)
_ctx: Context | None = None
_count = 0


class Q(BaseModel):
    question: str


def _shape(table: str) -> dict:
    if table == "game_details":
        return {"table": table, "id": "int"}
    if table == "player_box_scores":
        return {"table": table, "game_id": "int", "person_id": "int"}
    return {"table": table, "id": "int"}


def infer_chat_return(question: str, trace: Trace | None = None) -> dict:
    """Use Qwen to infer only the semantic fields the frontend should return.

    Part 1 already supplies an exact return block in questions.json. Free-text chat does
    not, so this adapter asks the same small model to choose from a closed list of
    supported answer fields. It cannot invent a generic ``answer`` field. The actual
    retrieval, SQL, aggregation, validation and evidence construction still happen in
    planner.answer_question().
    """
    field_types = {
        "player_name": "str",
        "points": "int",
        "games": "int",
        "opponent": "str",
        "margin": "int",
        "deficit": "int",
        "score": "str",
        "winner": "str",
        "status": "str",
        "reason": "str",
    }
    allowed = list(field_types)
    schema = {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {"type": "string", "enum": allowed},
                "minItems": 1,
                "uniqueItems": True,
            }
        },
        "required": ["fields"],
        "additionalProperties": False,
    }
    prompt = f"""You classify what answer fields a basketball question requests.
Return JSON only. Choose one or more fields from this exact list:
{', '.join(allowed)}

Meanings:
- player_name: a player's name
- points: points scored
- games: number of games
- opponent: the opposing team
- margin: final point margin of a win or loss
- deficit: an in-game deficit that was overcome
- score: final game score
- winner: winning team
- status: injury/availability status
- reason: injury/availability reason

Select every field needed to fully express the requested answer.
Important:
- "scored" or "scoring" about a player means points, not final score.
- "final score" means score.
- For "biggest/largest win" or "biggest/largest loss/defeat", include opponent, margin, and score even if the wording does not explicitly name all three; margin is the quantity that defines biggest/largest.
- "who led" / "who scored the most" requires player_name plus the requested statistic.
- "most games" requires player_name and games.
- A go-ahead basket asks for player_name.
- Why a player missed a game asks for status and reason.
- A largest comeback/deficit question asks for opponent, deficit, and score.

Question: {question}
JSON:"""
    raw = ollama_generate_full(LLM_MODEL, prompt, json_schema=schema)
    parsed = __import__("json").loads(raw["response"])
    fields = parsed["fields"]
    if trace is not None:
        trace.log("chat.fields", "LLM-inferred return fields", fields)
        trace.log("chat.fields.stats",
                  f"prompt tokens read={raw.get('prompt_eval_count', 0)}, "
                  f"tokens written={raw.get('eval_count', 0)}, "
                  f"seconds={raw.get('total_duration', 0) / 1e9:.1f}")

    result: dict = {"answerable": "bool"}
    for field in fields:
        result[field] = field_types[field]

    # Evidence shape follows the kind of fact requested. Narrative questions need the
    # narrative row together with its game; player facts also cite the box-score row.
    lowered = question.lower()
    if "status" in fields or "reason" in fields:
        evidence = [_shape("injury_notes"), _shape("game_details")]
    elif "deficit" in fields or any(x in lowered for x in ("go-ahead", "recap", "comeback")):
        evidence = [_shape("game_details"), _shape("game_recaps")]
        if "player_name" in fields:
            evidence.append(_shape("player_box_scores"))
    elif "player_name" in fields or any(f in fields for f in ("points", "games")):
        evidence = [_shape("game_details"), _shape("player_box_scores")]
    else:
        evidence = [_shape("game_details")]
    result["evidence"] = evidence
    return result


def describe_evidence(cx, item: dict) -> str | None:
    table = item["table"]
    if table == "game_details":
        row = cx.execute(text(
            "SELECT LEFT(CAST(g.game_timestamp AS TEXT), 10) AS day, h.name AS home, a.name AS away, "
            "g.home_points, g.away_points FROM game_details g "
            "JOIN teams h ON h.team_id = g.home_team_id JOIN teams a ON a.team_id = g.away_team_id "
            "WHERE g.game_id = :i"), {"i": item["id"]}).mappings().first()
        return row and f"{row['day']}: {row['away']} {row['away_points']} at {row['home']} {row['home_points']}"
    if table == "player_box_scores":
        row = cx.execute(text(
            "SELECT p.first_name, p.last_name, b.points, b.offensive_reb + b.defensive_reb AS reb, "
            "b.assists, b.starter FROM player_box_scores b JOIN players p ON p.player_id = b.person_id "
            "WHERE b.game_id = :g AND b.person_id = :p"),
            {"g": item["game_id"], "p": item["person_id"]}).mappings().first()
        return row and (f"{row['first_name']} {row['last_name']}: {row['points']} pts, {row['reb']} reb, "
                        f"{row['assists']} ast{' (starter)' if row['starter'] else ''}")
    if table == "game_recaps":
        row = cx.execute(text("SELECT text FROM game_recaps WHERE recap_id = :i"), {"i": item["id"]}).first()
        return row and row[0]
    if table == "injury_notes":
        row = cx.execute(text("SELECT status, text FROM injury_notes WHERE note_id = :i"), {"i": item["id"]}).first()
        return row and f"[{row[0]}] {row[1]}"
    return None


def format_answer(result: dict) -> str:
    values = [(k, v) for k, v in result.items() if k not in ("answerable", "evidence")]
    if len(values) == 1:
        return str(values[0][1])
    return ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in values)


@app.post("/api/chat")
def answer(q: Q) -> dict:
    global _ctx, _count
    if _ctx is None:
        _ctx = Context(eng)
    _count += 1
    trace = Trace(f"chat{_count}", q.question)
    try:
        return_spec = infer_chat_return(q.question, trace)
        trace.log("chat.return", "inferred return block", return_spec)
        question = {"id": f"chat{_count}", "question": q.question, "return": return_spec}
        result = answer_question(_ctx, question, trace)
    except Exception as e:
        trace.log("error", f"unexpected {type(e).__name__}: {e}", traceback.format_exc())
        trace.abstain(f"unexpected error: {type(e).__name__}")
        result = {"answerable": False, "evidence": []}

    if not result.get("answerable"):
        append_log(CHAT_LOG_PATH, trace)
        return {"answer": f"I can't answer that confidently, so I won't guess. Reason: {trace.reason}.",
                "evidence": []}

    evidence = result.get("evidence", [])
    shown = [dict(x) for x in evidence[:MAX_EVIDENCE_SHOWN]]
    with eng.connect() as cx:
        for item in shown:
            item["doc"] = describe_evidence(cx, item)
    answer_text = format_answer(result)
    if len(evidence) > len(shown):
        answer_text += f" ({len(evidence)} evidence rows; showing the first {len(shown)})"
    append_log(CHAT_LOG_PATH, trace)
    return {"answer": answer_text, "evidence": shown}
