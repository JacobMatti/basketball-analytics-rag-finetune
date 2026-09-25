"""JSON query-plan pipeline: Qwen fills in a form, Python does everything exact.

For each question:
  1. Python detects the teams, players, dates and months the question names.
  2. Qwen fills in a JSON plan (one model call; Ollama constrains it to a schema):
       source   which rows: player_games, games, recaps or notes
       filters  player, team, opponent, date, month, date_from, result, min_margin,
                starter, min_stat, game (first/last)
       select   which rows the answer comes from:
                  all        every matching row
                  top_row    the single row with the highest/lowest value of a column
                  top_group  the player whose rows have the highest/lowest total
                             (or number) of rows
       answer   for each answer field, how to compute it:
                  value  the column's value in the selected row(s)
                  sum    the total of a column over the selected rows
                  count  the number of selected rows
                  text   read it from the recap or note text (a second, small
                         model call that only extracts a phrase)
  3. Python checks the plan against the question (every name, date and number in it
     must come from the question), writes parameterized SQL, runs it read-only,
     selects rows, computes every answer, and builds and verifies the evidence.
  4. Anything that doesn't check out abstains, with the reason logged.

Qwen never writes SQL, never does arithmetic and never sees a database id.

Debugging: every prompt, raw model reply, parsed plan, check, SQL statement with
parameters, row, computation and evidence check is printed (RAG_DEBUG, default 2
here) and saved to part1/debug/planner_log.jsonl.

usage:
    python -m backend.planner              every question in part1/questions.json
    python -m backend.planner --only 1 5   just those question ids
"""
import argparse
import json
import os
import re
import time
import traceback

os.environ.setdefault("RAG_DEBUG", "2")

import jsonschema
import sqlalchemy as sa

from backend.config import DB_DSN, LLM_MODEL
from backend.debug import Trace, print_summary, write_log
from backend.common import (
    ALL_STAR_BREAK_DAYS, Context, ToolError, abstain_result, evidence_from_rows,
    evidence_fully_grounded, normalize_month, question_dates, resolve_player, run_logged,
)
from backend.utils import NUM_CTX, ollama_generate_full

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.json"))
DEBUG_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "debug"))
LOG_PATH = os.path.join(DEBUG_DIR, "planner_log.jsonl")

MAX_ATTEMPTS = 2


# The plan form's vocabulary: the row sources, filters and columns Qwen can choose from.
SOURCES = ["player_games", "games", "recaps", "notes"]
PLAYER_STATS = ["points", "rebounds", "assists", "steals", "blocks", "turnovers", "minutes",
                "offensive_reb", "defensive_reb", "fg2_made", "fg2_attempted", "fg3_made",
                "fg3_attempted", "ft_made", "ft_attempted"]
GAME_NUMBERS = ["margin", "team_points", "opponent_points"]
NUMERIC_COLUMNS = PLAYER_STATS + GAME_NUMBERS
TEXT_COLUMNS = ["player_name", "team", "opponent", "game_date", "result", "score", "winner", "status"]
ANSWER_COLUMNS = NUMERIC_COLUMNS + TEXT_COLUMNS

SOURCE_FILTERS = {
    "player_games": {"player", "team", "opponent", "date", "month", "date_from", "result",
                     "min_margin", "starter", "min_stat", "game"},
    "games": {"team", "opponent", "date", "month", "date_from", "result", "min_margin", "game"},
    "recaps": {"team", "opponent", "date", "month", "date_from", "result", "min_margin", "game"},
    "notes": {"player", "team", "opponent", "date", "month", "date_from"},
}
EMPTY_FILTER = {"player": "", "team": "", "opponent": "", "date": "", "month": "", "date_from": "",
                "result": "", "min_margin": 0, "starter": "", "game": "",
                "min_stat": {"field": "none", "value": 0}}
ID_COLUMNS = {"game_id", "person_id", "recap_id", "note_id", "game_timestamp"}


def plan_schema(return_spec: dict) -> dict:
    """JSON schema for a plan, built from the question's return block.

    Ollama uses it to constrain generation, so every value is one of the allowed
    choices and every answer field gets exactly one operation. "none" is listed
    first in every choice: a small model drifts toward the first option, and for
    boxes a question doesn't use, "none" is the right one.
    """
    fields = [k for k in return_spec if k not in ("answerable", "evidence")]
    operation = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "enum": ["value", "sum", "count", "text"]},
            "field": {"type": "string", "enum": ["none"] + ANSWER_COLUMNS},
        },
        "required": ["op", "field"],
        "additionalProperties": False,
    }
    string = {"type": "string"}
    filters = {
        "player": string, "team": string, "opponent": string, "date": string, "month": string,
        "date_from": string,
        "result": {"type": "string", "enum": ["", "W", "L"]},
        "min_margin": {"type": "integer"},
        "starter": {"type": "string", "enum": ["", "starters", "bench"]},
        "game": {"type": "string", "enum": ["", "first", "last"]},
        "min_stat": {
            "type": "object",
            "properties": {"field": {"type": "string", "enum": ["none"] + PLAYER_STATS},
                           "value": {"type": "integer"}},
            "required": ["field", "value"],
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "source": {"type": "string", "enum": SOURCES},
            "filters": {"type": "object", "properties": filters, "required": list(filters),
                        "additionalProperties": False},
            "select": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["all", "top_row", "top_group"]},
                    "by": {"type": "string", "enum": ["none", "count"] + NUMERIC_COLUMNS},
                    "order": {"type": "string", "enum": ["highest", "lowest"]},
                },
                "required": ["mode", "by", "order"],
                "additionalProperties": False,
            },
            "answer": {"type": "object", "properties": {f: operation for f in fields},
                       "required": fields, "additionalProperties": False},
        },
        "required": ["answerable", "source", "filters", "select", "answer"],
        "additionalProperties": False,
    }


# The prompt: worked examples (other teams and players) and the instructions Qwen sees.
ALL = {"mode": "all", "by": "none", "order": "highest"}

EXAMPLE_SPECS = [
    ("How many rebounds did Tyrese Holloway grab in December 2025, and in how many games?",
     ["Tyrese Holloway"], "rebounds (int), games (int)", "player_games", ALL,
     {"rebounds": {"op": "sum", "field": "rebounds"}, "games": {"op": "count", "field": "none"}},
     {"player": "Tyrese Holloway", "month": "2025-12"}),
    ("How many assists did Lamar Wolfe have against the Miners on November 5, 2025?",
     ["Lamar Wolfe"], "assists (int)", "player_games", ALL,
     {"assists": {"op": "value", "field": "assists"}},
     {"player": "Lamar Wolfe", "opponent": "Miners", "date": "2025-11-05"}),
    ("How many rebounds did Kofi Bellamy have in the Talons' season opener?",
     ["Kofi Bellamy"], "rebounds (int)", "player_games", ALL,
     {"rebounds": {"op": "value", "field": "rebounds"}},
     {"player": "Kofi Bellamy", "team": "Talons", "game": "first"}),
    ("What was the final score of the Foxes' game against the Pioneers on March 26, 2026, and who won?",
     [], "winner (str), score (str)", "games", ALL,
     {"winner": {"op": "value", "field": "winner"}, "score": {"op": "value", "field": "score"}},
     {"team": "Foxes", "opponent": "Pioneers", "date": "2026-03-26"}),
    ("What was the Admirals' biggest win of the season, who was it against, and what was the score?",
     [], "opponent (str), margin (int), score (str)", "games",
     {"mode": "top_row", "by": "margin", "order": "highest"},
     {"opponent": {"op": "value", "field": "opponent"}, "margin": {"op": "value", "field": "margin"},
      "score": {"op": "value", "field": "score"}},
     {"team": "Admirals", "result": "W"}),
    ("Who led the Admirals in rebounds against the Falcons on November 18, 2025, and how many did he have?",
     [], "player_name (str), rebounds (int)", "player_games",
     {"mode": "top_row", "by": "rebounds", "order": "highest"},
     {"player_name": {"op": "value", "field": "player_name"}, "rebounds": {"op": "value", "field": "rebounds"}},
     {"team": "Admirals", "opponent": "Falcons", "date": "2025-11-18"}),
    ("Which Talons bench player scored the most total points in games the Talons won by 15 or more?",
     [], "player_name (str), points (int)", "player_games",
     {"mode": "top_group", "by": "points", "order": "highest"},
     {"player_name": {"op": "value", "field": "player_name"}, "points": {"op": "sum", "field": "points"}},
     {"team": "Talons", "starter": "bench", "result": "W", "min_margin": 15}),
    ("Which player had the most games with 35 or more points against the Anchors?",
     [], "player_name (str), games (int)", "player_games",
     {"mode": "top_group", "by": "count", "order": "highest"},
     {"player_name": {"op": "value", "field": "player_name"}, "games": {"op": "count", "field": "none"}},
     {"opponent": "Anchors", "min_stat": {"field": "points", "value": 35}}),
    ("Why did Roman Gentry miss the Admirals' game against the Anchors on January 21, 2026?",
     ["Roman Gentry"], "status (str), reason (str)", "notes", ALL,
     {"status": {"op": "value", "field": "status"}, "reason": {"op": "text", "field": "none"}},
     {"player": "Roman Gentry", "team": "Admirals", "opponent": "Anchors", "date": "2026-01-21"}),
    ("Who hit the go-ahead basket in the Cobras' game against the Miners on November 5, 2025?",
     [], "player_name (str)", "recaps", ALL,
     {"player_name": {"op": "text", "field": "none"}},
     {"team": "Cobras", "opponent": "Miners", "date": "2025-11-05"}),
]


def examples_text(ctx: Context) -> str:
    parts = []
    for question, players, fields, source, select, answer, filters in EXAMPLE_SPECS:
        detected = detect(ctx, question, players_override=players)
        plan = {"answerable": True, "source": source, "filters": {**EMPTY_FILTER, **filters},
                "select": select, "answer": answer}
        parts.append(f"Question: {question}\nDetected: {detected_text(detected)}\n"
                     f"Answer fields: {fields}\nPlan: {json.dumps(plan)}")
    return "\n\n".join(parts)


PLAN_PROMPT_TEMPLATE = """You turn a basketball question into a JSON plan. You do not answer the question and you
never do arithmetic: the program reads your plan, fetches the rows and computes the answer.

SOURCE (which rows to fetch)
  player_games  one row per player per game: player_name, team (his team), opponent, game_date,
                result, margin, score, starter, and stats: points, rebounds, assists, steals,
                blocks, turnovers, minutes, fg3_made and others
  games         one row per team per game: team, opponent, game_date, result, margin, score,
                team_points, opponent_points, winner
  recaps        the written recap of a game (halftime score, go-ahead basket, comebacks): text
  notes         pre-game injury reports: player_name, status, text (the reason)

FILTERS (leave a filter empty: "" or 0 or "none")
  player     full player name            team       the team the question is about
  opponent   the other team ("against X") date       "YYYY-MM-DD", one specific day
  month      "YYYY-MM"                   date_from  "YYYY-MM-DD", this day or later
  result     "W" wins, "L" losses        min_margin won or lost by at least this many
  starter    "starters" or "bench"       min_stat   a stat of at least a value, e.g. 30 or more points
  game       "first" (season opener) or "last" game of the team
  "After the All-Star break" means date_from "2026-02-19".

SELECT (which of the fetched rows the answer comes from)
  all        every fetched row
  top_row    the one row with the highest (or lowest) value of "by"
  top_group  the player whose rows have the highest total of "by"; by "count" = most rows (games)

ANSWER (for EACH answer field, how the program computes it from the selected rows)
  value  that column's value         sum    the total of a column over the rows
  count  the number of rows (games)  text   read it from the recap or note text

RULES
- Use only the teams, players, dates and months on the Detected line, and only numbers in the question.
- A team marked (team) goes in the team filter; a team marked (opponent) goes in the opponent filter.
- answerable is false only if the data cannot answer it (there is no plus-minus,
  quarter-by-quarter scoring or shot location data).

EXAMPLES

{examples}

Question: {question}
Detected: {detected}
Answer fields: {fields}
Plan:"""


# Reading the question in Python: teams with their roles, players, dates and months.
def _in_question(value: str, q: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(value.lower())}(?![a-z0-9])", q) is not None

OPPONENT_SIGNAL = r"\b(?:against|vs\.?|versus|over|to|beat|beating|defeated|facing|faced|hosted|at)\s+(?:the\s+){name}\b"
TEAM_SIGNALS = [
    r"\b{name}'",
    r"\b{name}\s+(?:starter|starters|player|players|bench|guard|guards|forward|forwards|center|centers)\b",
    r"\bled\s+the\s+{name}\b",
    r"\bwhich\s+{name}\b",
]


def team_roles(question: str, ctx: Context) -> dict[str, str | None]:
    """Each team the question names, with its role ("team", "opponent") or None if unclear."""
    q = question.lower()
    roles = {}
    for low, name in sorted(ctx.mascots.items()):
        if not _in_question(low, q):
            continue
        n = re.escape(low)
        opponent = re.search(OPPONENT_SIGNAL.format(name=n), q) is not None
        team = any(re.search(sig.format(name=n), q) for sig in TEAM_SIGNALS)
        roles[name] = "opponent" if opponent and not team else "team" if team and not opponent else None
    return roles


def detect(ctx: Context, question: str, players_override: list[str] | None = None) -> dict:
    """Everything Python can read from the question: teams with roles, players, days, months."""
    q = question.lower()
    players = players_override if players_override is not None else [
        name for low, name in sorted(ctx.player_display.items()) if _in_question(low, q)]
    days, months = question_dates(question, ctx)
    months = {m for m in months if not any(d.startswith(m) for d in days)}
    return {"teams": team_roles(question, ctx), "players": players,
            "days": sorted(days), "months": sorted(months)}


def detected_text(d: dict) -> str:
    teams = ", ".join(f"{t} ({r})" if r else t for t, r in d["teams"].items()) or "none"
    return (f"teams={teams}; players={', '.join(d['players']) or 'none'}; "
            f"dates={', '.join(d['days']) or 'none'}; months={', '.join(d['months']) or 'none'}")


def build_prompt(ctx: Context, question: dict) -> str:
    spec = question["return"]
    fields = ", ".join(f"{k} ({t})" for k, t in spec.items() if k not in ("answerable", "evidence"))
    return PLAN_PROMPT_TEMPLATE.format(examples=examples_text(ctx), question=question["question"],
                                       detected=detected_text(detect(ctx, question["question"])), fields=fields)


def ask_model(trace: Trace, stage: str, prompt: str, schema: dict) -> str:
    trace.log(f"{stage}.llm_input", f"full prompt ({len(prompt.split())} words)", prompt)
    data = ollama_generate_full(LLM_MODEL, prompt, json_schema=schema)
    reply = data.get("response", "")
    read = data.get("prompt_eval_count")
    trace.log(f"{stage}.llm_stats",
              f"prompt tokens read={read}, tokens written={data.get('eval_count')}, "
              f"seconds={round((data.get('total_duration') or 0) / 1e9, 1)}")
    if read is not None and read >= NUM_CTX - 16:
        trace.log(f"{stage}.warning", f"prompt likely TRUNCATED: read {read} of {NUM_CTX} tokens")
    trace.log(f"{stage}.llm_output", "raw model reply", reply)
    return reply


# Checking a plan: repair unambiguous slips first, then reject anything that doesn't
# follow from the question.
def columns_of(source: str, ctx: Context) -> set[str]:
    cols = set(ctx.view_columns[source])
    if "result" in cols and "team" in cols:
        cols.add("winner")
    return cols


def repair_plan(plan: dict, detected: dict, ctx: Context) -> list[str]:
    """Fix unambiguous slips in place, using what Python read from the question.

    A retry makes a small model rewrite the whole plan and often breaks parts that
    were right, so anything Python can decide on its own it decides here:
      - "at least 0" of a stat filters nothing, so it becomes no filter,
      - a player, day or month the question names is locked in (only when there is
        exactly one, and the source supports that filter),
      - a team whose role the wording makes clear goes in that box,
      - "first/last game" without a team gets the one team the question is about.
    Anything ambiguous is left for check_plan to accept or reject.

    :return: a description of each repair, for the log
    """
    repairs = []
    if not isinstance(plan, dict) or not isinstance(plan.get("filters"), dict):
        return repairs
    f = plan["filters"]
    source = plan.get("source")
    supported = SOURCE_FILTERS.get(source, set())

    stat = f.get("min_stat")
    if isinstance(stat, dict) and stat.get("field") != "none" and not stat.get("value"):
        f["min_stat"] = {"field": "none", "value": 0}
        repairs.append(f"min_stat {stat.get('field')} >= 0 filters nothing: removed")
    if isinstance(f.get("min_margin"), int) and f["min_margin"] < 0:
        f["min_margin"] = 0
        repairs.append("negative min_margin removed")

    if len(detected["players"]) == 1 and "player" in supported:
        name = detected["players"][0]
        if str(f.get("player", "")).strip().lower() != name.lower():
            repairs.append(f'player "{f.get("player", "")}" -> "{name}" (named in the question)')
            f["player"] = name
    if len(detected["days"]) == 1 and "date" in supported:
        day = detected["days"][0]
        if f.get("date") != day:
            repairs.append(f'date "{f.get("date", "")}" -> "{day}" (named in the question)')
            f["date"] = day
    elif not detected["days"] and len(detected["months"]) == 1 and "month" in supported \
            and not f.get("month") and not f.get("date_from"):
        f["month"] = detected["months"][0]
        repairs.append(f'month "" -> "{f["month"]}" (named in the question)')

    for name, role in detected["teams"].items():
        if role is None:
            continue
        other = "opponent" if role == "team" else "team"
        if role in supported and f.get(role) != name:
            repairs.append(f'{role} "{f.get(role, "")}" -> "{name}" (the question makes it the {role})')
            f[role] = name
        if str(f.get(other, "")).lower() == name.lower():
            repairs.append(f'{other} "{name}" removed (the question makes it the {role})')
            f[other] = ""

    if f.get("game") and not f.get("team"):
        candidates = [t for t, r in detected["teams"].items() if r != "opponent"]
        if len(candidates) == 1:
            f["team"] = candidates[0]
            repairs.append(f'team "" -> "{candidates[0]}" (the "{f["game"]} game" filter needs a team)')
    return repairs


def check_plan(plan: dict, question: dict, schema: dict, ctx: Context) -> tuple[dict | None, str | None]:
    """Validate the plan against the schema, the question and itself.

    :return: (normalized filters, None), or (None, problem). The problem is shown
        to the model on the retry.
    """
    try:
        jsonschema.validate(plan, schema)
    except jsonschema.ValidationError as e:
        return None, f"the plan does not match the schema: {e.message}"

    source = plan["source"]
    f = {**plan["filters"]}
    q = question["question"].lower()
    days, months = question_dates(question["question"], ctx)
    question_numbers = {int(n) for n in re.findall(r"\d+", q)}
    cols = columns_of(source, ctx)

    used = {k for k, v in f.items()
            if (k == "min_stat" and v["field"] != "none") or (k == "min_margin" and v) or
            (k not in ("min_stat", "min_margin") and v)}
    unsupported = used - SOURCE_FILTERS[source]
    if unsupported:
        return None, f"the {source} source has no {', '.join(sorted(unsupported))} filter"

    for key in ("player", "team", "opponent"):
        f[key] = f[key].strip()
        if not f[key]:
            continue
        known = ctx.player_names if key == "player" else set(ctx.mascots)
        if f[key].lower() not in known:
            kind = "player" if key == "player" else "team (use the mascot, e.g. Outlaws)"
            return None, f'"{f[key]}" is not a {kind} in the data'
        if not _in_question(f[key], q):
            return None, f'the {key} "{f[key]}" is not named in the question'
        if key != "player":
            f[key] = ctx.mascots[f[key].lower()]
    for key in ("date", "date_from"):
        f[key] = f[key].strip()
        if f[key] and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", f[key]):
            return None, f'{key} must look like "2026-01-15"; got "{f[key]}"'
    if f["date"] and f["date"] not in days:
        return None, f"the date {f['date']} is not a date the question names"
    if f["date_from"] and f["date_from"] not in days and not ("break" in q and f["date_from"] in ALL_STAR_BREAK_DAYS):
        return None, f"date_from {f['date_from']} does not come from the question"
    try:
        f["month"] = normalize_month(f["month"]) or ""
    except ToolError as e:
        return None, str(e)
    if f["month"] and f["month"] not in months:
        return None, f"the month {f['month']} is not a month the question names"
    if days and f["date"] not in days:
        return None, f"the question names the date {', '.join(sorted(days))}; set the date filter"
    if months and not days and not f["month"]:
        return None, f"the question names {', '.join(sorted(months))}; set the month filter"

    if f["min_margin"] and f["min_margin"] not in question_numbers:
        return None, f"min_margin {f['min_margin']} is not a number in the question"
    if f["min_stat"]["field"] != "none" and f["min_stat"]["value"] not in question_numbers:
        return None, f"min_stat value {f['min_stat']['value']} is not a number in the question"
    if f["game"] and not f["team"]:
        return None, 'the "game" filter (first or last) needs the team filter'

    sel = plan["select"]
    if sel["mode"] == "all" and sel["by"] != "none":
        return None, 'select mode "all" takes by "none"'
    if sel["mode"] != "all":
        if sel["by"] == "none":
            return None, f'select mode "{sel["mode"]}" needs a "by" column'
        if sel["by"] == "count" and sel["mode"] != "top_group":
            return None, 'by "count" only works with mode "top_group"'
        if sel["by"] != "count" and sel["by"] not in cols:
            return None, f'the {source} source has no column "{sel["by"]}" to rank by'
        if sel["mode"] == "top_group" and "player_name" not in cols:
            return None, f"top_group groups by player, but {source} has no players"

    if sel["mode"] != "all":
        numeric_ops = [op for name, op in plan["answer"].items() if question["return"][name] == "int"]
        if numeric_ops:
            ranked_reported = any(
                (sel["by"] == "count" and op["op"] == "count") or
                (op["op"] in ("value", "sum") and op["field"] == sel["by"]) for op in numeric_ops)
            if not ranked_reported:
                return None, (f'the plan ranks by "{sel["by"]}" but the answer does not report it; '
                              "rank by the quantity the question asks about")
    for name, op in plan["answer"].items():
        kind = question["return"][name]
        if op["op"] in ("count", "text") and op["field"] != "none":
            return None, f'{name}: {op["op"]} takes field "none"'
        if op["op"] in ("value", "sum") and op["field"] == "none":
            return None, f"{name}: {op['op']} needs a column"
        if op["op"] in ("value", "sum") and op["field"] not in cols:
            return None, f'{name}: the {source} source has no column "{op["field"]}"'
        if op["op"] == "sum" and op["field"] not in NUMERIC_COLUMNS:
            return None, f"{name}: only numbers can be summed"
        if op["op"] == "sum" and sel["mode"] == "top_row":
            return None, f"{name}: top_row selects one row, so use value, not sum"
        if op["op"] == "text" and "text" not in cols:
            return None, f"{name}: text is only in the recaps and notes sources"
        if op["op"] == "count" and kind != "int":
            return None, f"{name}: count gives a number, but {name} is {kind}"
        if name in ANSWER_COLUMNS and op["op"] in ("value", "sum") and op["field"] != name:
            return None, f"{name} must come from the {name} column, not {op['field']}"
        if name not in ANSWER_COLUMNS and op["op"] in ("value", "sum"):
            return None, f"{name} is not a column: use count for a number of games, or text"
    return f, None


# Running a plan: parameterized SQL, row selection, arithmetic and text reading, all in Python.
def fetch_rows(ctx: Context, trace: Trace, source: str, f: dict) -> list[dict]:
    """Translate the plan's filters into parameterized SQL over one view and run it."""
    conditions, params = [], {}
    if f["player"]:
        conditions.append("lower(player_name) = lower(:player)")
        params["player"] = f["player"]
    for key in ("team", "opponent"):
        if f[key]:
            conditions.append(f"{key} = :{key}")
            params[key] = f[key]
    if f["date"]:
        conditions.append("game_date = :date")
        params["date"] = f["date"]
    if f["month"]:
        conditions.append("game_date LIKE :month_pattern")
        params["month_pattern"] = f"{f['month']}-%"
    if f["date_from"]:
        conditions.append("game_date >= :date_from")
        params["date_from"] = f["date_from"]
    if f["result"]:
        conditions.append("result = :result")
        params["result"] = f["result"]
    if f["min_margin"]:
        conditions.append("margin >= :min_margin")
        params["min_margin"] = f["min_margin"]
    if f["starter"]:
        conditions.append("starter = :starter")
        params["starter"] = f["starter"] == "starters"
    if f["min_stat"]["field"] != "none":
        conditions.append(f"{f['min_stat']['field']} >= :min_stat")
        params["min_stat"] = f["min_stat"]["value"]
    if f["game"] == "first":
        conditions.append("team_game_number = 1")
    elif f["game"] == "last":
        conditions.append("team_game_number = (SELECT MAX(team_game_number) FROM games WHERE team = :team)")
    keys = {"player_games": "person_id", "notes": "note_id", "recaps": "recap_id", "games": "team"}[source]
    sql = f"SELECT * FROM {source}"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += f" ORDER BY game_date, game_id, {keys}"
    rows = run_logged(ctx, trace, "plan.sql", sql, params)
    for r in rows:
        r.pop("game_timestamp", None)
        if "result" in r and "team" in r:
            r["winner"] = r["team"] if r["result"] == "W" else r["opponent"]
    trace.log("plan.rows", f"{len(rows)} rows", rows[:40])
    return rows


def select_rows(rows: list[dict], sel: dict, trace: Trace) -> tuple[list[dict] | None, str | None]:
    """Pick the rows the answer comes from. Ties at the top abstain."""
    if sel["mode"] == "all":
        return rows, None
    sign = 1 if sel["order"] == "highest" else -1
    if sel["mode"] == "top_row":
        best = max(sign * r[sel["by"]] for r in rows)
        top = [r for r in rows if sign * r[sel["by"]] == best]
        trace.log("plan.select", f"top_row by {sel['by']} ({sel['order']}): value {sign * best}, "
                                 f"{len(top)} row(s) at the top")
        if len(top) > 1:
            return None, f"a tie: {len(top)} rows share the {sel['order']} {sel['by']} ({sign * best})"
        return top, None
    groups = {}
    for r in rows:
        groups.setdefault(r["player_name"], []).append(r)
    scores = {p: (len(g) if sel["by"] == "count" else sum(r[sel["by"]] for r in g)) for p, g in groups.items()}
    ranked = sorted(scores.items(), key=lambda kv: (-sign * kv[1], kv[0]))
    trace.log("plan.select", f"top_group by {sel['by']} ({sel['order']}), per player (top 10)", ranked[:10])
    best = ranked[0][1]
    leaders = [p for p, s in ranked if s == best]
    if len(leaders) > 1:
        return None, f"a tie: {', '.join(leaders)} share the {sel['order']} {sel['by']} ({best})"
    return groups[leaders[0]], None


TEXT_PROMPT = """Read this basketball text and copy out the requested facts.

Rules:
- Copy a short phrase exactly as it appears in the text, e.g. "sprained ankle", not a whole sentence.
- For a player, copy the name as written in the text.
- If the text does not clearly say it, use an empty string "".

Question: {question}
Text: {text}
Fields to copy: {fields}
JSON:"""


def read_text(ctx: Context, trace: Trace, question: str, rows: list[dict], fields: dict) -> tuple[dict | None, list, str | None]:
    """Second model call: extract phrases from the one text the rows contain.

    :return: (values, extra evidence rows, None), or (None, [], problem)
    """
    texts = sorted({r["text"] for r in rows if r.get("text")})
    if len(texts) != 1:
        return None, [], f"reading text needs exactly one recap or note, but {len(texts)} matched"
    text = texts[0]
    schema = {"type": "object", "properties": {k: {"type": "string"} for k in fields},
              "required": list(fields), "additionalProperties": False}
    prompt = TEXT_PROMPT.format(question=question, text=text,
                                fields=", ".join(f"{k} ({t})" for k, t in fields.items()))
    reply = ask_model(trace, "text", prompt, schema)
    try:
        extracted = json.loads(reply)
    except json.JSONDecodeError:
        return None, [], "the text reply was not valid JSON"
    values, extra = {}, []
    low = text.lower()
    for name, kind in fields.items():
        value = str(extracted.get(name, "")).strip()
        if not value:
            return None, [], f"the text does not say {name}"
        if "player" in name:
            games = {int(r["game_id"]) for r in rows}
            with ctx.eng.connect() as cx:
                player = resolve_player(cx, value, games, trace)
            if player is None:
                return None, [], f"{name} '{value}' does not match exactly one player in that game's box score"
            trace.log("text.player", f"'{value}' resolved to {player['player_name']} from the box score")
            values[name] = player["player_name"]
            extra.append({"game_id": player["game_id"], "person_id": player["person_id"]})
        elif kind == "int":
            m = re.search(r"\d+", value)
            if not m or not re.search(rf"(?<!\d){m.group(0)}(?!\d)", low):
                return None, [], f"{name} '{value}' is not a number found in the text"
            values[name] = int(m.group(0))
        else:
            missing = [w for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) >= 2 and w not in low]
            if missing:
                return None, [], f"{name} '{value}' has words not in the text: {missing}"
            values[name] = value
        trace.log("text.value", f"{name} = {values[name]!r}")
    return values, extra, None


def compute_answer(ctx: Context, trace: Trace, question: dict, plan: dict,
                   used: list[dict]) -> tuple[dict | None, list, str | None]:
    """Compute every answer field from the selected rows, as the plan says."""
    spec = question["return"]
    values, extra = {}, []
    text_fields = {}
    for name, op in plan["answer"].items():
        kind = spec[name]
        if op["op"] == "text":
            text_fields[name] = kind
            continue
        if op["op"] == "count":
            value = len(used)
        elif op["op"] == "sum":
            total = sum(r[op["field"]] for r in used)
            value = round(total, 1) if isinstance(total, float) else total
        else:
            distinct = {json.dumps(r[op["field"]], default=str) for r in used}
            if len(distinct) != 1:
                return None, [], f"{name}: the selected rows have {len(distinct)} different {op['field']} values"
            value = used[0][op["field"]]
        if kind == "int":
            value = int(round(value))
            if value < 0:
                return None, [], f"{name} is negative ({value})"
        values[name] = value
        inputs = [r[op["field"]] for r in used] if op["op"] in ("sum", "value") else None
        trace.log("plan.compute", f"{name} = {op['op']}({op['field']}) over {len(used)} rows = {value!r}",
                  {"inputs": inputs})
    if text_fields:
        extracted, extra, problem = read_text(ctx, trace, question["question"], used, text_fields)
        if extracted is None:
            return None, [], problem
        values.update(extracted)
    return values, extra, None


# Evidence comes only from the database keys of the rows used, and each row is checked to exist.
def build_evidence(ctx: Context, trace: Trace, rows: list[dict], shapes: list[dict]) -> tuple[list | None, str | None]:
    """Evidence from the selected rows' database keys, each checked to exist."""
    evidence = evidence_from_rows(rows, shapes)
    trace.log("evidence.built", f"{len(evidence)} evidence entries built from {len(rows)} rows", evidence)
    games = {e["id"] for e in evidence if e["table"] == "game_details"}
    checks = {
        "game_details": ("SELECT 1 AS ok FROM game_details WHERE game_id = :id", lambda e: {"id": e["id"]}),
        "player_box_scores": ("SELECT 1 AS ok FROM player_box_scores WHERE game_id = :game_id AND person_id = :person_id",
                              lambda e: {"game_id": e["game_id"], "person_id": e["person_id"]}),
        "game_recaps": ("SELECT game_id FROM game_recaps WHERE recap_id = :id", lambda e: {"id": e["id"]}),
        "injury_notes": ("SELECT game_id FROM injury_notes WHERE note_id = :id", lambda e: {"id": e["id"]}),
    }
    for e in evidence:
        sql, params = checks[e["table"]]
        found = run_logged(ctx, trace, "evidence.check", sql, params(e))
        if not found:
            return None, f"evidence row {e} does not exist in the database"
        if e["table"] in ("game_recaps", "injury_notes") and found[0]["game_id"] not in games:
            return None, f"{e['table']} row {e['id']} is not cited together with its game"
    if not evidence_fully_grounded(shapes, evidence):
        have = sorted({e["table"] for e in evidence})
        return None, f"evidence covers {have} but the question needs {[s['table'] for s in shapes]}"
    return evidence, None


# One question end to end: plan (with one retry), fetch, select, compute, build evidence.
# Any failed step abstains.
def answer_question(ctx: Context, question: dict, trace: Trace) -> dict:
    spec = question["return"]
    schema = plan_schema(spec)
    trace.log("plan.schema", "JSON schema given to Ollama to constrain the reply", schema, full=True)
    detected = detect(ctx, question["question"])
    trace.log("plan.detected", f"detected in the question: {detected_text(detected)}")
    prompt = build_prompt(ctx, question)

    plan, filters, problem = None, None, None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        reply = ask_model(trace, f"attempt{attempt}", prompt, schema)
        try:
            plan = json.loads(reply)
        except json.JSONDecodeError:
            plan, problem = None, "the reply was not valid JSON"
        else:
            trace.log(f"attempt{attempt}.plan", "parsed plan", plan)
            if isinstance(plan, dict) and plan.get("answerable") is False:
                trace.abstain("the model's plan says the data cannot answer the question")
                return abstain_result(spec)
            repairs = repair_plan(plan, detected, ctx) if isinstance(plan, dict) else []
            if repairs:
                trace.log(f"attempt{attempt}.repair", f"{len(repairs)} repair(s) made by Python", repairs)
                trace.log(f"attempt{attempt}.repaired_plan", "plan after repairs", plan)
            filters, problem = (check_plan(plan, question, schema, ctx) if isinstance(plan, dict)
                                else (None, "the reply was not a JSON object"))
        if filters is not None:
            trace.log(f"attempt{attempt}.check", "plan passed every check", filters)
            break
        trace.log(f"attempt{attempt}.check", f"plan rejected: {problem}")
        prompt = (f"{build_prompt(ctx, question)} {reply.strip()}\n\n"
                  f"That plan was rejected: {problem}\nWrite a corrected plan.\nPlan:")
    if filters is None:
        trace.abstain(f"no valid plan after {MAX_ATTEMPTS} attempts (last problem: {problem})")
        return abstain_result(spec)

    rows = fetch_rows(ctx, trace, plan["source"], filters)
    if not rows:
        trace.abstain("no rows matched the plan's filters (the data has nothing to answer with)")
        return abstain_result(spec)

    used, problem = select_rows(rows, plan["select"], trace)
    if used is None:
        trace.abstain(problem)
        return abstain_result(spec)

    values, extra_rows, problem = compute_answer(ctx, trace, question, plan, used)
    if values is None:
        trace.abstain(problem)
        return abstain_result(spec)

    evidence, problem = build_evidence(ctx, trace, used + extra_rows, spec.get("evidence", []))
    if evidence is None:
        trace.abstain(problem)
        return abstain_result(spec)

    result = {"answerable": True, **values, "evidence": evidence}
    result = {key: result[key] for key in spec}
    trace.answered(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer questions with the JSON-plan pipeline.")
    parser.add_argument("--only", type=int, nargs="*", help="question ids to run (default: all)")
    args = parser.parse_args()

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]
    print(f"JSON-plan pipeline: {len(questions)} question(s)")
    started = time.time()
    ctx = Context(sa.create_engine(DB_DSN))

    outputs, traces = [], []
    for q in questions:
        trace = Trace(f"Q{q['id']}", q["question"])
        trace.log("start", "return block", q["return"])
        try:
            result = answer_question(ctx, q, trace)
        except Exception as e:
            trace.log("error", f"unexpected {type(e).__name__}: {e}", traceback.format_exc())
            trace.abstain(f"unexpected error: {type(e).__name__}")
            result = abstain_result(q["return"])
        traces.append(trace)
        outputs.append({"id": q["id"], "result": result})
        print(f"  -> {json.dumps(result)[:300]}")

    if args.only:
        path = os.path.join(DEBUG_DIR, "planner_partial_answers.json")
        print(f"\n--only was used, so part1/answers.json is not overwritten; results: {path}")
    else:
        path = ANSWERS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    write_log(LOG_PATH, traces)
    print_summary(traces)
    print(f"Total time: {round(time.time() - started, 1)}s")
    print(f"Answers written to {path}")
    print(f"Full log (prompts, replies, plans, SQL, rows, checks): {LOG_PATH}")


if __name__ == "__main__":
    main()
