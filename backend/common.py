"""Shared deterministic database helpers for the final planner pipeline."""
from __future__ import annotations
import json
import re
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from backend.debug import Trace
from backend.views import VIEW_DEFINITIONS

SQL_TIMEOUT_MS = 10_000
EVIDENCE_KEYS = {
    "game_details": ("game_id",),
    "player_box_scores": ("game_id", "person_id"),
    "game_recaps": ("recap_id",),
    "injury_notes": ("note_id",),
}
MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MONTH_NAMES = "|".join(MONTHS)
ALL_STAR_BREAK_DAYS = {f"2026-02-{d}" for d in range(13, 20)}

class ToolError(Exception):
    """Invalid planner helper input."""

def load_teams(cx: Connection) -> list[dict]:
    """Load team_id, city, name, and abbreviation for every team, once per run.

    :param cx: open SQLAlchemy connection
    :return: one dict per team with keys team_id, city, name, abbreviation
    """
    sql = "SELECT team_id, city, name, abbreviation FROM teams"
    return [dict(r) for r in cx.execute(text(sql)).mappings().all()]


def abstain_result(return_spec: dict) -> dict:
    """Build a schema-valid abstention result straight from a question's `return` block.

    answers_template.json's defaults are shape placeholders (candidates fill them
    in), not a real answer -- reusing them for an abstention silently shipped
    "answerable": true with id-0 evidence, which the grader reads as a fabricated
    fact rather than a deliberate "I don't know". An abstention must always be
    answerable=false with empty/zeroed fields and no evidence.

    :param return_spec: question["return"], the declared field name -> type map
    :return: a result dict with answerable False, ints 0, strings "", evidence []
    """
    zeros = {"bool": False, "int": 0, "str": ""}
    result = {key: zeros[type_name] for key, type_name in return_spec.items() if key != "evidence"}
    result["answerable"] = False
    result["evidence"] = []
    return result


def evidence_fully_grounded(declared_shapes: list[dict], grounded: list[dict]) -> bool:
    """Check that every evidence table the question requires has a grounded row.

    A question is only answerable if every table its `return.evidence` block
    lists actually has at least one verified supporting row -- e.g. a question
    that cites both game_details and game_recaps isn't answerable off
    game_details alone.

    :param declared_shapes: question["return"]["evidence"]
    :param grounded: output of evidence_from_rows()
    :return: True if every declared table has at least one grounded row
    """
    declared_tables = {s["table"] for s in declared_shapes}
    grounded_tables = {g["table"] for g in grounded}
    return bool(declared_tables) and declared_tables.issubset(grounded_tables)


def question_dates(question: str, ctx: Context) -> tuple[set[str], set[str]]:
    """Days (YYYY-MM-DD) and months (YYYY-MM) the question mentions.

    Handles "April 12, 2026", "October 22", "January 2026" and bare "January".
    A month named without a year gets the year in which the data has games in
    that month.
    """
    q = question.lower()
    days, months = set(), set()

    def year_for(month: int, year: str | None) -> list[str]:
        if year:
            return [year]
        return sorted({m[:4] for m in ctx.data_months if int(m[5:7]) == month})

    for m in re.finditer(rf"\b({MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s*(\d{{4}}))?", q):
        month, day = MONTHS[m.group(1)], int(m.group(2))
        for y in year_for(month, m.group(3)):
            days.add(f"{y}-{month:02d}-{day:02d}")
            months.add(f"{y}-{month:02d}")
    for m in re.finditer(rf"\b({MONTH_NAMES})\b(?:\s+(\d{{4}}))?", q):
        month = MONTHS[m.group(1)]
        for y in year_for(month, m.group(2)):
            months.add(f"{y}-{month:02d}")
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", q):
        days.add(m.group(0))
        months.add(m.group(0)[:7])
    return days, months


def to_plain(value):
    """Convert database values (e.g. Decimal) to plain Python numbers."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


def resolve_player(cx: Connection, name: str, game_ids: set[int], trace: Trace) -> dict | None:
    """Match a name read from text to exactly one player who played in these games.

    Recaps usually give only a last name, sometimes with a team abbreviation,
    e.g. "Ellison-Reyes (KER)". The box score is authoritative.

    :return: {"player_name", "game_id", "person_id"} or None if not exactly one match
    """
    cleaned = re.sub(r"\(.*?\)", "", name).strip().strip(".,").lower()
    if not cleaned or not game_ids:
        return None
    sql = (
        "SELECT DISTINCT b.game_id, b.person_id, p.first_name, p.last_name "
        "FROM player_box_scores b JOIN players p ON p.player_id = b.person_id "
        "WHERE b.game_id = ANY(:games) "
        "AND (lower(p.last_name) = :name OR lower(p.first_name || ' ' || p.last_name) = :name) "
        "ORDER BY b.game_id, b.person_id"
    )
    matches = cx.execute(text(sql), {"games": sorted(game_ids), "name": cleaned}).mappings().all()
    trace.log("verify.player", f"box-score matches for {cleaned!r}", [dict(m) for m in matches])
    if len({m["person_id"] for m in matches}) != 1:
        return None
    m = matches[0]
    return {"player_name": f"{m['first_name']} {m['last_name']}", "game_id": m["game_id"], "person_id": m["person_id"]}


def evidence_from_rows(rows: list[dict], shapes: list[dict]) -> list[dict]:
    """Build evidence from the key columns of rows, in the declared shapes.

    Evidence comes from what the database returned, never from anything the model
    typed. Only tables the question declares are cited, each row once.
    """
    evidence, seen = [], set()
    for row in rows:
        for shape in shapes:
            table = shape["table"]
            keys = EVIDENCE_KEYS.get(table)
            if keys is None or any(row.get(k) is None for k in keys):
                continue
            item = {"table": table}
            for field in shape:
                if field == "table":
                    continue
                source = keys[0] if field == "id" else field
                try:
                    item[field] = int(row[source])
                except (KeyError, TypeError, ValueError):
                    break
            else:
                marker = json.dumps(item, sort_keys=True)
                if marker not in seen:
                    seen.add(marker)
                    evidence.append(item)
    return evidence


class Context:
    """Database engine plus reference data every question needs.

    On creation it (re)creates the four views the SQL prompt queries, then loads
    once: team mascots and player names (for the entity check), each view's
    columns, and which months have games (to give a year to a month named
    without one).
    """

    def __init__(self, eng: Engine):
        self.eng = eng
        with eng.begin() as cx:
            for definition in VIEW_DEFINITIONS.values():
                cx.exec_driver_sql(definition)
        with eng.connect() as cx:
            self.teams = load_teams(cx)
            self.player_display = {
                r[0].lower(): r[0]
                for r in cx.execute(text("SELECT DISTINCT CONCAT(first_name, ' ', last_name) FROM players"))
            }
            self.view_columns = {
                view: list(cx.exec_driver_sql(f"SELECT * FROM {view} LIMIT 0").keys())
                for view in VIEW_DEFINITIONS
            }
            self.data_months = {
                r[0] for r in cx.execute(text("SELECT DISTINCT LEFT(CAST(game_timestamp AS TEXT), 7) FROM game_details"))
            }
        self.mascots = {t["name"].lower(): t["name"] for t in self.teams}
        self.player_names = set(self.player_display)
        self.base_columns = {c for cols in self.view_columns.values() for c in cols}

    def detect(self, question: str) -> str:
        """The teams, players, days and months the question names, for the SQL prompt.

        Found by exact matching against the database's own team mascots and player
        names, and by parsing dates, so the model never has to convert them itself.
        """
        q = question.lower()

        def named(value: str) -> bool:
            return re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", q) is not None

        teams = [name for low, name in sorted(self.mascots.items()) if named(low)]
        players = [name for low, name in sorted(self.player_display.items()) if named(low)]
        days, months = question_dates(question, self)
        months = sorted(m for m in months if not any(d.startswith(m) for d in days))
        return (f"teams={', '.join(teams) or 'none'}; players={', '.join(players) or 'none'}; "
                f"dates={', '.join(sorted(days)) or 'none'}; months={', '.join(months) or 'none'}")


def run_logged(ctx: Context, trace: Trace, stage: str, sql: str, params: dict) -> list[dict]:
    """Run one trusted, parameterized query read-only, logging SQL, params and row count."""
    trace.log(f"{stage}.sql", "SQL statement", {"sql": sql, "params": params})
    with ctx.eng.connect() as cx:
        cx.exec_driver_sql("SET TRANSACTION READ ONLY")
        cx.exec_driver_sql(f"SET LOCAL statement_timeout = {SQL_TIMEOUT_MS}")
        rows = [{k: to_plain(v) for k, v in r.items()} for r in cx.execute(text(sql), params).mappings().all()]
        cx.rollback()
    trace.log(f"{stage}.sql_result", f"{len(rows)} rows returned")
    return rows


def normalize_month(value) -> str | None:
    """Accept "2026-01", "2026-01-15", "January 2026" or "2026-01-%"; return "YYYY-MM"."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    s = str(value).strip().lower().rstrip("%").rstrip("-")
    m = re.fullmatch(r"(\d{4})-(\d{1,2})(?:-\d{1,2})?", s)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    m = re.fullmatch(r"([a-z]+)\s+(\d{4})", s)
    if m and m.group(1) in MONTHS:
        return f"{m.group(2)}-{MONTHS[m.group(1)]:02d}"
    raise ToolError(f'month must look like "2026-01"; got {value!r}')
