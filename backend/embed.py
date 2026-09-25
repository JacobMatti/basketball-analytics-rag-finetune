"""Build and embed natural-language documents for game_details.

Generates one document per row of game_details (joining in team names from the
teams table so the text reads naturally, e.g. "Spokane Outlaws (SPO)" instead of
a bare team_id), embeds the documents in batches with nomic-embed-text, and
stores both the document text and its embedding vector back on game_details.

usage:
    python -m backend.embed [--limit N]

    --limit N   Only embed the first N games (ordered by game_timestamp desc,
                game_id desc), for a fast smoke run instead of embedding the
                whole season.
"""
import argparse

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

from backend.config import DB_DSN, EMBED_MODEL
from backend.utils import ollama_embed_batch

# nomic-embed-text is an asymmetric model: text being indexed gets the
# "search_document: " prefix and text being searched gets "search_query: ".
# Ollama does not add these itself, so each side of retrieval adds its own.
DOCUMENT_PREFIX = "search_document: "


def build_game_doc(row: pd.Series) -> str:
    """Build a natural-language document for one game_details row.

    Joins in home/away team city+name+abbreviation (from the teams table, already
    merged onto `row` by the caller) so the document reads like a sentence a
    human would write, rather than raw foreign keys. This is what game retrieval
    in rag.py and server.py searches against.

    :param row: one row of game_details left-joined with teams for home and away
        (expects home_team_label, away_team_label, home_points, away_points,
        winning_team_id, home_team_id, game_timestamp, game_id)
    :return: one-paragraph natural-language description of the game
    """
    date = pd.to_datetime(row.game_timestamp, utc=True).strftime("%Y-%m-%d")
    home_points = int(row.home_points)
    away_points = int(row.away_points)
    if row.winning_team_id == row.home_team_id:
        winner_label, winner_points = row.home_team_label, home_points
        loser_label, loser_points = row.away_team_label, away_points
    else:
        winner_label, winner_points = row.away_team_label, away_points
        loser_label, loser_points = row.home_team_label, home_points

    return (
        f"On {date}, {row.home_team_label} hosted {row.away_team_label}. "
        f"Final score: {winner_label} {winner_points}, {loser_label} {loser_points}. "
        f"Winner: {winner_label}. (game_id={int(row.game_id)})"
    )


def main(limit: int | None) -> None:
    """Embed game_details documents and store them with their vectors.

    :param limit: if given, only embed this many games (most recent first),
        for a quick smoke test instead of a full run.
    """
    print("Starting Embedding Process")
    eng = sa.create_engine(DB_DSN)
    with eng.begin() as cx:
        cx.execute(text("ALTER DATABASE nba REFRESH COLLATION VERSION"))
        # `doc` holds the human-readable text so it can be inspected/debugged
        # directly in SQL; `embedding` is the vector derived from it.
        cx.execute(text("ALTER TABLE IF EXISTS game_details ADD COLUMN IF NOT EXISTS doc text;"))
        cx.execute(text("ALTER TABLE IF EXISTS game_details ADD COLUMN IF NOT EXISTS embedding vector(768);"))
        # No vector index here on purpose. game_details has under 1k rows and
        # even player_box_scores is only ~18k, so an exact cosine scan (what
        # rag.py's ORDER BY does without an index) costs milliseconds -- there's
        # no performance reason to add one. An approximate index (hnsw/ivfflat) would be actively
        # harmful for grading: pgvector's hnsw assigns node levels randomly
        # during construction, so a fresh embed builds a differently-shaped
        # graph every time, which can flip which rows tie for the last
        # top-k slot and make retrieve() disagree with the committed
        # part1/answers.json. If you add an index anyway, make sure
        # retrieval still returns results in the same order on every build
        # (e.g. keep an explicit tie-breaker in ORDER BY, as rag.py does).

        query = (
            "SELECT g.game_id, g.game_timestamp, g.home_team_id, g.away_team_id, "
            "g.home_points, g.away_points, g.winning_team_id, "
            "home.city || ' ' || home.name || ' (' || home.abbreviation || ')' AS home_team_label, "
            "away.city || ' ' || away.name || ' (' || away.abbreviation || ')' AS away_team_label "
            "FROM game_details g "
            "JOIN teams home ON home.team_id = g.home_team_id "
            "JOIN teams away ON away.team_id = g.away_team_id "
            "ORDER BY g.game_timestamp DESC, g.game_id DESC"
        )
        if limit is not None:
            query += " LIMIT :limit"
            df = pd.read_sql(text(query), cx, params={"limit": limit})
        else:
            df = pd.read_sql(text(query), cx)

        docs = [build_game_doc(r) for _, r in df.iterrows()]
        prefixed = [DOCUMENT_PREFIX + d for d in docs]
        vectors = ollama_embed_batch(EMBED_MODEL, prefixed)

        for game_id, doc, vec in zip(df.game_id, docs, vectors):
            cx.execute(
                text("UPDATE game_details SET doc = :doc, embedding = :v WHERE game_id = :gid"),
                {"doc": doc, "v": vec, "gid": int(game_id)},
            )
    print(f"Finished Embeddings: {len(df)} Rows Updated")


# ---------------------------------------------------------------------------
# TODO (candidate work): player_box_scores documents.
#
# Build one document per player_box_scores row (or per player-game) that reads
# naturally: player name (join players), their team and the opponent (join
# teams via game_details), the game date, and their stat line (points,
# rebounds, assists, etc.). Embed and store the same way as game_details
# (a `doc` + `embedding vector(768)` column, batched with ollama_embed_batch
# and the "search_document: " prefix; no vector index needed, see the note
# above main()). This is what lets
# rag.py answer player-specific questions like "how many points did X score
# in game Y" instead of only whole-game summaries.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (candidate work): game_recaps chunks.
#
# game_recaps.text holds a full recap per game. Decide how to chunk it (whole
# recap as one document is probably fine given the length here, but consider
# sentence/paragraph chunking if recaps get longer), embed each chunk, and
# store doc + embedding (+ a chunk index if you split). Recaps carry narrative
# detail (who hit the go-ahead shot, largest deficit overcome, etc.) that
# never shows up in the box score, so several of the harder questions in
# part1/questions.json are only answerable once this retrieval path exists.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TODO (candidate work): injury_notes rows.
#
# Each injury_notes row is already a short natural-language note; embed
# note.text directly (optionally prefixed with player/team/date context pulled
# via joins so retrieval isn't relying on the note text alone) and store
# doc + embedding. This is the retrieval path questions about why a player
# missed a game depend on.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed game_details documents into pgvector.")
    parser.add_argument("--limit", type=int, default=None, help="Only embed the first N games, for a quick smoke run.")
    args = parser.parse_args()
    main(args.limit)
