"""Flat read-only views used by the final semantic planner."""

_TEAM_GAMES = """WITH team_games AS (
  SELECT g.game_id, g.game_timestamp, t.team_id,
         t.name AS team, o.name AS opponent,
         LEFT(CAST(g.game_timestamp AS TEXT), 10) AS game_date,
         CASE WHEN g.winning_team_id = t.team_id THEN 'W' ELSE 'L' END AS result,
         CASE WHEN g.home_team_id = t.team_id THEN g.home_points ELSE g.away_points END AS team_points,
         CASE WHEN g.home_team_id = t.team_id THEN g.away_points ELSE g.home_points END AS opponent_points,
         ABS(g.home_points - g.away_points) AS margin,
         CONCAT(GREATEST(g.home_points, g.away_points), '-', LEAST(g.home_points, g.away_points)) AS score,
         ROW_NUMBER() OVER (PARTITION BY t.team_id ORDER BY g.game_timestamp, g.game_id) AS team_game_number,
         COUNT(*) OVER (PARTITION BY t.team_id) AS team_game_count
  FROM game_details g
  JOIN teams t ON t.team_id IN (g.home_team_id, g.away_team_id)
  JOIN teams o ON o.team_id IN (g.home_team_id, g.away_team_id) AND o.team_id <> t.team_id
)"""

VIEW_DEFINITIONS = {
    "games": f"""CREATE OR REPLACE VIEW games AS {_TEAM_GAMES}
SELECT game_id, game_timestamp, team, opponent, game_date, result, team_points,
       opponent_points, margin, score, team_game_number, team_game_count
FROM team_games""",
    "player_games": f"""CREATE OR REPLACE VIEW player_games AS {_TEAM_GAMES}
SELECT tg.game_id, b.person_id, tg.game_timestamp,
       CONCAT(p.first_name, ' ', p.last_name) AS player_name, tg.team, tg.opponent, tg.game_date,
       tg.result, tg.margin, tg.score, tg.team_game_number, b.starter,
       ROUND(CAST(b.seconds / 60 AS NUMERIC), 1) AS minutes, b.points,
       b.offensive_reb + b.defensive_reb AS rebounds, b.offensive_reb, b.defensive_reb,
       b.assists, b.steals, b.blocks, b.turnovers, b.fg2_made, b.fg2_attempted,
       b.fg3_made, b.fg3_attempted, b.ft_made, b.ft_attempted
FROM team_games tg
JOIN player_box_scores b ON b.game_id = tg.game_id AND b.team_id = tg.team_id
JOIN players p ON p.player_id = b.person_id""",
    "recaps": f"""CREATE OR REPLACE VIEW recaps AS {_TEAM_GAMES}
SELECT tg.game_id, r.recap_id, tg.game_timestamp, tg.team, tg.opponent, tg.game_date,
       tg.result, tg.margin, tg.score, tg.team_game_number, r.text
FROM team_games tg JOIN game_recaps r ON r.game_id = tg.game_id""",
    "notes": f"""CREATE OR REPLACE VIEW notes AS {_TEAM_GAMES}
SELECT tg.game_id, n.note_id, tg.game_timestamp,
       CONCAT(p.first_name, ' ', p.last_name) AS player_name, tg.team, tg.opponent, tg.game_date,
       n.status, n.text
FROM team_games tg
JOIN injury_notes n ON n.game_id = tg.game_id AND n.team_id = tg.team_id
JOIN players p ON p.player_id = n.player_id""",
}
