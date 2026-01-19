import pandas as pd
import os

INTERIM_DIR = "data/interim"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

matches = pd.read_csv(f"{INTERIM_DIR}/matches.csv")
deliveries = pd.read_csv(f"{INTERIM_DIR}/deliveries.csv")

matches = matches.dropna(subset=["winner", "team1", "team2", "venue"])
matches["team1_win"] = (matches["winner"] == matches["team1"]).astype(int)

# ---------------- MATCH FEATURES ---------------- #
venue_bias = matches.groupby("venue")["team1_win"].mean().reset_index()
venue_bias.rename(columns={"team1_win": "venue_team1_win_rate"}, inplace=True)
matches = matches.merge(venue_bias, on="venue", how="left")

h2h = matches.groupby(["team1", "team2"])["team1_win"].mean().reset_index()
h2h.rename(columns={"team1_win": "h2h_team1_win_rate"}, inplace=True)
matches = matches.merge(h2h, on=["team1", "team2"], how="left")
matches["h2h_team1_win_rate"].fillna(0.5, inplace=True)

matches.to_csv(f"{PROCESSED_DIR}/match_features.csv", index=False)

# ---------------- TEAM FEATURES ---------------- #
team_stats = deliveries.groupby("batting_team").agg(
    total_runs=("runs", "sum"),
    total_wickets=("is_wicket", "sum"),
    matches_played=("match_id", "nunique")
).reset_index().rename(columns={"batting_team": "team"})

team_stats["avg_runs_per_match"] = team_stats["total_runs"] / team_stats["matches_played"]
team_stats["avg_wickets_lost"] = team_stats["total_wickets"] / team_stats["matches_played"]

team_stats.to_csv(f"{PROCESSED_DIR}/team_features.csv", index=False)

# ---------------- PLAYER FEATURES (TEAM-AWARE) ---------------- #
player_stats = deliveries.groupby(
    ["batter", "batting_team"]
).agg(
    total_runs=("runs", "sum"),
    balls_faced=("ball", "count"),
    wickets_lost=("is_wicket", "sum")
).reset_index()

player_stats = player_stats.dropna(subset=["batter"])

player_stats["strike_rate"] = (
    player_stats["total_runs"] / player_stats["balls_faced"]
) * 100

player_stats["form_score"] = (
    0.6 * player_stats["strike_rate"] +
    0.4 * (player_stats["total_runs"] / player_stats["balls_faced"])
)

player_stats.to_csv(f"{PROCESSED_DIR}/player_features.csv", index=False)

print("✅ Feature engineering completed successfully.")
