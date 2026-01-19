import pandas as pd
import os

deliveries = pd.read_csv("data/interim/deliveries.csv")
matches = pd.read_csv("data/interim/matches.csv")

# Attach opponent team
df = deliveries.merge(
    matches[["match_id", "team1", "team2"]],
    on="match_id",
    how="left"
)

df["opponent"] = df.apply(
    lambda x: x["team2"] if x["batting_team"] == x["team1"] else x["team1"],
    axis=1
)

player_opp = df.groupby(
    ["batter", "batting_team", "opponent"]
).agg(
    total_runs=("runs", "sum"),
    balls_faced=("ball", "count")
).reset_index()

player_opp["strike_rate"] = (
    player_opp["total_runs"] / player_opp["balls_faced"]
) * 100

player_opp["form_score"] = (
    0.7 * player_opp["strike_rate"] +
    0.3 * (player_opp["total_runs"] / player_opp["balls_faced"])
)

os.makedirs("data/processed", exist_ok=True)
player_opp.to_csv("data/processed/player_vs_opponent.csv", index=False)

print("✅ Player vs Opponent dataset created")
