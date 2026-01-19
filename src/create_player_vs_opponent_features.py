import pandas as pd
import os

deliveries = pd.read_csv("data/interim/deliveries.csv")
matches = pd.read_csv("data/interim/matches.csv")

df = deliveries.merge(matches[["match_id", "team1", "team2"]], on="match_id")

df["opponent"] = df.apply(
    lambda x: x["team2"] if x["batting_team"] == x["team1"] else x["team1"],
    axis=1
)

player_opp = df.groupby(
    ["batter", "batting_team", "opponent"]
).agg(
    runs=("runs", "sum"),
    balls=("ball", "count"),
    wickets=("is_wicket", "sum")
).reset_index()

player_opp["strike_rate"] = (player_opp["runs"] / player_opp["balls"]) * 100
player_opp["form_score"] = 0.6 * player_opp["strike_rate"] + 0.4 * (
    player_opp["runs"] / player_opp["balls"]
)

os.makedirs("data/processed", exist_ok=True)
player_opp.to_csv("data/processed/player_vs_opponent.csv", index=False)

print("✅ Player vs Opponent features created")
