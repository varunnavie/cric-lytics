import pandas as pd
import os

matches = pd.read_csv("data/interim/matches.csv")
deliveries = pd.read_csv("data/interim/deliveries.csv")

innings_scores = deliveries.groupby(
    ["match_id", "batting_team"]
)["runs"].sum().reset_index()

matches = matches.merge(innings_scores, on="match_id")

venue_stats = matches.groupby("venue").agg(
    avg_first_innings_score=("runs", "mean"),
    matches_played=("match_id", "count")
).reset_index()

os.makedirs("data/processed", exist_ok=True)
venue_stats.to_csv("data/processed/venue_features.csv", index=False)

print("✅ venue_features.csv created")
