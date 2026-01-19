import pandas as pd
import os

matches = pd.read_csv("data/interim/matches.csv")
deliveries = pd.read_csv("data/interim/deliveries.csv")

# Aggregate team scores per match
team_scores = (
    deliveries.groupby(["match_id", "batting_team"])["runs"]
    .sum()
    .reset_index()
    .rename(columns={"batting_team": "team", "runs": "team_score"})
)

df = matches.merge(team_scores, on="match_id", how="inner")

# Identify opponent
df["opponent"] = df.apply(
    lambda x: x["team2"] if x["team"] == x["team1"] else x["team1"],
    axis=1
)

# Toss features
df["won_toss"] = (df["team"] == df["winner"]).astype(int)

# Batting order (simplified heuristic)
df["batting_first"] = df.groupby("match_id").cumcount() == 0
df["batting_first"] = df["batting_first"].astype(int)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/team_score_training.csv", index=False)

print("✅ Team score training dataset with toss decision created")
