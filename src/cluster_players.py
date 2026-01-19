import pandas as pd
from sklearn.cluster import KMeans
import os

df = pd.read_csv("data/processed/player_features.csv")
clusters = []

def assign_role(row):
    if row["strike_rate"] > 140 and row["balls_faced"] > 200:
        return "Opener"
    if row["strike_rate"] > 150 and row["balls_faced"] < 150:
        return "Finisher"
    if row["wickets_lost"] > 25:
        return "Bowler"
    if row["strike_rate"] < 90:
        return "Anchor"
    return "Middle-order"

for team in df["batting_team"].unique():
    team_df = df[df["batting_team"] == team].copy()

    if len(team_df) < 11:
        continue

    X = team_df[["strike_rate", "form_score", "balls_faced"]]

    kmeans = KMeans(n_clusters=4, random_state=42)
    team_df["cluster"] = kmeans.fit_predict(X)

    team_df["role"] = team_df.apply(assign_role, axis=1)

    clusters.append(team_df)

final_df = pd.concat(clusters)
os.makedirs("data/processed", exist_ok=True)
final_df.to_csv("data/processed/player_clusters.csv", index=False)

print("✅ Player clustering + role assignment completed")
