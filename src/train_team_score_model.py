import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib
import os

df = pd.read_csv("data/processed/team_score_training.csv")

le_team = LabelEncoder()
le_venue = LabelEncoder()

df["team_enc"] = le_team.fit_transform(df["team"])
df["opp_enc"] = le_team.transform(df["opponent"])
df["venue_enc"] = le_venue.fit_transform(df["venue"])

X = df[
    ["team_enc", "opp_enc", "venue_enc", "won_toss", "batting_first"]
]
y = df["team_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"✅ Team Score Model MAE: {mean_absolute_error(y_test, preds):.2f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/team_score_model.pkl")
joblib.dump(le_team, "models/team_encoder.pkl")
joblib.dump(le_venue, "models/venue_encoder.pkl")
