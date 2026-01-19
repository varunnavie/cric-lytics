import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import os

# Load player data
df = pd.read_csv("data/processed/player_features.csv")

# Features & target
X = df[["balls_faced", "strike_rate", "wickets_lost"]]
y = df["total_runs"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model (Explainable + Non-linear)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ Player Runs Model MAE: {mae:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/player_runs_model.pkl")

print("✅ Player model saved at models/player_runs_model.pkl")
