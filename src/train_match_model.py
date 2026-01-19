import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("data/processed/match_features.csv")

# ---------------- SELECT FEATURES ---------------- #
features = [
    "venue_team1_win_rate",
    "h2h_team1_win_rate"
]

X = df[features]
y = df["team1_win"]

# ---------------- TRAIN-TEST SPLIT ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ---------------- #
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ---------------- #
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Match Winner Model Accuracy: {acc:.2f}")

# ---------------- SAVE MODEL ---------------- #
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/match_win_model.pkl")

print("✅ Model saved at models/match_win_model.pkl")
