import os
import yaml
import pandas as pd
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
DATA_DIR = "data/raw/t20_yaml"
INTERIM_DIR = "data/interim"

os.makedirs(INTERIM_DIR, exist_ok=True)

MATCH_ROWS = []
DELIVERY_ROWS = []

# ---------------- PARSING ---------------- #
yaml_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".yaml")]

print(f"Found {len(yaml_files)} YAML files. Starting parsing...")

for file in tqdm(yaml_files, desc="Parsing T20 YAML files"):
    file_path = os.path.join(DATA_DIR, file)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # ---------- MATCH LEVEL DATA ---------- #
        info = data.get("info", {})
        event = info.get("event", {})

        match_id = (
            event.get("match_number")
            or event.get("stage")
            or file.replace(".yaml", "")
        )

        teams = info.get("teams", [])
        outcome = info.get("outcome", {})
        winner = outcome.get("winner")

        MATCH_ROWS.append({
            "match_id": match_id,
            "team1": teams[0] if len(teams) > 0 else None,
            "team2": teams[1] if len(teams) > 1 else None,
            "winner": winner,
            "venue": info.get("venue"),
            "date": info.get("dates", [None])[0],
            "gender": info.get("gender"),
            "match_type": info.get("match_type"),
            "overs": info.get("overs")
        })

        # ---------- BALL BY BALL DATA ---------- #
        innings = data.get("innings", [])

        for inning in innings:
            for _, inning_data in inning.items():
                batting_team = inning_data.get("team")

                for delivery in inning_data.get("deliveries", []):
                    ball_number, details = list(delivery.items())[0]

                    batter = (
                        details.get("batter")
                        or details.get("batsman")
                        or details.get("striker")
                    )

                    DELIVERY_ROWS.append({
                        "match_id": match_id,
                        "batting_team": batting_team,
                        "ball": ball_number,
                        "batter": batter,
                        "bowler": details.get("bowler"),
                        "runs": details.get("runs", {}).get("batsman", 0),
                        "extras": details.get("runs", {}).get("extras", 0),
                        "is_wicket": 1 if "wicket" in details else 0
                    })

    except Exception as e:
        print(f"⚠️ Error processing file {file}: {e}")

# ---------------- SAVE CSV ---------------- #
matches_df = pd.DataFrame(MATCH_ROWS)
deliveries_df = pd.DataFrame(DELIVERY_ROWS)

matches_df.to_csv(f"{INTERIM_DIR}/matches.csv", index=False)
deliveries_df.to_csv(f"{INTERIM_DIR}/deliveries.csv", index=False)

print("✅ YAML parsing completed successfully.")
print(f"Matches saved: {len(matches_df)}")
print(f"Deliveries saved: {len(deliveries_df)}")
