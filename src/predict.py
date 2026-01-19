import joblib
import pandas as pd

model = joblib.load("models/team_score_model.pkl")
team_encoder = joblib.load("models/team_encoder.pkl")
venue_encoder = joblib.load("models/venue_encoder.pkl")

def predict_first_innings_score(team, opponent, venue, team_df, venue_df,
                                toss_winner, toss_decision):

    team_row = team_df[team_df["team"] == team]
    venue_row = venue_df[venue_df["venue"] == venue]

    base_score = team_row["avg_runs_per_match"].values[0]
    venue_avg = venue_row["avg_first_innings_score"].values[0]

    team_enc = team_encoder.transform([team])[0]
    opp_enc = team_encoder.transform([opponent])[0]
    venue_enc = venue_encoder.transform([venue])[0]

    won_toss = int(team == toss_winner)

    if toss_winner == team:
        batting_first = int(toss_decision.lower() == "bat")
    else:
        batting_first = int(toss_decision.lower() == "bowl")

    X = [[team_enc, opp_enc, venue_enc, won_toss, batting_first]]

    predicted = model.predict(X)[0]

    return int(0.6 * predicted + 0.4 * venue_avg)
