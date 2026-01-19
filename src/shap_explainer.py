import shap
import pandas as pd
import joblib

# Load model & encoders
score_model = joblib.load("models/team_score_model.pkl")
team_encoder = joblib.load("models/team_encoder.pkl")
venue_encoder = joblib.load("models/venue_encoder.pkl")

# Create SHAP explainer once
explainer = shap.Explainer(score_model)

def explain_team_score(team, opponent, venue, won_toss):
    team_enc = team_encoder.transform([team])[0]
    opp_enc = team_encoder.transform([opponent])[0]
    venue_enc = venue_encoder.transform([venue])[0]

    X = pd.DataFrame([{
        "team_enc": team_enc,
        "opp_enc": opp_enc,
        "venue_enc": venue_enc,
        "won_toss": won_toss
    }])

    shap_values = explainer(X)

    return shap_values, X
