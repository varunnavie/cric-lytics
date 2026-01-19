import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.shap_explainer import explain_team_score
from src.predict import predict_first_innings_score
from src.chase_logic import (
    chase_success_probability,
    predict_chasing_score
)
from src.xi_selection import suggest_playing_xi

# ---------------- LOAD DATA ---------------- #
team_df = pd.read_csv("data/processed/team_features.csv")
venue_df = pd.read_csv("data/processed/venue_features.csv")
player_df = pd.read_csv("data/processed/player_clusters.csv")

st.set_page_config(layout="wide")
st.title("🏏 Cricket Match Prediction Dashboard")

# ================= INPUT SECTION ================= #
with st.container():
    st.subheader("🎯 Match Setup")

    col1, col2, col3 = st.columns(3)

    with col1:
        team1 = st.selectbox("Team A", team_df["team"].unique())
        toss_winner = st.selectbox("Toss Winner", [team1])

    with col2:
        team2 = st.selectbox("Team B", team_df["team"].unique())
        toss_decision = st.selectbox("Toss Decision", ["Bat", "Bowl"])

    with col3:
        venue = st.selectbox("Venue", venue_df["venue"].unique())

# ================= PREDICTION ================= #
if st.button("🔮 Predict Match", use_container_width=True):

    # -------- Batting Order -------- #
    if toss_decision == "Bat":
        first = toss_winner
        second = team2 if toss_winner == team1 else team1
    else:
        second = toss_winner
        first = team2 if toss_winner == team1 else team1

    # -------- Score Predictions -------- #
    first_score = predict_first_innings_score(
        first, second, venue,
        team_df, venue_df,
        toss_winner, toss_decision
    )

    venue_avg = venue_df[
        venue_df["venue"] == venue
    ]["avg_first_innings_score"].values[0]

    strengths = dict(
        zip(team_df["team"], team_df["avg_runs_per_match"] / 200)
    )

    chase_prob = chase_success_probability(
        first_score,
        venue_avg,
        strengths[second]
    )

    second_score = predict_chasing_score(
        first_score,
        venue_avg,
        strengths[second]
    )

    winner = second if chase_prob > 0.5 else first
    win_prob = chase_prob if winner == second else 1 - chase_prob

    # ================= RESULTS ================= #
    st.markdown("---")
    st.subheader("📊 Match Prediction Summary")

    r1, r2, r3 = st.columns(3)

    r1.metric(f"{first} Score", first_score)
    r2.metric(f"{second} Expected Score", second_score)
    r3.metric("Predicted Winner", winner, f"{win_prob*100:.1f}%")

    # ================= PLAYING XI ================= #
    st.markdown("---")
    st.subheader("🧢 Suggested Playing XIs")

    xi1, xi2 = st.columns(2)

    with xi1:
        st.markdown(f"### {team1}")
        st.dataframe(
            suggest_playing_xi(team1, team2, player_df),
            use_container_width=True
        )

    with xi2:
        st.markdown(f"### {team2}")
        st.dataframe(
            suggest_playing_xi(team2, team1, player_df),
            use_container_width=True
        )

    # ================= SHAP EXPLAINABILITY ================= #
    st.markdown("---")
    with st.expander("🧠 AI Explainability (Why this score?)", expanded=False):

        shap_values, X_explain = explain_team_score(
            first,
            second,
            venue,
            1 if toss_winner == first else 0
        )

        st.write("🔍 **Feature contribution to predicted first innings score**")

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
