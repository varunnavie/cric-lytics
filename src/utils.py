import joblib
import pandas as pd

match_model = joblib.load("models/match_win_model.pkl")
score_model = joblib.load("models/team_score_model.pkl")
player_model = joblib.load("models/player_runs_model.pkl")

team_encoder = joblib.load("models/team_encoder.pkl")
venue_encoder = joblib.load("models/venue_encoder.pkl")

teams_df = pd.read_csv("data/processed/team_features.csv")
matches_df = pd.read_csv("data/processed/match_features.csv")
players_df = pd.read_csv("data/processed/player_clusters.csv")
player_opp_df = pd.read_csv("data/processed/player_vs_opponent.csv")

def predict_team_score(team, opponent, venue, toss_winner, toss_decision):
    try:
        team_enc = team_encoder.transform([team])[0]
        opp_enc = team_encoder.transform([opponent])[0]
        venue_enc = venue_encoder.transform([venue])[0]
    except:
        return 140

    won_toss = 1 if team == toss_winner else 0

    # batting_first logic
    if toss_winner == team:
        batting_first = 1 if toss_decision == "bat" else 0
    else:
        batting_first = 0 if toss_decision == "bat" else 1

    X = [[team_enc, opp_enc, venue_enc, won_toss, batting_first]]
    return int(score_model.predict(X)[0])


def predict_match_winner(team1, team2, venue):
    venue_rate = matches_df[matches_df["venue"] == venue]["venue_team1_win_rate"].mean()
    h2h = matches_df[
        (matches_df["team1"] == team1) & (matches_df["team2"] == team2)
    ]["h2h_team1_win_rate"].mean()

    venue_rate = venue_rate if not pd.isna(venue_rate) else 0.5
    h2h = h2h if not pd.isna(h2h) else 0.5

    prob = match_model.predict_proba([[venue_rate, h2h]])[0][1]
    return (team1 if prob >= 0.5 else team2), prob

'''def suggest_playing_xi(team, opponent=None):
    team_players = players_df[players_df["batting_team"] == team].copy()

    if opponent:
        # Simple opponent-adjustment heuristic
        opponent_factor = (
            matches_df[
                (matches_df["team1"] == team) & (matches_df["team2"] == opponent)
            ]["h2h_team1_win_rate"].mean()
        )

        opponent_factor = opponent_factor if not pd.isna(opponent_factor) else 0.5

        team_players["adjusted_form"] = (
            team_players["form_score"] * (0.8 + opponent_factor)
        )
    else:
        team_players["adjusted_form"] = team_players["form_score"]

    xi = []
    role_plan = {
        "Opener": 2,
        "Middle-order": 3,
        "Anchor": 2,
        "Finisher": 2,
        "Bowler": 2
    }

    for role, count in role_plan.items():
        group = team_players[team_players["role"] == role]
        selected = group.sort_values(
            ["adjusted_form", "strike_rate"],
            ascending=False
        ).head(count)

        xi.extend(
            selected[["batter", "role"]].to_dict("records")
        )

    return xi[:11]
player_opp_df = pd.read_csv("data/processed/player_vs_opponent.csv")

def suggest_playing_xi(team, opponent):
    # Filter player vs opponent stats
    opp_players = player_opp_df[
        (player_opp_df["batting_team"] == team) &
        (player_opp_df["opponent"] == opponent)
    ]

    # If no data → fallback to overall clusters
    if opp_players.empty:
        base_players = players_df[players_df["batting_team"] == team].copy()
        base_players["adjusted_form"] = base_players["form_score"]
    else:
        base_players = players_df.merge(
            opp_players[["batter", "form_score"]],
            on="batter",
            how="left",
            suffixes=("", "_opp")
        )

        base_players["adjusted_form"] = base_players["form_score_opp"].fillna(
            base_players["form_score"] * 0.85
        )

    xi = []
    role_plan = {
        "Opener": 2,
        "Middle-order": 3,
        "Anchor": 2,
        "Finisher": 2,
        "Bowler": 2
    }

    for role, count in role_plan.items():
        group = base_players[base_players["role"] == role]
        selected = group.sort_values(
            ["adjusted_form", "strike_rate"],
            ascending=False
        ).head(count)

        xi.extend(
            selected[["batter", "role", "adjusted_form"]]
            .to_dict("records")
        )

    return xi[:11]'''
def suggest_playing_xi(team, opponent):
    team_players = players_df[
        players_df["batting_team"] == team
    ].copy()

    opp_perf = player_opp_df[
        (player_opp_df["batting_team"] == team) &
        (player_opp_df["opponent"] == opponent)
    ]

    team_players = team_players.merge(
        opp_perf[["batter", "batting_team", "form_score"]],
        on=["batter", "batting_team"],
        how="left",
        suffixes=("", "_opp")
    )

    team_players["adjusted_form"] = team_players["form_score_opp"].fillna(
        team_players["form_score"] * 0.85
    )

    role_plan = {
        "Opener": 2,
        "Middle-order": 3,
        "Anchor": 2,
        "Finisher": 2,
        "Bowler": 2
    }

    xi = []
    for role, count in role_plan.items():
        selected = (
            team_players[team_players["role"] == role]
            .sort_values(
                ["adjusted_form", "strike_rate"],
                ascending=False
            )
            .head(count)
        )

        xi.extend(
            selected[["batter", "role", "adjusted_form"]]
            .to_dict("records")
        )

    return xi[:11]


venue_df = pd.read_csv("data/processed/venue_features.csv")

def get_venue_base_score(venue):
    row = venue_df[venue_df["venue"] == venue]
    if row.empty:
        return 150
    return int(row["avg_first_innings_score"].values[0])

