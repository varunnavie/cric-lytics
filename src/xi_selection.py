import pandas as pd   # ✅ ADD THIS

def suggest_playing_xi(team, opponent, df):
    df = df[df["batting_team"] == team].copy()

    df["selection_score"] = (
        0.6 * df["form_score"] +
        0.4 * df["strike_rate"]
    )

    xi = []

    role_plan = {
        "Opener": 2,
        "Middle-order": 3,
        "Anchor": 2,
        "Finisher": 2,
        "Bowler": 2
    }

    for role, n in role_plan.items():
        players = (
            df[df["role"] == role]
            .sort_values("selection_score", ascending=False)
            .head(n)
        )
        xi.append(players)

    xi_df = pd.concat(xi).drop_duplicates(subset="batter")

    # 🔒 SAFETY: Always return 11 players
    if len(xi_df) < 11:
        remaining = (
            df[~df["batter"].isin(xi_df["batter"])]
            .sort_values("selection_score", ascending=False)
        )
        xi_df = pd.concat([xi_df, remaining.head(11 - len(xi_df))])

    return xi_df.head(11)[["batter", "role", "form_score", "strike_rate"]]
