# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="NFL Betting Predictions", layout="wide")

# -------------------------
# Load config from secrets (Streamlit Cloud)
# -------------------------
cfg = st.secrets if hasattr(st, "secrets") else {}
app_cfg = cfg.get("app", {})
DEFAULT_SEASON = app_cfg.get("default_season", 2023)
UNIT_SIZE = app_cfg.get("unit_size", 100)
STARTING_BANKROLL = app_cfg.get("starting_bankroll", 10000)
SPREAD_THRESHOLD = cfg.get("betting_thresholds", {}).get("spread_edge", 2.0)
TOTAL_THRESHOLD = cfg.get("betting_thresholds", {}).get("total_edge", 3.0)

st.title(app_cfg.get("title", "🏈 NFL Betting Predictions"))

# -------------------------
# Helper: load CSVs
# -------------------------
DATA_DIR = "data"
GAMES_CSV = os.path.join(DATA_DIR, "games.csv")
ODDS_CSV = os.path.join(DATA_DIR, "odds.csv")

def load_games_csv(path):
    if not os.path.exists(path):
        st.error(f"Missing games CSV at: {path}\nUpload your games.csv into the data/ folder in your repo.")
        st.stop()
    df = pd.read_csv(path)
    return df

def load_odds_csv(path):
    if not os.path.exists(path):
        st.error(f"Missing odds CSV at: {path}\nUpload your odds.csv into the data/ folder in your repo.")
        st.stop()
    df = pd.read_csv(path)
    return df

# -------------------------
# Load & merge data
# -------------------------
games_df = load_games_csv(GAMES_CSV)
odds_df = load_odds_csv(ODDS_CSV)

# Try to merge on game keys (game_id preferred). Fall back to season/week/home/away.
merge_on = None
if "game_id" in games_df.columns and "game_id" in odds_df.columns:
    merge_on = ["game_id"]
else:
    merge_on = ["season", "week", "home_team", "away_team"]

merged = pd.merge(games_df, odds_df, how="left", on=merge_on, suffixes=("", "_odds"))

# Drop rows missing essential odds
merged = merged.dropna(subset=["spread_closing", "total_closing"], how="any")
if len(merged) == 0:
    st.error("After merging games and odds, no rows remain. Check your CSV keys (season/week/home_team/away_team or game_id).")
    st.stop()

# -------------------------
# Derived fields
# -------------------------
merged["spread_actual"] = merged["home_score"] - merged["away_score"]
merged["total_points"] = merged["home_score"] + merged["away_score"]

# If features missing, create placeholders
if "elo_diff" not in merged.columns:
    merged["elo_diff"] = np.random.normal(0, 5, size=len(merged))
if "epa_diff" not in merged.columns:
    merged["epa_diff"] = np.random.normal(0, 0.05, size=len(merged))

# -------------------------
# Train simple models
# -------------------------
X_all = merged[["elo_diff", "epa_diff"]].values
y_spread = merged["spread_actual"].values
y_total = merged["total_points"].values

spread_model = LinearRegression().fit(X_all, y_spread)
total_model = LinearRegression().fit(X_all, y_total)

# -------------------------
# UI: season & week selection
# -------------------------
seasons = sorted(merged["season"].unique())
season = st.sidebar.selectbox("Season", seasons, index=max(0, seasons.index(DEFAULT_SEASON)) if DEFAULT_SEASON in seasons else 0)
weeks = sorted(merged[merged["season"]==season]["week"].unique())
week = st.sidebar.selectbox("Week", weeks)

subset = merged[(merged["season"]==season) & (merged["week"]==week)].copy().reset_index(drop=True)

# -------------------------
# Predictions + edges + bankroll sim
# -------------------------
bankroll = STARTING_BANKROLL
unit = UNIT_SIZE
rows = []
for _, r in subset.iterrows():
    X = np.array([[r["elo_diff"], r["epa_diff"]]])
    spread_pred = spread_model.predict(X)[0]
    total_pred = total_model.predict(X)[0]

    spread_edge = spread_pred - r["spread_closing"]
    total_edge = total_pred - r["total_closing"]

    pnl = 0
    bets = []
    # Spread bet (simple flat unit; adjust with Kelly later)
    if abs(spread_edge) > SPREAD_THRESHOLD:
        # decide pick relative to home perspective
        pick_home = spread_edge > 0  # model believes home beats closing by > threshold
        # check whether bet won: if model picks home spread, home must beat closing line
        if pick_home:
            won = (r["spread_actual"] > -r["spread_closing"])
        else:
            won = (r["spread_actual"] < -r["spread_closing"])
        pnl += unit if won else -unit
        bets.append(f"{'Home' if pick_home else 'Away'} spread ({spread_edge:.2f}) {'W' if won else 'L'}")

    # Total bet
    if abs(total_edge) > TOTAL_THRESHOLD:
        bet_over = total_edge > 0
        if bet_over:
            won = (r["total_points"] > r["total_closing"])
        else:
            won = (r["total_points"] < r["total_closing"])
        pnl += unit if won else -unit
        bets.append(f"{'Over' if bet_over else 'Under'} ({total_edge:.2f}) {'W' if won else 'L'}")

    bankroll += pnl
    rows.append({
        "Game": f"{r['away_team']} @ {r['home_team']}",
        "Spread Line": r["spread_closing"],
        "Spread Pred": round(spread_pred,2),
        "Spread Edge": round(spread_edge,2),
        "Total Line": r["total_closing"],
        "Total Pred": round(total_pred,2),
        "Total Edge": round(total_edge,2),
        "Bets": "; ".join(bets),
        "P&L": pnl,
        "Bankroll": bankroll
    })

pred_df = pd.DataFrame(rows)

# -------------------------
# Display results
# -------------------------
st.subheader(f"Predictions for Season {season} — Week {week}")
st.dataframe(pred_df, use_container_width=True)

st.subheader("Bankroll progression (simulated)")
st.line_chart(pred_df["Bankroll"].fillna(method="ffill"))
