# train.py — FINAL 100% WORKING VERSION (Dec 2025)
from utils import (
    fetch_current_season_stats,
    fetch_team_stats,
)
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path
import json
from datetime import datetime

print("=" * 70)
print("NBA BETTING AGENT 2025-26 — FINAL CORRECT TRAINING SCRIPT")
print("=" * 70)

Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ===================================================================
# 1. PLAYER MODEL — NOW FIXED WITH REAL TARGETS (NO RANDOM VARIANCE!)
# ===================================================================
print("\n[1/2] Training PLAYER model with REAL points per game...")

df_players = fetch_current_season_stats()
print(f"   Raw players loaded: {len(df_players)}")

# Clean data
df_players = df_players.dropna(subset=['PLAYER_NAME', 'PTS', 'MIN'])
df_players = df_players[df_players['GP'] >= 5].copy()
df_players = df_players[df_players['MIN'] >= 12.0].copy()  # reasonable minutes
print(f"   After filtering: {len(df_players)} active players")

# YOUR CSV IS ALREADY PER-GAME → DO NOT DIVIDE AGAIN!
df_players['MIN_PG']   = df_players['MIN'].astype(float)
df_players['PTS_PG']   = df_players['PTS'].astype(float)
df_players['FGA_PG']   = df_players['FGA'].astype(float)
df_players['FG3A_PG']  = df_players['FG3A'].astype(float)
df_players['FTA_PG']   = df_players['FTA'].astype(float)
df_players['AST_PG']   = df_players['AST'].astype(float)
df_players['REB_PG']   = df_players['REB'].astype(float)
df_players['STL_PG']   = df_players['STL'].astype(float)
df_players['BLK_PG']   = df_players['BLK'].astype(float)
df_players['TOV_PG']   = df_players['TOV'].astype(float)

# True Shooting Percentage
df_players['TS_PCT'] = df_players['PTS'] / (2 * (df_players['FGA'] + 0.44 * df_players['FTA']))
df_players['TS_PCT'] = df_players['TS_PCT'].clip(0.40, 0.80).fillna(0.58)

# USG_PCT
if 'USG_PCT' not in df_players.columns or df_players['USG_PCT'].isna().all():
    poss_est = df_players['FGA_PG'] + 0.44 * df_players['FTA_PG'] + df_players['TOV_PG']
    df_players['USG_PCT'] = (poss_est / df_players['MIN_PG']) * 48 * 5
df_players['USG_PCT'] = df_players['USG_PCT'].clip(12.0, 42.0).fillna(25.0)

# Simple realistic PER
df_players['PER'] = (
    df_players['PTS_PG'] +
    df_players['REB_PG'] +
    df_players['AST_PG'] +
    3 * (df_players['STL_PG'] + df_players['BLK_PG']) -
    df_players['TOV_PG'] -
    (df_players['FGA_PG'] - df_players['FGM'])
)
df_players['PER'] = df_players['PER'].clip(5.0, 38.0)

# Fill missing basic columns
defaults = {'FG_PCT': 0.45, 'FG3_PCT': 0.35, 'AGE': 27, 'PACE': 100.0}
for col, val in defaults.items():
    if col not in df_players.columns:
        df_players[col] = val
    else:
        df_players[col] = df_players[col].fillna(val)

# FINAL FEATURE LIST — MUST MATCH main.py EXACTLY
FEATURES = [
    'MIN_PG', 'USG_PCT', 'TS_PCT', 'FTA_PG', 'AST_PG',
    'FG3A_PG', 'PER', 'FG_PCT', 'FG3_PCT', 'AGE', 'PACE'
]

X = df_players[FEATURES].copy()
y = df_players['PTS'].astype(float)  # ← REAL PPG, no fake multipliers!

print(f"   Training samples: {len(X)} | Mean PPG: {y.mean():.1f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

player_model = XGBRegressor(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)

player_model.fit(X_train, y_train)

pred_test = player_model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
r2 = r2_score(y_test, pred_test)

print(f"   PLAYER MODEL → MAE: {mae:.2f} | R²: {r2:.4f}")

joblib.dump(player_model, "models/player_model_2025.pkl")
print("   player_model_2025.pkl → SAVED")

# ===================================================================
# 2. TEAM MODEL — YOUR ORIGINAL LOGIC (IT WAS ALREADY GOOD)
# ===================================================================
print("\n[2/2] Training TEAM model (unchanged & working)...")

df_teams = fetch_team_stats()
print(f"   Teams loaded: {len(df_teams)}")

# Find correct column names
ortg_col = next((c for c in df_teams.columns if 'OFF' in c.upper() and 'RATING' in c.upper()), 'OFF_RATING')
drtg_col = next((c for c in df_teams.columns if 'DEF' in c.upper() and 'RATING' in c.upper()), 'DEF_RATING')
pace_col = next((c for c in df_teams.columns if 'PACE' in c.upper()), 'PACE')

X_team_sim = []
y_total_sim = []

for _ in range(3000):
    row1, row2 = df_teams.sample(2, replace=False).iloc[0], df_teams.sample(1).iloc[0]
    if row1.name == row2.name:
        continue

    h_off = float(row1[ortg_col]) + 3.5   # home advantage
    h_def = float(row1[drtg_col])
    h_pace = float(row1[pace_col])
    a_off = float(row2[ortg_col])
    a_def = float(row2[drtg_col])
    a_pace = float(row2[pace_col])

    X_team_sim.append([h_off, h_def, h_pace, a_off, a_def, a_pace])

    avg_pace = (h_pace + a_pace) / 2
    home_pts = ((h_off + a_def) / 2) * avg_pace / 100
    away_pts = ((a_off + h_def) / 2) * avg_pace / 100

    home_pts *= np.random.normal(1.0, 0.07)
    away_pts *= np.random.normal(1.0, 0.07)

    y_total_sim.append(home_pts + away_pts)

X_team = pd.DataFrame(X_team_sim, columns=[
    'home_OFF', 'home_DEF', 'home_PACE', 'away_OFF', 'away_DEF', 'away_PACE'
])

team_model = XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    random_state=42,
    n_jobs=-1
)
team_model.fit(X_team, y_total_sim)

joblib.dump(team_model, "models/team_model_2025.pkl")
print("   team_model_2025.pkl → SAVED")

# ===================================================================
# SAVE METADATA
# ===================================================================
metadata = {
    "version": "v3.0-FINAL-FIXED",
    "training_date": datetime.now().isoformat(),
    "player_model": {
        "features": FEATURES,
        "samples": len(X),
        "mae": round(float(mae), 3),
        "r2": round(float(r2), 4),
        "target_mean_ppg": round(float(y.mean()), 2)
    },
    "team_model": {
        "features": X_team.columns.tolist(),
        "samples": len(X_team)
    },
    "note": "Player model now uses REAL PPG targets — no synthetic variance!"
}

with open("models/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)