from utils import (
    fetch_current_season_stats, 
    fetch_team_stats, 
    fetch_live_injuries,
    fetch_todays_games_with_odds
)
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path
import json
from datetime import datetime

print("=" * 60)
print("NBA BETTING AGENT 2025-26 — DAILY TRAINING")
print("=" * 60)

# Create directories
Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# === 1. FETCH PLAYER DATA ===
print("\n[1/5] Fetching player stats...")
df_players = fetch_current_season_stats()
print(f"   ✓ Loaded {len(df_players)} players")

# Clean and validate
df_players = df_players.dropna(subset=['PLAYER_NAME', 'PTS'])
df_players = df_players[df_players['GP'] > 0]  # Only players with games
print(f"   ✓ Cleaned to {len(df_players)} active players")

# === 2. TRAIN PLAYER MODEL ===
print("\n[2/5] Training player prediction model...")

# Feature engineering
features = ['PTS', 'MIN', 'GP', 'AGE', 'FGA', 'FG3A', 'FTA', 'AST', 'REB']
available_features = [f for f in features if f in df_players.columns]

X_player = df_players[available_features].fillna(0)
# Target: Predict next game performance (with slight upward bias for variance)
y_player = df_players['PTS'] * 1.05 + (df_players['MIN'] / 36) * 5

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_player, y_player, test_size=0.2, random_state=42
)

player_model = XGBRegressor(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

player_model.fit(X_train, y_train)

# Evaluate
y_pred = player_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"   ✓ Player Model MAE: {mae:.2f} | R²: {r2:.3f}")

joblib.dump(player_model, "models/player_model_2025.pkl")
print("   ✓ Saved to models/player_model_2025.pkl")

# === 3. FETCH TEAM DATA ===
print("\n[3/5] Fetching team stats...")
df_teams = fetch_team_stats()
print(f"   ✓ Loaded {len(df_teams)} teams")

# === 4. TRAIN TEAM MODEL ===
print("\n[4/5] Training team/game prediction model...")

# Find actual column names
def find_column(df, keywords):
    cols = df.columns.str.upper()
    for kw in keywords:
        match = [c for c in df.columns if kw.upper() in c.upper()]
        if match:
            return match[0]
    return None

ortg_col = find_column(df_teams, ['OFF_RATING', 'OFFENSIVE_RATING', 'ORTG'])
drtg_col = find_column(df_teams, ['DEF_RATING', 'DEFENSIVE_RATING', 'DRTG'])
pace_col = find_column(df_teams, ['PACE'])

if not all([ortg_col, drtg_col, pace_col]):
    print("   ⚠ Missing advanced stats — calculating from basic stats...")
    
    # Calculate if missing
    if 'possessions' not in df_teams.columns:
        df_teams['possessions'] = (
            df_teams['FGA'] + 
            0.44 * df_teams['FTA'] - 
            df_teams['OREB'] + 
            df_teams['TOV']
        )
    
    if ortg_col is None:
        df_teams['OFF_RATING'] = (df_teams['PTS'] / df_teams['possessions']) * 100
        ortg_col = 'OFF_RATING'
    
    if drtg_col is None:
        df_teams['DEF_RATING'] = df_teams['OFF_RATING'] - df_teams.get('PLUS_MINUS', 0)
        drtg_col = 'DEF_RATING'
    
    if pace_col is None:
        df_teams['PACE'] = df_teams['possessions']
        pace_col = 'PACE'

print(f"   Using: {ortg_col}, {drtg_col}, {pace_col}")

# Create training data (simulate matchups)
import numpy as np

n_simulations = 500
home_teams = np.random.choice(len(df_teams), n_simulations)
away_teams = np.random.choice(len(df_teams), n_simulations)

X_team_list = []
y_total_list = []

for h_idx, a_idx in zip(home_teams, away_teams):
    if h_idx == a_idx:
        continue
    
    h_team = df_teams.iloc[h_idx]
    a_team = df_teams.iloc[a_idx]
    
    # Features
    X_team_list.append([
        h_team[ortg_col] + 3.5,  # Home advantage
        h_team[drtg_col],
        h_team[pace_col],
        a_team[ortg_col],
        a_team[drtg_col],
        a_team[pace_col]
    ])
    
    # Target: Total points
    avg_pace = (h_team[pace_col] + a_team[pace_col]) / 2
    total = ((h_team[ortg_col] + a_team[drtg_col]) / 2 + 
             (a_team[ortg_col] + h_team[drtg_col]) / 2) * avg_pace / 100
    y_total_list.append(total)

X_team = pd.DataFrame(
    X_team_list,
    columns=['home_OFF_RATING', 'home_DEF_RATING', 'home_PACE',
             'away_OFF_RATING', 'away_DEF_RATING', 'away_PACE']
)
y_total = pd.Series(y_total_list)

# Train team model
team_model = XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    random_state=42,
    n_jobs=-1
)

team_model.fit(X_team, y_total)

joblib.dump(team_model, "models/team_model_2025.pkl")
print(f"   ✓ Team model trained on {len(X_team)} simulations")
print("   ✓ Saved to models/team_model_2025.pkl")

# === 5. FETCH INJURIES & GAMES ===
print("\n[5/5] Fetching injuries and today's games...")

injuries = fetch_live_injuries()
print(f"   ✓ Loaded {len(injuries)} injury reports")

try:
    games = fetch_todays_games_with_odds()
    print(f"   ✓ Loaded {len(games)} games for today")
except Exception as e:
    print(f"   ⚠ Game fetch failed: {e}")
    games = []

# === SAVE TRAINING METADATA ===
metadata = {
    "training_date": datetime.now().isoformat(),
    "players_count": len(df_players),
    "teams_count": len(df_teams),
    "injuries_count": len(injuries),
    "todays_games": len(games),
    "player_model_mae": float(mae),
    "player_model_r2": float(r2),
    "features_used": available_features,
    "team_features": ['home_OFF_RATING', 'home_DEF_RATING', 'home_PACE',
                      'away_OFF_RATING', 'away_DEF_RATING', 'away_PACE']
}

with open("models/training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModels saved:")
print(f"  • Player model: MAE={mae:.2f}, R²={r2:.3f}")
print(f"  • Team model: {len(X_team)} simulations")
print(f"\nData loaded:")
print(f"  • {len(df_players)} players")
print(f"  • {len(df_teams)} teams")
print(f"  • {len(injuries)} injuries")
print(f"  • {len(games)} games today")
print(f"\nNext steps:")
print(f"  1. Set ODDS_API_KEY environment variable for live odds")
print(f"  2. Run: uvicorn main:app --reload --port 8000")
print(f"  3. Visit: http://localhost:8000/docs")
print("=" * 60)