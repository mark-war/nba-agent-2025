# train.py — INCREMENTAL LEARNING with Online Updates
from utils import (
    fetch_current_season_stats,
    fetch_team_stats,
    fetch_live_injuries,
    fetch_todays_games_with_odds
)
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

def incremental_train_player_model(retrain_from_scratch=False):
    """
    Incremental training - updates model with new data
    
    Args:
        retrain_from_scratch: If True, trains completely new model
                             If False, uses warm_start to update existing model
    """
    
    logger.info("="*70)
    logger.info("NBA BETTING AGENT - INCREMENTAL TRAINING")
    logger.info("="*70)
    
    # Fetch fresh data
    logger.info("\n[1/4] Fetching latest player stats...")
    df_players = fetch_current_season_stats()
    logger.info(f"   Raw players loaded: {len(df_players)}")
    
    # Clean data
    df_players = df_players.dropna(subset=['PLAYER_NAME', 'PTS', 'MIN'])
    df_players = df_players[df_players['GP'] >= 5].copy()
    df_players = df_players[df_players['MIN'] >= 12.0].copy()
    logger.info(f"   After filtering: {len(df_players)} active players")
    
    # Calculate per-game stats
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
    
    # Advanced metrics
    df_players['TS_PCT'] = df_players['PTS'] / (2 * (df_players['FGA'] + 0.44 * df_players['FTA']))
    df_players['TS_PCT'] = df_players['TS_PCT'].clip(0.40, 0.80).fillna(0.58)
    
    if 'USG_PCT' not in df_players.columns or df_players['USG_PCT'].isna().all():
        poss_est = df_players['FGA_PG'] + 0.44 * df_players['FTA_PG'] + df_players['TOV_PG']
        df_players['USG_PCT'] = (poss_est / df_players['MIN_PG']) * 48 * 5
    df_players['USG_PCT'] = df_players['USG_PCT'].clip(12.0, 42.0).fillna(25.0)
    
    df_players['PER'] = (
        df_players['PTS_PG'] +
        df_players['REB_PG'] +
        df_players['AST_PG'] +
        3 * (df_players['STL_PG'] + df_players['BLK_PG']) -
        df_players['TOV_PG'] -
        (df_players['FGA_PG'] - df_players['FGM'])
    )
    df_players['PER'] = df_players['PER'].clip(5.0, 38.0)
    
    # Fill defaults
    defaults = {'FG_PCT': 0.45, 'FG3_PCT': 0.35, 'AGE': 27, 'PACE': 100.0}
    for col, val in defaults.items():
        if col not in df_players.columns:
            df_players[col] = val
        else:
            df_players[col] = df_players[col].fillna(val)
    
    FEATURES = [
        'MIN_PG', 'USG_PCT', 'TS_PCT', 'FTA_PG', 'AST_PG',
        'FG3A_PG', 'PER', 'FG_PCT', 'FG3_PCT', 'AGE', 'PACE'
    ]
    
    X = df_players[FEATURES].copy()
    y = df_players['PTS'].astype(float)
    
    logger.info(f"\n[2/4] Training player model...")
    logger.info(f"   Samples: {len(X)} | Mean PPG: {y.mean():.1f}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Load existing model if incremental update
    if not retrain_from_scratch and Path("models/player_model_2025.pkl").exists():
        logger.info("   Loading existing model for incremental update...")
        try:
            player_model = joblib.load("models/player_model_2025.pkl")
            # Use warm_start for incremental learning
            player_model.n_estimators += 100  # Add 100 more trees
            player_model.fit(X_train, y_train, xgb_model=player_model.get_booster())
            logger.info("   ✓ Incremental update completed")
        except Exception as e:
            logger.warning(f"   Failed to load existing model: {e}")
            logger.info("   Training from scratch instead...")
            retrain_from_scratch = True
    
    if retrain_from_scratch or not Path("models/player_model_2025.pkl").exists():
        logger.info("   Training new model from scratch...")
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
    
    # Evaluate
    pred_test = player_model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    
    logger.info(f"   PLAYER MODEL → MAE: {mae:.2f} | R²: {r2:.4f}")
    
    # Save model
    joblib.dump(player_model, "models/player_model_2025.pkl")
    logger.info("   player_model_2025.pkl → SAVED")
    
    # Save training data for future incremental updates
    df_players.to_csv("data/2025_26_players.csv", index=False)
    logger.info("   player data → SAVED")
    
    return player_model, mae, r2, FEATURES

def train_team_model():
    """Train team model (less frequent updates needed)"""
    
    logger.info("\n[3/4] Training team model...")
    
    df_teams = fetch_team_stats()
    logger.info(f"   Teams loaded: {len(df_teams)}")
    
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
        
        h_off = float(row1[ortg_col]) + 3.5
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
    df_teams.to_csv("data/2025_26_teams.csv", index=False)
    logger.info("   team_model_2025.pkl → SAVED")
    
    return team_model

def update_injuries_and_games():
    """Update injuries and today's games"""
    
    logger.info("\n[4/4] Updating injuries and games...")
    
    try:
        injuries = fetch_live_injuries()
        logger.info(f"   ✓ Loaded {len(injuries)} injury reports")
        pd.DataFrame(injuries).to_csv("data/injuries.csv", index=False)
    except Exception as e:
        logger.error(f"   ⚠ Injury fetch failed: {e}")
        injuries = []
    
    try:
        games = fetch_todays_games_with_odds()
        logger.info(f"   ✓ Loaded {len(games)} games for today")
        
        with open("data/todays_games.json", "w") as f:
            json.dump({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "timestamp": datetime.now().isoformat(),
                "games": games
            }, f, indent=2)
    except Exception as e:
        logger.error(f"   ⚠ Game/odds fetch failed: {e}")
        games = []
    
    return injuries, games

def main(retrain_from_scratch=False):
    """Main training pipeline"""
    
    start_time = datetime.now()
    
    # Train player model (incremental or from scratch)
    player_model, mae, r2, features = incremental_train_player_model(retrain_from_scratch)
    
    # Train team model (less frequent)
    team_model = train_team_model()
    
    # Update injuries and games
    injuries, games = update_injuries_and_games()
    
    # Save metadata
    metadata = {
        "version": "v4.0-INCREMENTAL",
        "training_date": datetime.now().isoformat(),
        "training_duration_seconds": (datetime.now() - start_time).total_seconds(),
        "retrained_from_scratch": retrain_from_scratch,
        "player_model": {
            "features": features,
            "samples": len(pd.read_csv("data/2025_26_players.csv")),
            "mae": round(float(mae), 3),
            "r2": round(float(r2), 4)
        },
        "team_model": {
            "samples": 3000
        },
        "data_updates": {
            "injuries": len(injuries),
            "games": len(games)
        }
    }
    
    with open("models/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Duration: {metadata['training_duration_seconds']:.1f}s")
    logger.info(f"Model Performance: MAE={mae:.2f}, R²={r2:.4f}")
    logger.info("="*70)
    
    return metadata

if __name__ == "__main__":
    import sys
    
    # Check if --full flag is passed
    full_retrain = "--full" in sys.argv or "-f" in sys.argv
    
    if full_retrain:
        logger.info("Running FULL retraining from scratch...")
    else:
        logger.info("Running INCREMENTAL update...")
    
    main(retrain_from_scratch=full_retrain)