#!/usr/bin/env python3
"""
train.py - Unified NBA Betting Agent Training System
Combines standard and injury-aware learning in one flexible pipeline
Works with or without dynamic_player_handler.py

Features:
- Incremental learning (warm_start)
- Optional injury-aware features
- Advanced injury-adjusted targets
- Both player and team model training
- Handles inactive/rookie players
- Daily data updates
"""

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
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ===== TRY TO IMPORT INJURY FEATURES (Optional Enhancement) =====
try:
    from dynamic_player_handler import (
        refresh_player_database,
        InjuryAwareDataBuilder,
        DynamicPlayerLookup
    )
    INJURY_FEATURES_AVAILABLE = True
    logger.info("✓ Advanced injury learning system available")
except ImportError:
    INJURY_FEATURES_AVAILABLE = False
    logger.info("ℹ Running in standard mode (add dynamic_player_handler.py for injury learning)")


# ==================== ENHANCED FEATURE ENGINEERING ====================

def calculate_enhanced_features(df_players: pd.DataFrame, use_injury_features: bool = True) -> pd.DataFrame:
    """
    Calculate all features including optional injury-aware features
    
    Args:
        df_players: Raw player stats dataframe
        use_injury_features: Whether to calculate advanced injury features
    """
    logger.info("   Calculating features...")
    df = df_players.copy()
    
    # ===== BASIC PER-GAME STATS =====
    # Helper function to safely get column
    def safe_col(col_name, default):
        if col_name in df.columns:
            return df[col_name].fillna(default).astype(float)
        else:
            return default
    
    df['MIN_PG'] = safe_col('MIN', 0)
    df['PTS_PG'] = safe_col('PTS', 0)
    df['FGA_PG'] = safe_col('FGA', 0)
    df['FG3A_PG'] = safe_col('FG3A', 0)
    df['FTA_PG'] = safe_col('FTA', 0)
    df['AST_PG'] = safe_col('AST', 0)
    df['REB_PG'] = safe_col('REB', 0)
    df['STL_PG'] = safe_col('STL', 0)
    df['BLK_PG'] = safe_col('BLK', 0)
    df['TOV_PG'] = safe_col('TOV', 0)
    df['FG_PCT'] = safe_col('FG_PCT', 0.45)
    df['FG3_PCT'] = safe_col('FG3_PCT', 0.35)
    df['AGE'] = safe_col('AGE', 27)
    df['PACE'] = safe_col('PACE', 100.0)
    
    # ===== ADVANCED EFFICIENCY METRICS =====
    # True Shooting %
    df['TS_PCT'] = df['PTS_PG'] / (2 * (df['FGA_PG'] + 0.44 * df['FTA_PG']))
    df['TS_PCT'] = df['TS_PCT'].fillna(0.55).clip(0.40, 0.80)
    
    # Usage %
    if 'USG_PCT' not in df.columns or df['USG_PCT'].isna().all():
        poss_est = df['FGA_PG'] + 0.44 * df['FTA_PG'] + df['TOV_PG']
        df['USG_PCT'] = np.where(
            df['MIN_PG'] > 0,
            (poss_est / df['MIN_PG']) * 48 * 5,
            25.0
        )
    df['USG_PCT'] = df['USG_PCT'].fillna(25.0).clip(12.0, 42.0)
    
    # Player Efficiency Rating
    df['FGM'] = df['FGA_PG'] * df['FG_PCT']
    df['PER'] = (
        df['PTS_PG'] + df['REB_PG'] + df['AST_PG'] +
        3 * (df['STL_PG'] + df['BLK_PG']) - df['TOV_PG'] -
        (df['FGA_PG'] - df['FGM'])
    )
    df['PER'] = df['PER'].clip(5.0, 38.0)
    
    # ===== INJURY-AWARE FEATURES (Optional) =====
    if use_injury_features and INJURY_FEATURES_AVAILABLE:
        logger.info("   Adding advanced injury features...")
        
        # Games Played Ratio (availability throughout season)
        max_gp = df['GP'].max() if df['GP'].max() > 0 else 1
        df['GP_RATIO'] = df['GP'] / max_gp
        
        # Availability Score (combines GP and injury risk)
        if 'INJURY_RISK_SCORE' in df.columns:
            df['AVAILABILITY_SCORE'] = df['GP_RATIO'] * (1 - df['INJURY_RISK_SCORE'])
        else:
            df['AVAILABILITY_SCORE'] = df['GP_RATIO']
        
        # Recovery Factor (recent return from injury)
        if 'DAYS_SINCE_INJURY' in df.columns:
            df['RECOVERY_FACTOR'] = np.where(
                df['DAYS_SINCE_INJURY'] < 14,
                0.85,  # 85% within 2 weeks
                np.where(
                    df['DAYS_SINCE_INJURY'] < 30,
                    0.95,  # 95% within a month
                    1.0    # 100% after a month
                )
            )
        else:
            df['RECOVERY_FACTOR'] = 1.0
        
        # Age-Injury Interaction (older players = higher risk)
        if 'INJURY_COUNT_LAST_YEAR' in df.columns:
            df['AGE_INJURY_RISK'] = (df['AGE'] / 30) * df['INJURY_COUNT_LAST_YEAR']
        else:
            df['AGE_INJURY_RISK'] = 0
    
    logger.info(f"   ✓ Features calculated for {len(df)} players")
    return df


def calculate_injury_adjusted_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust training targets based on injury impact
    Helps model learn true ability vs current injured performance
    """
    if not INJURY_FEATURES_AVAILABLE or 'RECOVERY_FACTOR' not in df.columns:
        df['PTS_TARGET'] = df['PTS_PG'].copy()
        return df
    
    logger.info("   Adjusting targets for injury impact...")
    
    df = df.copy()
    df['PTS_ACTUAL'] = df['PTS_PG'].copy()
    
    # For recently injured players, estimate their true ability
    # (current stats may be suppressed due to injury recovery)
    df['PTS_ADJUSTED'] = df['PTS_ACTUAL'] / df['RECOVERY_FACTOR']
    
    # Cap adjustments at reasonable levels
    df['PTS_ADJUSTED'] = df['PTS_ADJUSTED'].clip(
        upper=df['PTS_ACTUAL'] * 1.3  # Max 30% adjustment
    )
    
    # Use adjusted target for players recovering from injury
    df['PTS_TARGET'] = np.where(
        df['RECOVERY_FACTOR'] < 1.0,
        df['PTS_ADJUSTED'],
        df['PTS_ACTUAL']
    )
    
    adjusted_count = (df['RECOVERY_FACTOR'] < 1.0).sum()
    logger.info(f"   ✓ Adjusted targets for {adjusted_count} recovering players")
    
    return df


# ==================== PLAYER MODEL TRAINING ====================

def incremental_train_player_model(retrain_from_scratch: bool = False, season: str = '2025-26'):
    """
    Train player model with optional injury awareness
    Supports incremental learning for efficient daily updates
    
    Args:
        retrain_from_scratch: If True, trains completely new model
                             If False, uses warm_start to update existing model
        season: NBA season (e.g., '2025-26')
    """
    
    logger.info("="*70)
    logger.info("NBA BETTING AGENT - PLAYER MODEL TRAINING")
    if INJURY_FEATURES_AVAILABLE:
        logger.info("Mode: Advanced Injury-Aware Learning ✨")
    else:
        logger.info("Mode: Standard Training")
    logger.info("="*70)
    
    # ===== STEP 1: FETCH DATA =====
    logger.info("\n[1/5] Fetching latest player data...")
    
    if INJURY_FEATURES_AVAILABLE:
        # Use advanced player database (includes inactive/rookies)
        df_players = refresh_player_database(season)
        logger.info(f"   Total players in database: {len(df_players)}")
    else:
        # Use standard stats fetch
        df_players = fetch_current_season_stats()
        logger.info(f"   Players loaded: {len(df_players)}")
    
    # ===== STEP 2: FILTER FOR TRAINING =====
    df_players = df_players.dropna(subset=['PLAYER_NAME', 'PTS', 'MIN'])
    df_players = df_players[df_players['GP'] >= 5].copy()
    df_players = df_players[df_players['MIN'] >= 12.0].copy()
    logger.info(f"   Training candidates: {len(df_players)} active players")
    
    # ===== STEP 3: FEATURE ENGINEERING =====
    logger.info("\n[2/5] Feature engineering...")
    df_features = calculate_enhanced_features(
        df_players, 
        use_injury_features=INJURY_FEATURES_AVAILABLE
    )
    
    # ===== STEP 4: INJURY-ADJUSTED TARGETS =====
    logger.info("\n[3/5] Preparing training targets...")
    df_with_targets = calculate_injury_adjusted_targets(df_features)
    
    # ===== STEP 5: PREPARE TRAINING DATA =====
    
    # Base features (always available)
    BASE_FEATURES = [
        'MIN_PG', 'USG_PCT', 'TS_PCT', 'FTA_PG', 'AST_PG',
        'FG3A_PG', 'PER', 'FG_PCT', 'FG3_PCT', 'AGE', 'PACE'
    ]
    
    # Injury features (optional)
    INJURY_FEATURES = [
        'GP_RATIO', 'AVAILABILITY_SCORE', 'RECOVERY_FACTOR', 'AGE_INJURY_RISK'
    ]
    
    # Combine available features
    FEATURES = BASE_FEATURES.copy()
    if INJURY_FEATURES_AVAILABLE:
        for feat in INJURY_FEATURES:
            if feat in df_with_targets.columns:
                FEATURES.append(feat)
    
    # Also check for features from InjuryAwareDataBuilder
    if INJURY_FEATURES_AVAILABLE and 'INJURY_RISK_SCORE' in df_with_targets.columns:
        extra_features = [
            'INJURY_RISK_SCORE', 'DAYS_SINCE_INJURY', 
            'INJURY_COUNT_LAST_YEAR', 'CHRONIC_INJURY_FLAG'
        ]
        for feat in extra_features:
            if feat in df_with_targets.columns and feat not in FEATURES:
                FEATURES.append(feat)
    
    logger.info(f"   Using {len(FEATURES)} features:")
    logger.info(f"     • Base: {len(BASE_FEATURES)}")
    if len(FEATURES) > len(BASE_FEATURES):
        logger.info(f"     • Injury-aware: {len(FEATURES) - len(BASE_FEATURES)}")
    
    # Prepare X and y
    X = df_with_targets[FEATURES].copy()
    y = df_with_targets['PTS_TARGET'].astype(float)
    
    # Remove any NaN/Inf
    mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
    X = X[mask]
    y = y[mask]
    
    logger.info(f"\n[4/5] Training player model...")
    logger.info(f"   Final samples: {len(X)} | Mean PPG: {y.mean():.1f}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ===== STEP 6: TRAIN MODEL =====
    
    # Incremental update if model exists
    if not retrain_from_scratch and Path("models/player_model_2025.pkl").exists():
        logger.info("   Loading existing model for incremental update...")
        try:
            player_model = joblib.load("models/player_model_2025.pkl")
            
            # Check feature compatibility
            if hasattr(player_model, 'n_features_in_') and player_model.n_features_in_ != len(FEATURES):
                logger.warning(f"   Feature mismatch: model has {player_model.n_features_in_}, data has {len(FEATURES)}")
                logger.info("   Training from scratch instead...")
                retrain_from_scratch = True
            else:
                # Incremental learning: add more trees
                player_model.n_estimators += 100
                player_model.fit(X_train, y_train, xgb_model=player_model.get_booster())
                logger.info("   ✓ Incremental update completed (+100 trees)")
        except Exception as e:
            logger.warning(f"   Failed to load existing model: {e}")
            logger.info("   Training from scratch instead...")
            retrain_from_scratch = True
    
    # Train from scratch if needed
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
    
    # ===== STEP 7: EVALUATE =====
    pred_test = player_model.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    
    logger.info(f"\n   MODEL PERFORMANCE:")
    logger.info(f"   • MAE: {mae:.2f} points")
    logger.info(f"   • R²:  {r2:.4f}")
    
    # Feature importance
    if hasattr(player_model, 'feature_importances_'):
        importance = dict(zip(FEATURES, player_model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("\n   TOP 5 FEATURES:")
        for feat, imp in top_features:
            logger.info(f"     {feat:25s} → {imp:.4f}")
    
    # ===== STEP 8: SAVE =====
    joblib.dump(player_model, "models/player_model_2025.pkl")
    df_with_targets.to_csv("data/2025_26_players.csv", index=False)
    logger.info("\n   ✓ Model and data saved")
    
    return player_model, mae, r2, FEATURES


# ==================== TEAM MODEL TRAINING ====================

def train_team_model():
    """
    Train team total prediction model
    Less frequent updates needed (team stats change slowly)
    """
    
    logger.info("\n[5/5] Training team model...")
    
    df_teams = fetch_team_stats()
    logger.info(f"   Teams loaded: {len(df_teams)}")
    
    # Find correct column names (different APIs use different names)
    ortg_col = next((c for c in df_teams.columns if 'OFF' in c.upper() and 'RATING' in c.upper()), 'OFF_RATING')
    drtg_col = next((c for c in df_teams.columns if 'DEF' in c.upper() and 'RATING' in c.upper()), 'DEF_RATING')
    pace_col = next((c for c in df_teams.columns if 'PACE' in c.upper()), 'PACE')
    
    # Simulate matchups
    X_team_sim = []
    y_total_sim = []
    
    for _ in range(3000):
        row1, row2 = df_teams.sample(2, replace=False).iloc[0], df_teams.sample(1).iloc[0]
        if row1.name == row2.name:
            continue
        
        # Home team gets +3.5 point advantage
        h_off = float(row1[ortg_col]) + 3.5
        h_def = float(row1[drtg_col])
        h_pace = float(row1[pace_col])
        a_off = float(row2[ortg_col])
        a_def = float(row2[drtg_col])
        a_pace = float(row2[pace_col])
        
        X_team_sim.append([h_off, h_def, h_pace, a_off, a_def, a_pace])
        
        # Calculate expected points
        avg_pace = (h_pace + a_pace) / 2
        home_pts = ((h_off + a_def) / 2) * avg_pace / 100
        away_pts = ((a_off + h_def) / 2) * avg_pace / 100
        
        # Add realistic variance
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
    logger.info("   ✓ Team model saved")
    
    return team_model


# ==================== DATA UPDATES ====================

def update_injuries_and_games():
    """Update daily data: injuries and today's games"""
    
    logger.info("\n[BONUS] Updating daily data...")
    
    # Fetch injuries
    try:
        injuries = fetch_live_injuries()
        logger.info(f"   ✓ {len(injuries)} injury reports")
        pd.DataFrame(injuries).to_csv("data/injuries.csv", index=False)
        
        # Update injury history if available
        if INJURY_FEATURES_AVAILABLE:
            try:
                injury_builder = InjuryAwareDataBuilder()
                injury_builder.update_injury_history(injuries)
                logger.info("   ✓ Injury history database updated")
            except Exception as e:
                logger.warning(f"   Could not update injury history: {e}")
    except Exception as e:
        logger.error(f"   ⚠ Injury fetch failed: {e}")
        injuries = []
    
    # Fetch today's games
    try:
        games = fetch_todays_games_with_odds()
        logger.info(f"   ✓ {len(games)} games scheduled")
        
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


# ==================== MAIN PIPELINE ====================

def main(retrain_from_scratch: bool = False, season: str = '2025-26'):
    """
    Main training pipeline
    
    Args:
        retrain_from_scratch: If True, trains completely new models
        season: NBA season (e.g., '2025-26')
    """
    
    start_time = datetime.now()
    
    # Train player model (with optional injury awareness)
    player_model, mae, r2, features = incremental_train_player_model(
        retrain_from_scratch=retrain_from_scratch,
        season=season
    )
    
    # Train team model
    team_model = train_team_model()
    
    # Update daily data
    injuries, games = update_injuries_and_games()
    
    # Save metadata
    metadata = {
        "version": "v4.0-UNIFIED",
        "training_date": datetime.now().isoformat(),
        "training_duration_seconds": round((datetime.now() - start_time).total_seconds(), 1),
        "retrained_from_scratch": retrain_from_scratch,
        "season": season,
        "injury_learning_enabled": INJURY_FEATURES_AVAILABLE,
        "player_model": {
            "features": features,
            "feature_count": len(features),
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
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Duration:        {metadata['training_duration_seconds']}s")
    logger.info(f"Model Version:   {metadata['version']}")
    logger.info(f"Performance:     MAE={mae:.2f}, R²={r2:.4f}")
    logger.info(f"Features Used:   {len(features)} ({len(features) - 11} injury features)" if len(features) > 11 else f"Features Used:   {len(features)}")
    logger.info(f"Injury Learning: {'✓ ENABLED' if INJURY_FEATURES_AVAILABLE else '✗ Not Available'}")
    logger.info(f"Data Updates:    {len(injuries)} injuries, {len(games)} games")
    logger.info("="*70)
    
    return metadata


# ==================== CLI ====================

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    full_retrain = "--full" in sys.argv or "-f" in sys.argv
    season = "2025-26"  # Default season
    
    # Check for season argument
    for arg in sys.argv:
        if arg.startswith("--season="):
            season = arg.split("=")[1]
    
    # Run training
    if full_retrain:
        logger.info("🔄 Running FULL retraining from scratch...\n")
    else:
        logger.info("⚡ Running INCREMENTAL update...\n")
    
    metadata = main(retrain_from_scratch=full_retrain, season=season)
    
    # Exit with success code
    exit(0)