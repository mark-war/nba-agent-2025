#!/usr/bin/env python3
"""
Injury-Aware Training System
- Learns from injury patterns
- Adjusts predictions based on injury history
- Updates daily with new injury data
- Handles inactive/rookie players
"""

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

# Import our dynamic player system
from dynamic_player_handler import (
    refresh_player_database,
    InjuryAwareDataBuilder,
    DynamicPlayerLookup
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Path("models").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

# ==================== ENHANCED FEATURE ENGINEERING ====================

def calculate_enhanced_features(df_players: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature calculation with injury awareness
    """
    logger.info("Calculating enhanced features...")
    
    # Basic stats (per-game already)
    df = df_players.copy()
    
    # Fill missing values
    df['MIN_PG'] = df['MIN'].fillna(0).astype(float)
    df['PTS_PG'] = df['PTS'].fillna(0).astype(float)
    df['FGA_PG'] = df['FGA'].fillna(0).astype(float)
    df['FG3A_PG'] = df['FG3A'].fillna(0).astype(float)
    df['FTA_PG'] = df['FTA'].fillna(0).astype(float)
    df['AST_PG'] = df['AST'].fillna(0).astype(float)
    df['REB_PG'] = df['REB'].fillna(0).astype(float)
    df['STL_PG'] = df['STL'].fillna(0).astype(float)
    df['BLK_PG'] = df['BLK'].fillna(0).astype(float)
    df['TOV_PG'] = df['TOV'].fillna(0).astype(float)
    df['FG_PCT'] = df['FG_PCT'].fillna(0.45).astype(float)
    df['FG3_PCT'] = df['FG3_PCT'].fillna(0.35).astype(float)
    df['AGE'] = df['AGE'].fillna(27).astype(float)
    df['PACE'] = df['PACE'].fillna(100.0).astype(float)
    
    # True Shooting %
    df['TS_PCT'] = df['PTS_PG'] / (2 * (df['FGA_PG'] + 0.44 * df['FTA_PG']))
    df['TS_PCT'] = df['TS_PCT'].fillna(0.55).clip(0.40, 0.80)
    
    # Usage %
    if 'USG_PCT' not in df.columns or df['USG_PCT'].isna().all():
        poss_est = df['FGA_PG'] + 0.44 * df['FTA_PG'] + df['TOV_PG']
        df['USG_PCT'] = (poss_est / df['MIN_PG']) * 48 * 5
    df['USG_PCT'] = df['USG_PCT'].fillna(25.0).clip(12.0, 42.0)
    
    # Player Efficiency Rating
    df['FGM'] = df['FGA_PG'] * df['FG_PCT']
    df['PER'] = (
        df['PTS_PG'] + df['REB_PG'] + df['AST_PG'] +
        3 * (df['STL_PG'] + df['BLK_PG']) - df['TOV_PG'] -
        (df['FGA_PG'] - df['FGM'])
    )
    df['PER'] = df['PER'].clip(5.0, 38.0)
    
    # ===== INJURY-AWARE FEATURES =====
    
    # Games Played Ratio (how much of season they've played)
    df['GP_RATIO'] = df['GP'] / df['GP'].max() if df['GP'].max() > 0 else 0
    
    # Availability Score (combines GP and injury risk)
    if 'INJURY_RISK_SCORE' in df.columns:
        df['AVAILABILITY_SCORE'] = df['GP_RATIO'] * (1 - df['INJURY_RISK_SCORE'])
    else:
        df['AVAILABILITY_SCORE'] = df['GP_RATIO']
    
    # Recent Form (if they've been injured, their stats may not reflect true ability)
    if 'DAYS_SINCE_INJURY' in df.columns:
        # Players recently returned from injury may underperform
        df['RECOVERY_FACTOR'] = np.where(
            df['DAYS_SINCE_INJURY'] < 14,
            0.85,  # 85% of normal within 2 weeks
            np.where(
                df['DAYS_SINCE_INJURY'] < 30,
                0.95,  # 95% within a month
                1.0    # 100% after a month
            )
        )
    else:
        df['RECOVERY_FACTOR'] = 1.0
    
    # Age-Injury Interaction (older players = slower recovery)
    if 'INJURY_COUNT_LAST_YEAR' in df.columns:
        df['AGE_INJURY_RISK'] = (df['AGE'] / 30) * df['INJURY_COUNT_LAST_YEAR']
    else:
        df['AGE_INJURY_RISK'] = 0
    
    logger.info(f"Features calculated for {len(df)} players")
    
    return df

# ==================== INJURY-ADJUSTED TARGETS ====================

def calculate_injury_adjusted_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust target variables based on injury history
    This helps the model learn injury impact patterns
    """
    logger.info("Calculating injury-adjusted targets...")
    
    df = df.copy()
    
    # Original targets
    df['PTS_ACTUAL'] = df['PTS_PG'].copy()
    df['PTS_TARGET'] = df['PTS_PG'].copy()
    
    # If player has injury history, their "true" ability may be higher
    # than current stats show
    if 'INJURY_RISK_SCORE' in df.columns and 'RECOVERY_FACTOR' in df.columns:
        # Adjust for recent recovery
        df['PTS_ADJUSTED'] = df['PTS_ACTUAL'] / df['RECOVERY_FACTOR']
        
        # But cap at reasonable values
        df['PTS_ADJUSTED'] = df['PTS_ADJUSTED'].clip(
            upper=df['PTS_ACTUAL'] * 1.3  # Max 30% adjustment
        )
        
        # Use adjusted as training target for injured players
        df['PTS_TARGET'] = np.where(
            df['RECOVERY_FACTOR'] < 1.0,
            df['PTS_ADJUSTED'],
            df['PTS_ACTUAL']
        )
    
    logger.info("Target adjustments complete")
    
    return df

# ==================== MAIN TRAINING FUNCTION ====================

def train_injury_aware_model(
    retrain_from_scratch: bool = False,
    season: str = '2025-26'
):
    """
    Train player model with injury awareness
    """
    logger.info("="*70)
    logger.info("INJURY-AWARE MODEL TRAINING")
    logger.info("="*70)
    
    # Step 1: Refresh player database (includes inactive/rookies)
    logger.info("\n[1/6] Refreshing player database...")
    df_all_players = refresh_player_database(season)
    
    # Step 2: Filter for training (need at least some games played)
    df_train_candidates = df_all_players[
        (df_all_players['GP'] >= 5) &
        (df_all_players['MIN_PG'] >= 12.0)
    ].copy()
    
    logger.info(f"   Training candidates: {len(df_train_candidates)} players")
    
    # Step 3: Calculate features
    logger.info("\n[2/6] Calculating features...")
    df_features = calculate_enhanced_features(df_train_candidates)
    
    # Step 4: Adjust targets for injury impact
    logger.info("\n[3/6] Adjusting targets for injury impact...")
    df_with_targets = calculate_injury_adjusted_targets(df_features)
    
    # Step 5: Prepare training data
    logger.info("\n[4/6] Preparing training data...")
    
    # Enhanced feature set (includes injury features)
    ENHANCED_FEATURES = [
        'MIN_PG', 'USG_PCT', 'TS_PCT', 'FTA_PG', 'AST_PG',
        'FG3A_PG', 'PER', 'FG_PCT', 'FG3_PCT', 'AGE', 'PACE',
        # Injury-aware features
        'GP_RATIO', 'AVAILABILITY_SCORE', 'RECOVERY_FACTOR',
        'AGE_INJURY_RISK'
    ]
    
    # Check which features actually exist
    available_features = [f for f in ENHANCED_FEATURES if f in df_with_targets.columns]
    logger.info(f"   Using {len(available_features)} features")
    
    X = df_with_targets[available_features].copy()
    y = df_with_targets['PTS_TARGET'].astype(float)
    
    # Remove any NaN/Inf
    mask = ~(X.isna().any(axis=1) | np.isinf(X).any(axis=1))
    X = X[mask]
    y = y[mask]
    
    logger.info(f"   Final training samples: {len(X)}")
    logger.info(f"   Target mean PPG: {y.mean():.2f}")
    
    # Step 6: Train model
    logger.info("\n[5/6] Training model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if not retrain_from_scratch and Path("models/player_model_2025.pkl").exists():
        logger.info("   Incremental update...")
        try:
            player_model = joblib.load("models/player_model_2025.pkl")
            player_model.n_estimators += 100
            player_model.fit(
                X_train, y_train,
                xgb_model=player_model.get_booster()
            )
        except:
            logger.warning("   Incremental failed, training from scratch...")
            retrain_from_scratch = True
    
    if retrain_from_scratch or not Path("models/player_model_2025.pkl").exists():
        logger.info("   Training from scratch...")
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
    
    logger.info(f"   MAE: {mae:.2f} | R²: {r2:.4f}")
    
    # Feature importance
    feature_importance = dict(zip(available_features, player_model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info("\n   Top 5 Features:")
    for feat, imp in top_features:
        logger.info(f"     {feat}: {imp:.4f}")
    
    # Step 7: Save model and metadata
    logger.info("\n[6/6] Saving model...")
    
    joblib.dump(player_model, "models/player_model_2025.pkl")
    
    metadata = {
        "version": "v4.0-INJURY-AWARE",
        "training_date": datetime.now().isoformat(),
        "retrained_from_scratch": retrain_from_scratch,
        "player_model": {
            "features": available_features,
            "samples": len(X),
            "mae": round(float(mae), 3),
            "r2": round(float(r2), 4),
            "target_mean": round(float(y.mean()), 2),
            "feature_importance": {k: round(v, 4) for k, v in top_features}
        },
        "injury_awareness": {
            "enabled": True,
            "features_used": [f for f in available_features if 'INJURY' in f or 'RECOVERY' in f or 'AVAILABILITY' in f]
        },
        "data_sources": {
            "total_players": len(df_all_players),
            "active": len(df_all_players[df_all_players['PLAYER_STATUS'] == 'ACTIVE']),
            "rookies": len(df_all_players[df_all_players['PLAYER_STATUS'] == 'ROOKIE']),
            "inactive": len(df_all_players[df_all_players['PLAYER_STATUS'] == 'INACTIVE'])
        }
    }
    
    with open("models/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("✓ Model saved with injury awareness!")
    logger.info("="*70)
    
    return player_model, metadata

# ==================== PREDICTION WITH INJURY ADJUSTMENT ====================

def predict_with_injury_adjustment(
    player_model,
    features: pd.DataFrame,
    injury_status: str,
    days_since_injury: int = 999,
    recovery_factor: float = 1.0
) -> float:
    """
    Make prediction with injury adjustment
    """
    # Base prediction
    base_pred = float(player_model.predict(features)[0])
    
    # Adjust based on injury status
    if injury_status in ['OUT', 'Doubtful']:
        return 0.0
    elif injury_status == 'Questionable':
        adjusted = base_pred * 0.85 * recovery_factor
    elif injury_status == 'Probable':
        adjusted = base_pred * 0.95 * recovery_factor
    else:
        adjusted = base_pred * recovery_factor
    
    return adjusted

# ==================== CLI ====================

if __name__ == "__main__":
    import sys
    
    full_retrain = "--full" in sys.argv or "-f" in sys.argv
    
    if full_retrain:
        logger.info("Running FULL retraining...")
    else:
        logger.info("Running INCREMENTAL update...")
    
    model, metadata = train_injury_aware_model(
        retrain_from_scratch=full_retrain
    )
    
    logger.info("\n✓ Training complete!")
    logger.info(f"  Model version: {metadata['version']}")
    logger.info(f"  Injury awareness: {metadata['injury_awareness']['enabled']}")
    logger.info(f"  Performance: MAE={metadata['player_model']['mae']}, R²={metadata['player_model']['r2']}")