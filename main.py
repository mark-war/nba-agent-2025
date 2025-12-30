# main.py — PRODUCTION-READY with Caching, Smart Predictions & Scalability
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from difflib import SequenceMatcher
import asyncio
from functools import lru_cache
import logging

from name_mapper import get_canonical_name, normalize_name_lower

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

# Cache durations
INJURY_REFRESH_INTERVAL = timedelta(hours=1)
PREDICTION_CACHE_DURATION = timedelta(minutes=30)
GAME_CACHE_DURATION = timedelta(hours=6)

# ==================== LOAD MODELS & DATA ====================
logger.info("Loading ML models and data...")
player_model = joblib.load(MODELS_DIR / "player_model_2025.pkl")
team_model = joblib.load(MODELS_DIR / "team_model_2025.pkl")

# Load with error handling
try:
    df_players = pd.read_csv(DATA_DIR / "2025_26_players.csv")
    df_teams = pd.read_csv(DATA_DIR / "2025_26_teams.csv")
    
    # Clean player names - remove any NaN or non-string values
    df_players = df_players[df_players['PLAYER_NAME'].notna()].copy()
    df_players = df_players[df_players['PLAYER_NAME'].astype(str).str.strip() != ''].copy()
    
    logger.info(f"Data loaded | Players: {len(df_players)} | Teams: {len(df_teams)}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

# Load metadata
try:
    with open(MODELS_DIR / "training_metadata.json") as f:
        training_metadata = json.load(f)
    TRAINED_FEATURES = training_metadata['player_model']['features']
    MODEL_VERSION = training_metadata.get('version', 'v4.0')
    INJURY_LEARNING_ENABLED = training_metadata.get('injury_learning_enabled', False)
except Exception as e:
    logger.warning(f"Metadata load failed ({e}), using full feature set fallback")
    TRAINED_FEATURES = [
        'MIN_PG', 'USG_PCT', 'TS_PCT', 'FTA_PG', 'AST_PG',
        'FG3A_PG', 'PER', 'FG_PCT', 'FG3_PCT', 'AGE', 'PACE',
        'REB_PG', 'STL_PG', 'BLK_PG', 'TOV_PG', 'FG3M_PG',
        'GP_RATIO', 'AVAILABILITY_SCORE', 'RECOVERY_FACTOR',
        'AGE_INJURY_RISK', 'INJURY_RISK_SCORE', 'DAYS_SINCE_INJURY',
        'INJURY_COUNT_LAST_YEAR', 'CHRONIC_INJURY_FLAG'
    ]
    MODEL_VERSION = 'v4.0'
    INJURY_LEARNING_ENABLED = True  # Assume full capability

logger.info(f"Model Version: {MODEL_VERSION} | Injury Learning: {INJURY_LEARNING_ENABLED}")

# ==================== SMART CACHING SYSTEM ====================
class PredictionCache:
    """In-memory cache with TTL for predictions"""
    def __init__(self):
        self._cache = {}
        self._hits = 0
        self._misses = 0
        
    def get(self, key: str, max_age: timedelta = PREDICTION_CACHE_DURATION):
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < max_age:
                self._hits += 1
                return value
        self._misses += 1
        return None
    
    def set(self, key: str, value):
        self._cache[key] = (datetime.now(), value)
    
    def clear_old(self, max_age: timedelta = PREDICTION_CACHE_DURATION):
        """Remove stale cache entries"""
        now = datetime.now()
        keys_to_delete = [
            k for k, (ts, _) in self._cache.items() 
            if now - ts > max_age
        ]
        for k in keys_to_delete:
            del self._cache[k]
        return len(keys_to_delete)
    
    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return round(self._hits / total * 100, 1) if total > 0 else 0

prediction_cache = PredictionCache()

# ==================== INJURY MANAGEMENT ====================
INJURY_STATUS = {}
LAST_INJURY_UPDATE = datetime.now()

def load_injuries_from_csv():
    """Load injuries with normalization"""
    global INJURY_STATUS
    
    try:
        injuries_df = pd.read_csv(DATA_DIR / "injuries.csv")
        INJURY_STATUS = {}
        
        for _, row in injuries_df.iterrows():
            name = str(row['player_name']).strip().title()
            if name and name != "Nan":
                INJURY_STATUS[name] = {
                    "status": str(row['status']).strip(),
                    "injury_type": str(row.get('injury_type', 'Unknown')).strip(),
                    "team": str(row.get('team', 'UNK')),
                    "updated": datetime.now().isoformat()
                }
        
        logger.info(f"Loaded {len(INJURY_STATUS)} injury reports")
        return True
    except Exception as e:
        logger.error(f"Failed to load injuries: {e}")
        return False

def refresh_injuries():
    """Refresh injury data from CSV"""
    global LAST_INJURY_UPDATE
    
    success = load_injuries_from_csv()
    if success:
        LAST_INJURY_UPDATE = datetime.now()
        cleared = prediction_cache.clear_old()
        logger.info(f"Injuries refreshed | Cache cleared: {cleared} entries")
    return success

@lru_cache(maxsize=1024)
def get_injury_status_cached(player_name: str) -> str:
    """Cached version of injury lookup"""
    return json.dumps(get_injury_status(player_name))

def get_injury_status(player_name: str) -> Dict:
    """Enhanced fuzzy matching for injury status"""
    key = player_name.strip().title()
    
    # Direct match
    if key in INJURY_STATUS:
        return INJURY_STATUS[key]
    
    # Substring match (fast)
    for injury_name in INJURY_STATUS:
        if key in injury_name or injury_name in key:
            return INJURY_STATUS[injury_name]
    
    # Fuzzy match with similarity score
    best_match = None
    best_score = 0
    
    for injury_name in INJURY_STATUS:
        similarity = SequenceMatcher(None, key.lower(), injury_name.lower()).ratio()
        if similarity > best_score and similarity > 0.80:  # 80% threshold
            best_score = similarity
            best_match = injury_name
    
    if best_match:
        logger.info(f"Fuzzy matched: '{key}' → '{best_match}' ({best_score:.0%})")
        return INJURY_STATUS[best_match]
    
    return {"status": "Active", "injury_type": "None", "team": ""}

# Initial load
load_injuries_from_csv()

# ==================== FEATURE ENGINEERING ====================
@lru_cache(maxsize=512)
def calculate_features_from_row_cached(player_name: str) -> str:
    """Cached feature calculation"""
    matches = df_players[df_players['PLAYER_NAME'] == player_name]
    if matches.empty:
        return None
    return json.dumps(calculate_features_from_row(matches.iloc[0]))

def calculate_features_from_row(player_row):
    """Optimized feature calculation with vectorization"""
    try:
        # Convert to float safely
        min_pg   = float(player_row.get('MIN', 33.0))
        pts_pg   = float(player_row.get('PTS', 25.0))
        fga_pg   = float(player_row.get('FGA', 20.0))
        fg3a_pg  = float(player_row.get('FG3A', 8.0))
        fta_pg   = float(player_row.get('FTA', 8.0))
        ast_pg   = float(player_row.get('AST', 6.0))
        reb_pg   = float(player_row.get('REB', 8.0))
        stl_pg   = float(player_row.get('STL', 1.0))
        blk_pg   = float(player_row.get('BLK', 0.5))
        tov_pg   = float(player_row.get('TOV', 3.0))
        fg_pct   = float(player_row.get('FG_PCT', 0.47))
        fg3_pct  = float(player_row.get('FG3_PCT', 0.36))
        age      = float(player_row.get('AGE', 27))

        # Pace
        if 'PACE' in player_row and pd.notna(player_row.get('PACE')):
            pace = float(player_row['PACE'])
        else:
            pace = 100.0

        # True Shooting Percentage (TS%)
        ts_pct = pts_pg / (2 * (fga_pg + 0.44 * fta_pg)) if (fga_pg + 0.44 * fta_pg) > 0 else 0.55
        ts_pct = np.clip(ts_pct, 0.40, 0.80)

        # Usage Percentage (USG%)
        if 'USG_PCT' in player_row and pd.notna(player_row['USG_PCT']):
            usg_pct = float(player_row['USG_PCT'])
        else:
            poss = fga_pg + 0.44 * fta_pg + tov_pg
            usg_pct = (poss / min_pg) * 48 * 5 if min_pg > 0 else 25.0
        usg_pct = np.clip(usg_pct, 12.0, 42.0)

        # Player Efficiency Rating (PER) approximation
        fgm = fga_pg * fg_pct
        ft_pct = float(player_row.get('FT_PCT', 0.80))
        ftm = fta_pg * ft_pct
        
        per = (pts_pg + reb_pg + ast_pg + 3 * (stl_pg + blk_pg) - 
               tov_pg - (fga_pg - fgm) - (fta_pg - ftm))
        per = np.clip(per, 5.0, 38.0)

        features = {
            'PTS_PG': round(pts_pg, 1), 
            'MIN_PG': round(min_pg, 1), 
            'USG_PCT': round(usg_pct, 1), 
            'TS_PCT': round(ts_pct, 3), 
            'FTA_PG': round(fta_pg, 1), 
            'AST_PG': round(ast_pg, 1),
            'FG3A_PG': round(fg3a_pg, 1), 
            'PER': round(per, 1), 
            'FG_PCT': round(fg_pct, 3), 
            'FG3_PCT': round(fg3_pct, 3), 
            'AGE': int(age), 
            'PACE': round(pace, 1),
            'REB_PG': round(reb_pg, 1), 
            'STL_PG': round(stl_pg, 1),
            'BLK_PG': round(blk_pg, 1), 
            'TOV_PG': round(tov_pg, 1),
            'FG3M_PG': round(fg3a_pg * fg3_pct, 1),
        }

        # ENHANCED: Add injury-aware features if available
        if INJURY_LEARNING_ENABLED:
            for feat in ['GP_RATIO', 'AVAILABILITY_SCORE', 'RECOVERY_FACTOR', 
                        'AGE_INJURY_RISK', 'INJURY_RISK_SCORE', 'DAYS_SINCE_INJURY',
                        'INJURY_COUNT_LAST_YEAR', 'CHRONIC_INJURY_FLAG']:
                if feat in player_row and pd.notna(player_row[feat]):
                    features[feat] = float(player_row[feat])
        
        return features
        
    except Exception as e:
        logger.error(f"Feature calculation error: {e}")
        return None

def build_feature_vector(features_dict):
    """OPTIMIZED: Build feature vector matching trained model"""
    # Use exact features from training metadata
    values = []
    for col in TRAINED_FEATURES:
        val = features_dict.get(col)
        if val is None:
            # Smart defaults based on feature type
            if 'PCT' in col or 'RATIO' in col or 'SCORE' in col:
                val = 0.5
            elif 'DAYS' in col:
                val = 999
            elif 'COUNT' in col or 'FLAG' in col:
                val = 0
            else:
                val = 0.0
        values.append(float(val))
    
    return np.array([values], dtype=np.float32)

# ==================== FASTAPI APP ====================
app = FastAPI(
    title=f"NBA Betting Agent Pro — {MODEL_VERSION}", 
    version="4.0.0",
    description="Production-ready NBA prediction API with smart caching"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==================== BACKGROUND TASKS ====================
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    
    async def auto_refresh_injuries():
        """Auto-refresh injuries every hour"""
        while True:
            await asyncio.sleep(3600)  # 1 hour
            
            logger.info(f"[Auto-Refresh] Checking injuries at {datetime.now()}")
            
            try:
                from utils import fetch_live_injuries
                injuries = fetch_live_injuries()
                
                if injuries and len(injuries) > 0:
                    df = pd.DataFrame(injuries)
                    df.to_csv(DATA_DIR / "injuries.csv", index=False)
                    refresh_injuries()
                    logger.info(f"Auto-updated {len(injuries)} injuries")
                    
            except Exception as e:
                logger.error(f"Auto-refresh failed: {e}")
    
    async def cleanup_cache():
        """Clean up stale cache entries every 30 minutes"""
        while True:
            await asyncio.sleep(1800)  # 30 minutes
            cleared = prediction_cache.clear_old()
            logger.info(f"Cache cleanup: removed {cleared} stale entries")
    
    asyncio.create_task(auto_refresh_injuries())
    asyncio.create_task(cleanup_cache())
    logger.info("Background tasks started")

# ==================== REQUEST MODELS ====================
class PlayerRequest(BaseModel):
    player_name: str
    opponent_abbr: str = "AVG"

class GamePredictionRequest(BaseModel):
    home_team: str
    away_team: str
    spread_line: Optional[float] = None
    total_line: Optional[float] = None

class ParlayLeg(BaseModel):
    player: Optional[str] = None
    stat: Optional[str] = "points"
    line: Optional[float] = None
    over: Optional[bool] = True
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    bet_type: Optional[str] = None
    side: Optional[str] = None

# Build normalized lookup once at startup (after df_players is loaded)
PLAYER_NAME_LOOKUP = {
    normalize_name_lower(name): name
    for name in df_players['PLAYER_NAME'] if pd.notna(name)
}

# ==================== IMPROVED PROBABILITY MODELS ====================
def edge_to_prob_old(edge, weight=0.04):
    return np.clip(0.5 + edge * weight, 0.15, 0.85)

def edge_to_prob(edge: float, weight: float = 0.05) -> float:
    """
    OPTIMIZED: Convert edge to probability using sigmoid
    More realistic probability distribution
    """
    # Sigmoid function for smoother probability curve
    prob = 1 / (1 + np.exp(-edge * weight))
    return np.clip(prob, 0.20, 0.85)

def calculate_kelly_stake(prob: float, decimal_odds: float, bankroll: float = 100) -> Dict:
    """
    Calculate optimal Kelly Criterion stake
    """
    edge = (prob * decimal_odds) - 1
    
    if edge <= 0:
        return {"kelly_fraction": 0, "recommended_stake": 0, "note": "No edge"}
    
    # Kelly formula: (probability * odds - 1) / (odds - 1)
    kelly_fraction = edge / (decimal_odds - 1)
    
    # Use fractional Kelly (safer)
    fractional_kelly = kelly_fraction * 0.25  # 1/4 Kelly
    
    recommended_stake = round(bankroll * fractional_kelly, 2)
    
    return {
        "kelly_fraction": round(kelly_fraction, 4),
        "fractional_kelly": round(fractional_kelly, 4),
        "recommended_stake": recommended_stake,
        "bankroll": bankroll
    }

# ==================== ENHANCED PLAYER PREDICTION ====================
@app.post("/predict")
def predict_player_ml(req: PlayerRequest):
    """OPTIMIZED: Player prediction with advanced matchup analysis"""

    raw_name = req.player_name.strip()
    opponent = req.opponent_abbr.upper() or "AVG"

    # Name resolution
    canonical_name = get_canonical_name(raw_name)
    if not canonical_name:
        return {"error": "Invalid player name", "input": raw_name}

    norm_key = normalize_name_lower(canonical_name)

    # Check cache first
    cache_key = f"player_{canonical_name}_{opponent}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = prediction_cache.get(cache_key)
    if cached:
        cached['cached'] = True
        return cached

    # Injury check
    injury = get_injury_status(canonical_name)

    # Block OUT/Doubtful players
    if injury['status'] in ['OUT', 'Doubtful']:
        result = {
            "player": canonical_name,
            "input_name": raw_name if raw_name.lower() != canonical_name.lower() else None,
            "team": injury.get("team", "UNK"),
            "opponent": opponent,
            "status": injury['status'],
            "injury_type": injury['injury_type'],
            "recommendation": "AVOID - Player is OUT",
            "projected_pts": 0.0,
            "confidence": "N/A",
            "matchup": "N/A",
            "cached": False
        }
        prediction_cache.set(cache_key, result)
        return result

    # Find player in database
    player_name_in_df = None
    if norm_key in PLAYER_NAME_LOOKUP:
        player_name_in_df = PLAYER_NAME_LOOKUP[norm_key]
    else:
        from difflib import get_close_matches
        close = get_close_matches(norm_key, PLAYER_NAME_LOOKUP.keys(), n=1, cutoff=0.78)
        if close:
            player_name_in_df = PLAYER_NAME_LOOKUP[close[0]]

    if not player_name_in_df:
        return {
            "player": canonical_name,
            "input_name": raw_name if raw_name != canonical_name else None,
            "status": injury['status'],
            "recommendation": "NO STATS - Player has no 2025-26 data (rookie/injured?)",
            "projected_pts": 0.0,
            "confidence": "N/A",
            "note": "Possibly injured or new player"
        }

    # Get player data
    player_row = df_players[df_players['PLAYER_NAME'] == player_name_in_df]
    if player_row.empty:
        raise HTTPException(500, "Internal error")

    p = player_row.iloc[0]
    team_abbr = p.get('TEAM_ABBREVIATION', 'UNK')

    # fallback prediction cache
    cached = prediction_cache.get(cache_key)
    if cached:
        return cached

    # Confidence from injury
    confidence = "High"
    if injury['status'] == "Questionable":
        confidence = "Low"
    elif injury['status'] == "Probable":
        confidence = "Medium"

    # Calculate features
    features_dict = calculate_features_from_row(p)
    if not features_dict:
        raise HTTPException(500, "Feature error")

    season_avg = features_dict['PTS_PG']

    # Build feature vector and predict
    features = build_feature_vector(features_dict)
    pts_raw = float(player_model.predict(features)[0])

    # IMPROVED: Smart clipping based on consistency
    consistency_factor = 1.0
    if 'GP' in p and p['GP'] < 10:
        consistency_factor = 0.8  # Less data = more conservative
    
    pts_projection = np.clip(
        pts_raw, 
        season_avg * 0.5 * consistency_factor, 
        season_avg * 1.5 * consistency_factor
    )

    # ENHANCED: Matchup analysis
    matchup_factor = 1.0
    matchup_quality = "Neutral"
    opp_details = {}
    
    if opponent != "AVG":
        opp_row = df_teams[df_teams['TEAM_ABBREVIATION'] == opponent]
        if not opp_row.empty:
            opp_def = float(opp_row['DEF_RATING'].iloc[0])
            league_avg_def = df_teams['DEF_RATING'].mean()
            
            opp_details = {
                "def_rating": round(opp_def, 1),
                "league_avg": round(league_avg_def, 1),
                "rank": int((df_teams['DEF_RATING'] < opp_def).sum() + 1)
            }
            
            # More nuanced matchup adjustment
            if opp_def < league_avg_def - 3:
                matchup_factor = 1.10
                matchup_quality = "⭐ Very Favorable"
            elif opp_def < league_avg_def - 1:
                matchup_factor = 1.05
                matchup_quality = "✓ Favorable"
            elif opp_def > league_avg_def + 3:
                matchup_factor = 0.90
                matchup_quality = "⚠️ Tough"
            elif opp_def > league_avg_def + 1:
                matchup_factor = 0.95
                matchup_quality = "Challenging"
            
            pts_projection *= matchup_factor

    # ENHANCED: Injury-adjusted confidence
    confidence_score = 100.0
    confidence_factors = []
    
    if injury['status'] == "Questionable":
        pts_projection *= 0.82
        confidence_score *= 0.60
        confidence_factors.append("Questionable status (-40%)")
    elif injury['status'] == "Probable":
        pts_projection *= 0.93
        confidence_score *= 0.85
        confidence_factors.append("Probable status (-15%)")
    
    # Games played factor
    if 'GP' in p and p['GP'] < 10:
        confidence_score *= 0.85
        confidence_factors.append(f"Limited games ({p['GP']} GP)")
    
    # Recent injury history (if available)
    if 'RECOVERY_FACTOR' in features_dict and features_dict['RECOVERY_FACTOR'] < 1.0:
        recovery = features_dict['RECOVERY_FACTOR']
        pts_projection *= recovery
        confidence_score *= recovery
        confidence_factors.append(f"Recovering from injury ({recovery:.0%})")

    pts_projection = round(pts_projection, 1)
    
    # Confidence label
    if confidence_score >= 90:
        confidence = "🟢 Very High"
    elif confidence_score >= 75:
        confidence = "🟡 High"
    elif confidence_score >= 60:
        confidence = "🟠 Medium"
    else:
        confidence = "🔴 Low"

    # IMPROVED: Recommendation logic
    recommendation = "🔵 MONITOR"
    bet_suggestion = None
    
    diff_from_avg = pts_projection - season_avg
    diff_pct = (diff_from_avg / season_avg) * 100
    
    if confidence_score >= 75:  # Only recommend if confident
        if diff_pct > 8:
            recommendation = f"✅ BET OVER {pts_projection - 0.5:.1f}"
            bet_suggestion = {
                "type": "OVER",
                "line": pts_projection - 0.5,
                "edge": round(diff_pct, 1),
                "reasoning": f"+{diff_pct:.1f}% vs season avg"
            }
        elif diff_pct < -8:
            recommendation = f"✅ BET UNDER {pts_projection + 0.5:.1f}"
            bet_suggestion = {
                "type": "UNDER",
                "line": pts_projection + 0.5,
                "edge": round(abs(diff_pct), 1),
                "reasoning": f"{diff_pct:.1f}% vs season avg"
            }
        elif abs(diff_pct) < 3:
            recommendation = "⚪ NO BET - Too close to average"

    result = {
        "player": player_name_in_df,
        "input_name": raw_name if raw_name.lower() != player_name_in_df.lower() else None,
        "team": team_abbr,
        "opponent": opponent,
        "status": injury['status'],
        "injury_type": injury.get('injury_type', 'None'),
        
        # Core predictions
        "projected_pts": pts_projection,
        "season_avg_pts": round(season_avg, 1),
        "projection_vs_avg": f"{diff_pct:+.1f}%",
        
        # Confidence breakdown
        "confidence": confidence,
        "confidence_score": round(confidence_score, 1),
        "confidence_factors": confidence_factors if confidence_factors else ["No concerns"],
        
        # Matchup
        "matchup": matchup_quality,
        "matchup_factor": f"{matchup_factor:.2f}x",
        "opponent_defense": opp_details if opp_details else None,
        
        # Betting
        "recommendation": recommendation,
        "bet_suggestion": bet_suggestion,
        
        # Additional stats
        "rebounds_per_game": round(features_dict.get('REB_PG', 0), 1),
        "assists_per_game": round(features_dict.get('AST_PG', 0), 1),
        "threes_made_per_game": round(features_dict.get('FG3M_PG', 0), 1),
        
        # Meta
        "cached": False,
        "model_version": MODEL_VERSION,
        "injury_learning": INJURY_LEARNING_ENABLED
    }

    prediction_cache.set(cache_key, result)
    return result

# ==================== ENHANCED GAME PREDICTION ====================
@app.post("/predict-game")
def predict_game_ml(req: GamePredictionRequest):
    """OPTIMIZED: Game prediction with advanced analytics"""
    
    cache_key = f"game_{req.home_team}_{req.away_team}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = prediction_cache.get(cache_key, max_age=GAME_CACHE_DURATION)
    if cached:
        cached['cached'] = True
        return cached
    
    logger.info(f"predict_game_ml called with home: {req.home_team}, away: {req.away_team}")
    logger.info(f"Available teams: {sorted(df_teams['TEAM_ABBREVIATION'].unique())}")

    home = df_teams[df_teams['TEAM_ABBREVIATION'] == req.home_team.upper()]
    away = df_teams[df_teams['TEAM_ABBREVIATION'] == req.away_team.upper()]

    logger.info(f"Home found: {not home.empty}, Away found: {not away.empty}")
    
    if home.empty or away.empty:
        raise HTTPException(404, "Team not found")

    h, a = home.iloc[0], away.iloc[0]
    
    # Get team stats
    home_off = float(h.get('OFF_RATING', 115)) + 3.5  # Home court advantage
    home_def = float(h.get('DEF_RATING', 110))
    home_pace = float(h.get('PACE', 100))
    away_off = float(a.get('OFF_RATING', 112))
    away_def = float(a.get('DEF_RATING', 113))
    away_pace = float(a.get('PACE', 99))
    
    # Predict total
    features = np.array([[home_off, home_def, home_pace, away_off, away_def, away_pace]], dtype=np.float32)
    total_pred = float(team_model.predict(features)[0])
    
    # Calculate spread
    avg_pace = (home_pace + away_pace) / 2
    home_expected_pts = ((home_off + away_def) / 2) * avg_pace / 100
    away_expected_pts = ((away_off + home_def) / 2) * avg_pace / 100
    spread_pred = home_expected_pts - away_expected_pts

    # Win probability (more sophisticated)
    win_prob_home = 1 / (1 + np.exp(-spread_pred / 3.5))

    # Pace analysis
    pace_diff = abs(home_pace - away_pace)
    pace_note = "Similar pace" if pace_diff < 2 else "Pace mismatch" if pace_diff > 4 else "Slight pace difference"
    
    # Compare to market lines if provided
    spread_edge = None
    total_edge = None
    spread_recommendation = None
    total_recommendation = None
    spread_kelly = None
    total_kelly = None
    
    if req.spread_line is not None:
        spread_edge = spread_pred - req.spread_line
        # spread_edge = abs(spread_pred - req.spread_line)
        # if spread_edge > 3:
        #     if spread_pred > req.spread_line:
        #         spread_recommendation = f"BET {req.home_team} {req.spread_line}"
        #     else:
        #         spread_recommendation = f"BET {req.away_team} +{abs(req.spread_line)}"
        if abs(spread_edge) > 2.5:
            prob = edge_to_prob(abs(spread_edge), weight=0.08)
            decimal_odds = round(1 / prob, 2)
            
            if spread_pred > req.spread_line:
                spread_recommendation = f"✅ BET {req.home_team} {req.spread_line:+.1f}"
                side = req.home_team
            else:
                spread_recommendation = f"✅ BET {req.away_team} {-req.spread_line:+.1f}"
                side = req.away_team
            
            spread_kelly = calculate_kelly_stake(prob, decimal_odds)
            spread_kelly.update({
                "side": side,
                "probability": round(prob * 100, 1),
                "decimal_odds": decimal_odds
            })
    
    if req.total_line is not None:
        total_edge = total_pred - req.total_line
        
        if abs(total_edge) > 4:
            prob = edge_to_prob(abs(total_edge), weight=0.06)
            decimal_odds = round(1 / prob, 2)
            direction = "OVER" if total_pred > req.total_line else "UNDER"
            
            total_recommendation = f"✅ BET {direction} {req.total_line}"
            
            total_kelly = calculate_kelly_stake(prob, decimal_odds)
            total_kelly.update({
                "side": direction,
                "probability": round(prob * 100, 1),
                "decimal_odds": decimal_odds
            })

    result = {
        "game": f"{req.away_team.upper()} @ {req.home_team.upper()}",
        
        # Core predictions
        "projected_spread": round(spread_pred, 1),
        "projected_total": round(total_pred, 1),
        "win_prob_home": round(win_prob_home * 100, 1),
        "win_prob_away": round((1 - win_prob_home) * 100, 1),
        
        # Expected scores
        "expected_home_pts": round(home_expected_pts, 1),
        "expected_away_pts": round(away_expected_pts, 1),
        
        # Market comparison
        "spread_edge": round(spread_edge, 1) if spread_edge is not None else None,
        "total_edge": round(total_edge, 1) if total_edge is not None else None,
        "spread_recommendation": spread_recommendation,
        "total_recommendation": total_recommendation,
        
        # Kelly suggestions
        "spread_kelly": spread_kelly,
        "total_kelly": total_kelly,
        
        # Team analysis
        "pace_analysis": {
            "home_pace": round(home_pace, 1),
            "away_pace": round(away_pace, 1),
            "note": pace_note
        },
        "team_ratings": {
            "home": {"off": round(home_off - 3.5, 1), "def": round(home_def, 1)},
            "away": {"off": round(away_off, 1), "def": round(away_def, 1)}
        },
        
        "cached": False,
        "model_version": MODEL_VERSION
    }

    prediction_cache.set(cache_key, result)
    return result

# ==================== OPTIMIZED BEST BETS ====================
@app.get("/best-bets")
@app.post("/best-bets")
def best_bets(
    date: Optional[str] = None,
    days_offset: Optional[int] = None,
    min_edge: float = 2.5,  # Lowered threshold
    max_bets: int = 10,
    min_confidence: int = 70
    ):
    """OPTIMIZED: Get best betting opportunities with quality filtering"""
    
    if not date:
        offset = days_offset or 0
        target = (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
    else:
        target = date

    # Check cache
    cache_file = CACHE_DIR / f"best_bets_{target}_edge{min_edge}.json"
    if cache_file.exists():
        cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - cache_time < GAME_CACHE_DURATION:
            with open(cache_file) as f:
                cached_data = json.load(f)
                logger.info(f"Using cached best bets for {target}")
                return cached_data

    # Fetch games
    try:
        from utils import fetch_games_with_odds_for_date
        games_str = target.replace("-", "")
        games = fetch_games_with_odds_for_date(games_str)
    except Exception as e:
        logger.error(f"Game fetch error: {e}")
        games = []

    if not games:
        return {
            "date": target,
            "total_opportunities": 0,
            "best_bets": [],
            "message": "No games found for this date"
        }

    bets = []
    for game in games:
        try:
            home = game.get("home_team")
            away = game.get("away_team")
            spread = game.get("spread")
            total = game.get("total")
            
            if not home or not away:
                continue
            
            pred = predict_game_ml(GamePredictionRequest(
                home_team=home,
                away_team=away,
                spread_line=spread,
                total_line=total
            ))
            
            # SPREAD BET
            if pred.get('spread_edge') is not None and abs(pred['spread_edge']) >= min_edge:
                kelly = pred.get('spread_kelly', {})
                prob = kelly.get('probability', 50)
                
                if prob >= min_confidence:
                    bets.append({
                        "game": pred['game'],
                        "bet_type": "SPREAD",
                        "recommendation": pred['spread_recommendation'],
                        "edge": round(abs(pred['spread_edge']), 1),
                        "probability": prob,
                        "decimal_odds": kelly.get('decimal_odds', 1.91),
                        "ev": round((prob/100 * kelly.get('decimal_odds', 1.91)) - 1, 3),
                        "kelly_stake": kelly.get('recommended_stake', 0),
                        "our_projection": pred['projected_spread'],
                        "market_line": spread,
                        "confidence": "🟢 High" if prob >= 80 else "🟡 Medium",
                        "game_time": game.get('game_time', 'TBD'),
                        "quality_score": round(abs(pred['spread_edge']) * (prob/100), 2)
                    })
            
            # TOTAL BET
            if pred.get('total_edge') is not None and abs(pred['total_edge']) >= min_edge:
                kelly = pred.get('total_kelly', {})
                prob = kelly.get('probability', 50)
                
                if prob >= min_confidence:
                    bets.append({
                        "game": pred['game'],
                        "bet_type": "TOTAL",
                        "recommendation": pred['total_recommendation'],
                        "edge": round(abs(pred['total_edge']), 1),
                        "probability": prob,
                        "decimal_odds": kelly.get('decimal_odds', 1.91),
                        "ev": round((prob/100 * kelly.get('decimal_odds', 1.91)) - 1, 3),
                        "kelly_stake": kelly.get('recommended_stake', 0),
                        "our_projection": pred['projected_total'],
                        "market_line": total,
                        "confidence": "🟢 High" if prob >= 80 else "🟡 Medium",
                        "game_time": game.get('game_time', 'TBD'),
                        "quality_score": round(abs(pred['total_edge']) * (prob/100), 2)
                    })
                
        except Exception as e:
            logger.error(f"Error processing game: {e}")
            continue

    # Sort by quality score (edge * probability)
    bets.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

    # Calculate stats
    high_conf_count = len([b for b in bets if "🟢" in b.get('confidence', '')])
    avg_edge = round(np.mean([b['edge'] for b in bets]), 1) if bets else 0
    avg_prob = round(np.mean([b['probability'] for b in bets]), 1) if bets else 0

    result = {
        "date": target,
        "games_analyzed": len(games),
        "total_opportunities": len(bets),
        "best_bets": bets[:max_bets],
        "filters": {
            "min_edge": min_edge,
            "min_confidence": min_confidence
        },
        "statistics": {
            "high_confidence_count": high_conf_count,
            "average_edge": avg_edge,
            "average_probability": avg_prob,
            "total_kelly_recommended": round(sum([b.get('kelly_stake', 0) for b in bets[:max_bets]]), 2)
        },
        "model_version": MODEL_VERSION
    }

    # Cache the result
    with open(cache_file, 'w') as f:
        json.dump(result, f)

    return result

# ==================== ENHANCED PARLAY BUILDER ====================
@app.post("/build-parlay")
async def build_parlay(legs: List[ParlayLeg], bankroll: float = 100):
    """OPTIMIZED: Parlay builder with risk analysis and Kelly sizing"""
    if not legs:
        raise HTTPException(400, "No legs provided")

    total_decimal = 1.0
    details = []
    total_prob = 1.0

    for idx, leg in enumerate(legs, 1):
        leg_detail = {}
        leg_odds = 1.0
        prob = 0.5

        # ——— PLAYER PROP ———
        if leg.player:
            if leg.stat.lower() != "triple_double" and leg.line is None:
                raise HTTPException(400, f"Line required for {leg.player}")

            pred = predict_player_ml(PlayerRequest(player_name=leg.player, opponent_abbr="AVG"))

            player_status = pred.get("status", "Active")
            if player_status in ["OUT", "Doubtful"]:
                # SKIP the leg instead of crashing
                details.append({
                    "leg_number": idx,
                    "type": "player_prop",
                    "player": pred["player"],
                    "bet": f"{pred['player']} {'OVER' if leg.over else 'UNDER'} {leg.line} {leg.stat.upper()}",
                    "status": player_status,
                    "injury_type": pred.get("injury_type", "Unknown"),
                    "note": f"⚠️ SKIPPED - Player is {player_status}",
                    "skipped": True
                })
                continue  # Skip to next leg

            base = {
                "points": pred.get("projected_pts", 20.0),
                "rebounds": pred.get("rebounds_per_game", 6.0),
                "assists": pred.get("assists_per_game", 5.0),
                "steals": pred.get("steals_per_game", 0.8),
                "blocks": pred.get("blocks_per_game", 0.6),
                "threes": pred.get("threes_made_per_game", 2.0),
                "pra": pred.get("projected_pts", 20.0) + pred.get("rebounds_per_game", 6.0) + pred.get("assists_per_game", 5.0)
            }

            stat_map = {
                "points": "Points", "rebounds": "Rebounds", "assists": "Assists",
                "steals": "Steals", "blocks": "Blocks", "threes": "3PM", "pra": "PRA"
            }
            stat_key = leg.stat.lower()

            # === TRIPLE-DOUBLE SPECIAL CASE ===
            if stat_key == "triple_double":
                pts = base["points"]
                reb = base["rebounds"]
                ast = base["assists"]

                # Simple but effective approximation: assume normal distribution around projection
                # P(stat >= 10) ≈ 1 - CDF(9.5, mean=proj, std=proj*0.35)
                import math

                def prob_ge_10(mean):
                    if mean < 5: return 0.0
                    if mean > 20: return 1.0
                    # Approximate using normal: mean=proj, std ≈ 35% of mean
                    std = mean * 0.35
                    if std == 0: return 1.0 if mean >= 10 else 0.0
                    z = (9.5 - mean) / std
                    # Very rough erf approximation for CDF
                    return 0.5 * (1 + math.tanh(-z * 0.8))  # Good enough for ranking

                p_pts = prob_ge_10(pts)
                p_reb = prob_ge_10(reb)
                p_ast = prob_ge_10(ast)

                prob = p_pts * p_reb * p_ast
                prob = max(prob, 0.001)  # Avoid division by zero

                leg_odds = round(max(1.0 / prob, 1.10), 2)

                leg_detail = {
                    "leg_number": idx,
                    "type": "player_prop",
                    "player": pred["player"],
                    "bet": f"{pred['player']} {'YES' if leg.over else 'NO'} Triple-Double",
                    "projection": f"PTS: {pts:.1f} | REB: {reb:.1f} | AST: {ast:.1f}",
                    "probability": round(prob * 100, 1),
                    "decimal_odds": leg_odds,
                    "ev": round((prob * leg_odds) - 1, 3),
                    "status": pred.get("status", "Active"),
                    "confidence": pred.get("confidence", "Medium"),
                    "note": "Independent probability (no correlation modeled)"
                }

            # === REGULAR STATS (points, pra, etc.) ===
            else:
                stat_map = {
                    "points": "Points", "rebounds": "Rebounds", "assists": "Assists",
                    "steals": "Steals", "blocks": "Blocks", "threes": "3PM", "pra": "PRA"
                }
                if stat_key not in base:
                    stat_key = "points"

                projection = base[stat_key]
                edge = (projection - leg.line) if leg.over else (leg.line - projection)
                prob = edge_to_prob(edge, weight=0.06)
                leg_odds = round(max(1.0 / prob, 1.05), 2)

                leg_detail = {
                    "leg_number": idx,
                    "type": "player_prop",
                    "player": pred["player"],
                    "bet": f"{pred['player']} {'OVER' if leg.over else 'UNDER'} {leg.line} {stat_map.get(stat_key, stat_key.upper())}",
                    "projection": round(projection, 1),
                    "edge": round(edge, 1),
                    "probability": round(prob * 100, 1),
                    "decimal_odds": leg_odds,
                    "status": pred.get("status", "Active"),
                    "confidence": pred.get("confidence", "Medium")
                }

        # ——— GAME BETS ———
        elif leg.home_team and leg.away_team and leg.bet_type:
            home = leg.home_team.upper()
            away = leg.away_team.upper()
            
            try:
                game_pred = predict_game_ml(GamePredictionRequest(home_team=home, away_team=away))
            except Exception as e:
                logger.error(f"Game prediction failed for {away} @ {home}: {str(e)}")
                continue  # Gracefully skip invalid games

            if leg.bet_type == "spread" and leg.line is not None:
                proj = game_pred["projected_spread"]
                edge = abs(proj - leg.line)
                side = home if proj > leg.line else away
                display_line = leg.line if proj > leg.line else -leg.line
                prob = edge_to_prob(edge, weight=0.07)
                leg_odds = round(max(1.0 / prob, 1.10), 2)

                leg_detail = {
                    "leg_number": idx,
                    "type": "spread",
                    "bet": f"{side} {display_line:+.1f}",
                    "game": f"{away} @ {home}",
                    "projection": round(proj, 1),
                    "edge": round(edge, 1),
                    "probability": round(prob * 100, 1),
                    "decimal_odds": leg_odds
                }

            elif leg.bet_type == "total" and leg.line is not None:
                proj = game_pred["projected_total"]
                direction = "OVER" if proj > leg.line else "UNDER"
                edge = abs(proj - leg.line)
                prob = edge_to_prob(edge, weight=0.06)
                leg_odds = round(max(1.0 / prob, 1.10), 2)

                leg_detail = {
                    "leg_number": idx,
                    "type": "total",
                    "bet": f"{direction} {leg.line}",
                    "game": f"{away} @ {home}",
                    "projection": round(proj, 1),
                    "edge": round(edge, 1),
                    "probability": round(prob * 100, 1),
                    "decimal_odds": leg_odds
                }

            elif leg.bet_type == "moneyline":
                prob_home = game_pred["win_prob_home"] / 100
                prob_away = 1 - prob_home

                if leg.side == "HOME":
                    team = home
                    prob = prob_home
                elif leg.side == "AWAY":
                    team = away
                    prob = prob_away
                else:
                    # AI mode
                    team = home if prob_home > prob_away else away
                    prob = max(prob_home, prob_away)

                prob = np.clip(prob, 0.15, 0.85)
                leg_odds = round(max(1.0 / prob, 1.10), 2)
                ev = round((prob * leg_odds) - 1, 3)

                leg_detail = {
                    "leg_number": idx,
                    "type": "moneyline",
                    "bet": f"{team} Moneyline",
                    "who_to_bet": f"BET {team} TO WIN",
                    "game": f"{away} @ {home}",
                    "win_probability": round(prob * 100, 1),
                    "decimal_odds": leg_odds,
                    "ev": ev,
                    "warning": "Negative EV bet" if ev < 0 else None,
                    "mode": "Manual" if leg.side else "AI-selected side"
                }

        if leg_detail:
            total_decimal *= leg_odds
            total_prob *= prob
            details.append(leg_detail)

    # Calculate parlay metrics
    parlay_ev = round((total_prob * total_decimal) - 1, 3)
    kelly = calculate_kelly_stake(total_prob, total_decimal, bankroll)

    # Risk assessment
    if total_decimal > 15:
        risk_level = "🔴 Very High"
        risk_note = "Lottery ticket - entertainment only"
    elif total_decimal > 8:
        risk_level = "🟠 High"
        risk_note = "High variance - small stakes recommended"
    elif total_decimal > 4:
        risk_level = "🟡 Medium"
        risk_note = "Moderate risk - suitable for recreational betting"
    else:
        risk_level = "🟢 Low"
        risk_note = "Conservative parlay - good risk/reward balance"

    skipped_legs = [d for d in details if d.get("skipped")]
    active_legs = [d for d in details if not d.get("skipped")]

    return {
        "parlay_odds": round(total_decimal, 2),
        "parlay_probability": round(total_prob * 100, 1),
        "expected_value": parlay_ev,
        "kelly_recommendation": kelly,
        "possible_win_per_100": round((total_decimal - 1) * 100, 2),
        "break_even_win_rate": round((1 / total_decimal) * 100, 1),
        "total_legs": len(details),
        "total_legs_requested": len(legs),
        "active_legs": len(active_legs),
        "skipped_legs_count": len(skipped_legs),
        "legs": details,
        "risk_assessment": {
            "risk_level": risk_level,
            "note": risk_note,
            "recommendation": "✅ PLACE BET" if parlay_ev > 0.05 and total_decimal < 10 
                else "⚠️ PROCEED WITH CAUTION" if parlay_ev > 0 
                else "❌ AVOID - Negative EV"
        },
        "model_version": MODEL_VERSION
    }

# ==================== DATA MANAGEMENT ====================
@app.post("/refresh-injuries")
def refresh_injuries_endpoint(background_tasks: BackgroundTasks):
    """Manually trigger injury refresh"""
    
    def do_refresh():
        try:
            from utils import fetch_live_injuries
            injuries = fetch_live_injuries()
            if injuries:
                pd.DataFrame(injuries).to_csv(DATA_DIR / "injuries.csv", index=False)
                refresh_injuries()
        except Exception as e:
            logger.error(f"Refresh failed: {e}")
    
    background_tasks.add_task(do_refresh)
    return {
        "message": "Refresh initiated", 
        "current_injuries": len(INJURY_STATUS),
        "last_update": LAST_INJURY_UPDATE.isoformat()
    }

@app.get("/injuries")
def get_injuries(status: Optional[str] = None):
    """Get current injury reports"""
    
    injuries = [{"player": p, **info} for p, info in INJURY_STATUS.items()]
    
    if status:
        injuries = [inj for inj in injuries if inj['status'].lower() == status.lower()]
    
    return {
        "total": len(injuries),
        "injuries": sorted(injuries, key=lambda x: x['status']),
        "last_update": LAST_INJURY_UPDATE.isoformat(),
        "statuses": {
            "OUT": len([i for i in injuries if i['status'] == 'OUT']),
            "Questionable": len([i for i in injuries if i['status'] == 'Questionable']),
            "Probable": len([i for i in injuries if i['status'] == 'Probable'])
        }
    }

@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics"""
    return {
        "cache_size": len(prediction_cache._cache),
        "cache_hit_rate": f"{prediction_cache.hit_rate}%",
        "total_hits": prediction_cache._hits,
        "total_misses": prediction_cache._misses,
        "last_cleanup": "Auto every 30 min",
        "injury_update": LAST_INJURY_UPDATE.isoformat()
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": MODEL_VERSION,
        "injury_learning": INJURY_LEARNING_ENABLED,
        "players_loaded": len(df_players),
        "teams_loaded": len(df_teams),
        "injuries_tracked": len(INJURY_STATUS),
        "cache_hit_rate": f"{prediction_cache.hit_rate}%",
        "last_injury_update": LAST_INJURY_UPDATE.isoformat()
    }

@app.get("/players")
def list_players(limit: int = 50, search: Optional[str] = None):
    """List available players"""
    players_list = df_players['PLAYER_NAME'].tolist()
    
    if search:
        players_list = [p for p in players_list if search.lower() in p.lower()]
    
    return {
        "total": len(players_list),
        "players": sorted(players_list[:limit])
    }

@app.get("/teams")
def list_teams():
    """List all NBA teams"""
    teams_list = df_teams[['TEAM_ABBREVIATION', 'OFF_RATING', 'DEF_RATING', 'PACE']].to_dict('records')
    return {
        "total": len(teams_list),
        "teams": sorted(teams_list, key=lambda x: x['OFF_RATING'], reverse=True)
    }

@app.get("/")
def root():
    """API root endpoint"""
    return {
            "service": f"NBA Betting Agent Pro — {MODEL_VERSION}",
            "version": "4.0.0 Optimized",
            "status": "running",
            "features": {
            "injury_learning": INJURY_LEARNING_ENABLED,
            "kelly_criterion": True,
            "advanced_matchups": True,
            "confidence_scoring": True
        },
        "endpoints": {
            "predict": "POST /predict - Player predictions with confidence",
            "predict_game": "POST /predict-game - Game predictions with Kelly stakes",
            "best_bets": "GET /best-bets - Quality-filtered opportunities",
            "build_parlay": "POST /build-parlay - Parlay builder with risk analysis",
            "injuries": "GET /injuries - Current injury reports",
            "health": "GET /health - System health check"
        },
        "cache": {
            "size": len(prediction_cache._cache),
            "hit_rate": f"{prediction_cache.hit_rate}%"
        },
        "documentation": "/docs"
    }