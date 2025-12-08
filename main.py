# main.py — PRODUCTION-READY with Caching, Smart Predictions & Scalability
import functools
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
    logger.info(f"Data loaded | Players: {len(df_players)} | Teams: {len(df_teams)}")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    raise

# Load metadata
try:
    with open(MODELS_DIR / "training_metadata.json") as f:
        training_metadata = json.load(f)
    TRAINED_FEATURES = training_metadata['player_model']['features']
    MODEL_VERSION = training_metadata.get('version', 'v3.1')
except Exception as e:
    TRAINED_FEATURES = [
        'MIN_PG', 'USG_PCT', 'TS_PCT', 'FTA_PG', 'AST_PG',
        'FG3A_PG', 'PER', 'FG_PCT', 'FG3_PCT', 'AGE', 'PACE'
    ]
    MODEL_VERSION = 'v3.1'

# Optional: pre-build a normalized player lookup for speed
@functools.lru_cache(maxsize=1)
def _build_player_lookup():
    """Pre-normalize all player names for ultra-fast lookup"""
    lookup = {}
    for name in df_players['PLAYER_NAME']:
        norm = normalize_name_lower(name)
        lookup[norm] = name  # maps normalized → original/official name
    return lookup

PLAYER_NAME_LOOKUP = _build_player_lookup()

# ==================== SMART CACHING SYSTEM ====================
class PredictionCache:
    """In-memory cache with TTL for predictions"""
    def __init__(self):
        self._cache = {}
        
    def get(self, key: str, max_age: timedelta = PREDICTION_CACHE_DURATION):
        if key in self._cache:
            timestamp, value = self._cache[key]
            if datetime.now() - timestamp < max_age:
                return value
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
        prediction_cache.clear_old()  # Clear cache when injuries update
        logger.info(f"Injuries refreshed at {datetime.now().strftime('%H:%M:%S')}")
    return success

@lru_cache(maxsize=1024)
def get_injury_status_cached(player_name: str) -> str:
    """Cached version of injury lookup"""
    return json.dumps(get_injury_status(player_name))

def get_injury_status(player_name: str) -> Dict:
    """Enhanced fuzzy matching with accent/diacritic normalization"""
    if not player_name:
        return {"status": "Active", "injury_type": "None", "team": ""}

    # Normalize input name
    normalized_input = normalize_name_lower(player_name)
    canonical_input = get_canonical_name(player_name)

    # Build normalized injury lookup if not exists
    global _normalized_injury_cache
    if '_normalized_injury_cache' not in globals():
        _normalized_injury_cache = {
            normalize_name_lower(name): name for name in INJURY_STATUS.keys()
        }

    # 1. Exact match on normalized name
    if normalized_input in _normalized_injury_cache:
        original_name = _normalized_injury_cache[normalized_input]
        return INJURY_STATUS[original_name]

    # 2. Substring match
    for norm_key, orig_name in _normalized_injury_cache.items():
        if normalized_input in norm_key or norm_key in normalized_input:
            return INJURY_STATUS[orig_name]

    # 3. Fuzzy match fallback (still safe because we normalized)
    best_match = None
    best_score = 0
    for norm_key, orig_name in _normalized_injury_cache.items():
        similarity = SequenceMatcher(None, normalized_input, norm_key).ratio()
        if similarity > best_score and similarity > 0.80:
            best_score = similarity
            best_match = orig_name

    if best_match:
        logger.info(f"Fuzzy matched (normalized): '{player_name}' → '{best_match}' ({best_score:.0%})")
        return INJURY_STATUS[best_match]

    # Default: assume active
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
        pace     = float(player_row.get('PACE', 100.0)) if pd.notna(player_row.get('PACE')) else 100.0

        # Advanced metrics
        ts_pct = pts_pg / (2 * (fga_pg + 0.44 * fta_pg)) if (fga_pg + 0.44 * fta_pg) > 0 else 0.55

        if 'USG_PCT' in player_row and pd.notna(player_row['USG_PCT']):
            usg_pct = float(player_row['USG_PCT'])
        else:
            poss = fga_pg + 0.44 * fta_pg + tov_pg
            usg_pct = (poss / min_pg) * 48 * 5 if min_pg > 0 else 30.0
            usg_pct = min(usg_pct, 40.0)

        fgm = fga_pg * fg_pct
        per = (pts_pg + reb_pg + ast_pg + 3 * (stl_pg + blk_pg) - tov_pg - 
               (fga_pg - fgm) - (fta_pg - fta_pg * float(player_row.get('FT_PCT', 0.8))))
        per = max(per, 8.0)

        return {
            'PTS_PG': round(pts_pg, 1), 'MIN_PG': round(min_pg, 1), 
            'USG_PCT': round(usg_pct, 1), 'TS_PCT': round(ts_pct, 3), 
            'FTA_PG': round(fta_pg, 1), 'AST_PG': round(ast_pg, 1),
            'FG3A_PG': round(fg3a_pg, 1), 'PER': round(per, 1), 
            'FG_PCT': round(fg_pct, 3), 'FG3_PCT': round(fg3_pct, 3), 
            'AGE': int(age), 'PACE': round(pace, 1),
            'REB_PG': round(reb_pg, 1), 'STL_PG': round(stl_pg, 1),
            'BLK_PG': round(blk_pg, 1), 'TOV_PG': round(tov_pg, 1),
            'FG3M_PG': round(fg3a_pg * fg3_pct, 1),
        }
    except Exception as e:
        logger.error(f"Feature calculation error: {e}")
        return None

def build_feature_vector(features_dict):
    """Build feature vector for model input"""
    order = ['MIN_PG','USG_PCT','TS_PCT','FTA_PG','AST_PG','FG3A_PG','PER','FG_PCT','FG3_PCT','AGE','PACE']
    values = [float(features_dict[col]) for col in order]
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
            prediction_cache.clear_old()
            logger.info("Cache cleanup completed")
    
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

# ==================== PLAYER PREDICTION ====================
@app.post("/predictX")
def predict_player_mlX(req: PlayerRequest):
    """Predict player performance with smart caching"""
    
    # Check cache first
    cache_key = f"player_{req.player_name}_{req.opponent_abbr}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = prediction_cache.get(cache_key)
    if cached:
        logger.info(f"Cache hit for {req.player_name}")
        return cached
    
    # Check if injury data is stale
    global LAST_INJURY_UPDATE
    if datetime.now() - LAST_INJURY_UPDATE > INJURY_REFRESH_INTERVAL:
        logger.info("Injury data stale, refreshing...")
        refresh_injuries()
    
    # Find player
    matches = df_players[df_players['PLAYER_NAME'].str.lower().str.contains(req.player_name.lower(), na=False)]
    if matches.empty:
        return {
            "error": "Player not found", 
            "available_players": sorted(df_players['PLAYER_NAME'].head(20).tolist())
        }

    p = matches.iloc[0]
    player_name = p['PLAYER_NAME']
    team_abbr = p.get('TEAM_ABBREVIATION', 'UNK')
    
    injury = get_injury_status(player_name)
    
    # Handle injured players
    if injury['status'] in ['OUT', 'Doubtful']:
        result = {
            "player": player_name, 
            "team": team_abbr, 
            "opponent": req.opponent_abbr.upper(),
            "status": injury['status'], 
            "injury_type": injury['injury_type'],
            "recommendation": "AVOID ⛔", 
            "projected_pts": 0.0, 
            "confidence": "N/A",
            "cached": False
        }
        prediction_cache.set(cache_key, result)
        return result

    # Calculate features
    features_dict = calculate_features_from_row(p)
    if not features_dict:
        raise HTTPException(500, "Feature calculation failed")
    
    season_avg = features_dict['PTS_PG']
    
    # Get opponent defense rating
    opp_row = df_teams[df_teams['TEAM_ABBREVIATION'] == req.opponent_abbr.upper()]
    opp_def = float(opp_row['DEF_RATING'].iloc[0]) if not opp_row.empty else 110.0
    
    # Make prediction
    features = build_feature_vector(features_dict)
    pts_raw = float(player_model.predict(features)[0])
    pts_projection = np.clip(pts_raw, season_avg * 0.4, season_avg * 1.6)
    
    # Adjust for matchup
    league_avg_def = 112.0
    if opp_def < league_avg_def - 3:
        pts_projection *= 1.08
        matchup = "Favorable 🎯"
    elif opp_def > league_avg_def + 3:
        pts_projection *= 0.92
        matchup = "Tough 🛡️"
    else:
        matchup = "Neutral ⚖️"
    
    # Adjust for injury status
    confidence = "High"
    if injury['status'] == "Questionable":
        pts_projection *= 0.85
        confidence = "Low"
    elif injury['status'] == "Probable":
        pts_projection *= 0.95
        confidence = "Medium"
    
    pts_projection = round(pts_projection, 1)
    
    # Generate recommendation
    if confidence == "High" and pts_projection > season_avg * 1.05:
        recommendation = f"BET OVER {pts_projection - 0.5:.1f} 💰"
    elif confidence == "High" and pts_projection < season_avg * 0.95:
        recommendation = f"BET UNDER {pts_projection + 0.5:.1f} 💰"
    else:
        recommendation = "MONITOR 👀"
    
    result = {
        "player": player_name, 
        "team": team_abbr, 
        "opponent": req.opponent_abbr.upper(),
        "status": injury['status'], 
        "injury_type": injury.get('injury_type', 'None'),
        "projected_pts": pts_projection, 
        "season_avg_pts": round(season_avg, 1),
        "confidence": confidence, 
        "recommendation": recommendation,
        "matchup": matchup,
        "rebounds_per_game": features_dict.get('REB_PG', 0),
        "assists_per_game": features_dict.get('AST_PG', 0),
        "steals_per_game": features_dict.get('STL_PG', 0),
        "blocks_per_game": features_dict.get('BLK_PG', 0),
        "threes_made_per_game": features_dict.get('FG3M_PG', 0),
        "cached": False
    }
    
    # Cache the result
    prediction_cache.set(cache_key, result)
    
    return result

@app.post("/predict")
def predict_player_ml(req: PlayerRequest):
    """Predict player performance – now works with any name variation"""
    
    raw_name = req.player_name.strip()
    opponent = req.opponent_abbr.upper() or "AVG"

    # ── STEP 1: Canonical name resolution (handles accents, typos, etc.) ──
    canonical_name = get_canonical_name(raw_name)
    normalized_key = normalize_name_lower(canonical_name)

    # Use pre-built lookup for speed
    if normalized_key not in PLAYER_NAME_LOOKUP:
        # Fallback fuzzy search if not exact
        from difflib import get_close_matches
        close = get_close_matches(normalized_key, PLAYER_NAME_LOOKUP.keys(), n=5, cutoff=0.75)
        if close:
            canonical_name = PLAYER_NAME_LOOKUP[close[0]]
            logger.info(f"Name resolved: '{raw_name}' → '{canonical_name}'")
        else:
            return {
                "error": "Player not found",
                "did_you_mean": close[:3] if 'close' in locals() else None,
                "tip": "Try: Luka Doncic, Nikola Jokic, Giannis, etc.",
                "available_players": sorted(df_players['PLAYER_NAME'].head(15).tolist())
            }
    else:
        canonical_name = PLAYER_NAME_LOOKUP[normalized_key]

    # Cache key uses canonical name + hour (so same player = cache hit)
    cache_key = f"player_{canonical_name}_{opponent}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = prediction_cache.get(cache_key)
    if cached:
        logger.info(f"Cache hit for {canonical_name}")
        return cached

    # ── STEP 2: Get player row using official name ──
    player_row = df_players[df_players['PLAYER_NAME'] == canonical_name]
    if player_row.empty:
        raise HTTPException(500, "Player data missing after name resolution")

    p = player_row.iloc[0]
    team_abbr = p.get('TEAM_ABBREVIATION', 'UNK')

    # ── STEP 3: Injury check (also uses normalized lookup) ──
    injury = get_injury_status(canonical_name)  # Your improved function from before

    if injury['status'] in ['OUT', 'Doubtful']:
        result = {
            "player": canonical_name,
            "input_name": raw_name,
            "team": team_abbr,
            "opponent": opponent,
            "status": injury['status'],
            "injury_type": injury['injury_type'],
            "recommendation": "AVOID - Player is OUT/Doubtful",
            "projected_pts": 0.0,
            "confidence": "N/A",
            "cached": False
        }
        prediction_cache.set(cache_key, result)
        return result

    # ── REST OF PREDICTION LOGIC (unchanged) ──
    features_dict = calculate_features_from_row(p)
    if not features_dict:
        raise HTTPException(500, "Failed to calculate player features")

    season_avg = features_dict['PTS_PG']

    # Opponent defense
    opp_row = df_teams[df_teams['TEAM_ABBREVIATION'] == opponent]
    opp_def = float(opp_row['DEF_RATING'].iloc[0]) if not opp_row.empty else 110.0

    # Model prediction
    features = build_feature_vector(features_dict)
    pts_raw = float(player_model.predict(features)[0])
    pts_projection = np.clip(pts_raw, season_avg * 0.4, season_avg * 1.6)

    # Matchup adjustment
    league_avg_def = 112.0
    if opp_def < league_avg_def - 3:
        pts_projection *= 1.08
        matchup = "Favorable"
    elif opp_def > league_avg_def + 3:
        pts_projection *= 0.92
        matchup = "Tough"
    else:
        matchup = "Neutral"

    # Injury probability adjustment
    confidence = "High"
    if injury['status'] == "Questionable":
        pts_projection *= 0.85
        confidence = "Low"
    elif injury['status'] == "Probable":
        pts_projection *= 0.95
        confidence = "Medium"

    pts_projection = round(pts_projection, 1)

    # Recommendation
    if confidence == "High" and pts_projection > season_avg * 1.05:
        recommendation = f"STRONG BUY OVER {pts_projection - 0.5:.1f}"
    elif confidence == "High" and pts_projection < season_avg * 0.95:
        recommendation = f"STRONG BUY UNDER {pts_projection + 0.5:.1f}"
    else:
        recommendation = "HOLD / MONITOR"

    result = {
        "player": canonical_name,
        "input_name": raw_name if raw_name.lower() != canonical_name.lower() else None,
        "team": team_abbr,
        "opponent": opponent,
        "status": injury['status'],
        "injury_type": injury.get('injury_type', 'None'),
        "projected_pts": pts_projection,
        "season_avg_pts": round(season_avg, 1),
        "confidence": confidence,
        "recommendation": recommendation,
        "matchup": matchup,
        "rebounds_per_game": features_dict.get('REB_PG', 0),
        "assists_per_game": features_dict.get('AST_PG', 0),
        "threes_made_per_game": features_dict.get('FG3M_PG', 0),
        "cached": False,
        "model_version": MODEL_VERSION
    }

    prediction_cache.set(cache_key, result)
    return result

# ==================== GAME PREDICTION ====================
@app.post("/predict-game")
def predict_game_ml(req: GamePredictionRequest):
    """Predict game outcome with spread/total"""
    
    cache_key = f"game_{req.home_team}_{req.away_team}_{datetime.now().strftime('%Y%m%d%H')}"
    cached = prediction_cache.get(cache_key, max_age=GAME_CACHE_DURATION)
    if cached:
        logger.info(f"Cache hit for game {req.away_team} @ {req.home_team}")
        return cached
    
    home = df_teams[df_teams['TEAM_ABBREVIATION'] == req.home_team.upper()]
    away = df_teams[df_teams['TEAM_ABBREVIATION'] == req.away_team.upper()]
    
    if home.empty or away.empty:
        raise HTTPException(404, "Team not found")

    h, a = home.iloc[0], away.iloc[0]
    home_off = float(h.get('OFF_RATING', 115)) + 3.5
    home_def = float(h.get('DEF_RATING', 110))
    home_pace = float(h.get('PACE', 100))
    away_off = float(a.get('OFF_RATING', 112))
    away_def = float(a.get('DEF_RATING', 113))
    away_pace = float(a.get('PACE', 99))
    
    features = np.array([[home_off, home_def, home_pace, away_off, away_def, away_pace]], dtype=np.float32)
    total_pred = float(team_model.predict(features)[0])
    
    avg_pace = (home_pace + away_pace) / 2
    home_expected_pts = ((home_off + away_def) / 2) * avg_pace / 100
    away_expected_pts = ((away_off + home_def) / 2) * avg_pace / 100
    spread_pred = home_expected_pts - away_expected_pts
    win_prob_home = 1 / (1 + np.exp(-spread_pred / 3.5))
    
    # Compare to market lines if provided
    spread_edge = None
    total_edge = None
    spread_recommendation = None
    total_recommendation = None
    
    if req.spread_line is not None:
        spread_edge = abs(spread_pred - req.spread_line)
        if spread_edge > 3:
            if spread_pred > req.spread_line:
                spread_recommendation = f"BET {req.home_team} {req.spread_line}"
            else:
                spread_recommendation = f"BET {req.away_team} +{abs(req.spread_line)}"
    
    if req.total_line is not None:
        total_edge = abs(total_pred - req.total_line)
        if total_edge > 5:
            if total_pred > req.total_line:
                total_recommendation = f"BET OVER {req.total_line}"
            else:
                total_recommendation = f"BET UNDER {req.total_line}"
    
    result = {
        "game": f"{req.away_team.upper()} @ {req.home_team.upper()}",
        "projected_spread": round(spread_pred, 1),
        "projected_total": round(total_pred, 1),
        "win_prob_home": round(win_prob_home * 100, 1),
        "spread_edge": round(spread_edge, 1) if spread_edge else None,
        "total_edge": round(total_edge, 1) if total_edge else None,
        "spread_recommendation": spread_recommendation,
        "total_recommendation": total_recommendation,
        "cached": False
    }
    
    prediction_cache.set(cache_key, result)
    return result

# ==================== BEST BETS ====================
@app.get("/best-bets")
@app.post("/best-bets")
def best_bets(
    date: Optional[str] = None,
    days_offset: Optional[int] = None,
    min_edge: float = 3.0,
    max_bets: int = 10,
):
    """Get best betting opportunities for a date"""
    
    if not date:
        offset = days_offset or 0
        target = (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
    else:
        target = date

    # Check cache
    cache_file = CACHE_DIR / f"best_bets_{target}.json"
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
            
            if pred.get('spread_edge', 0) >= min_edge and pred.get('spread_recommendation'):
                bets.append({
                    "game": pred['game'],
                    "bet_type": "SPREAD",
                    "recommendation": pred['spread_recommendation'],
                    "edge": pred['spread_edge'],
                    "our_projection": pred['projected_spread'],
                    "market_line": spread,
                    "confidence": "High" if pred['spread_edge'] > 5 else "Medium",
                    "game_time": game.get('game_time', 'TBD'),
                    "win_prob": pred['win_prob_home']
                })
            
            if pred.get('total_edge', 0) >= min_edge and pred.get('total_recommendation'):
                bets.append({
                    "game": pred['game'],
                    "bet_type": "TOTAL",
                    "recommendation": pred['total_recommendation'],
                    "edge": pred['total_edge'],
                    "our_projection": pred['projected_total'],
                    "market_line": total,
                    "confidence": "High" if pred['total_edge'] > 7 else "Medium",
                    "game_time": game.get('game_time', 'TBD')
                })
                
        except Exception as e:
            logger.error(f"Error processing game: {e}")
            continue

    bets.sort(key=lambda x: x.get("edge", 0), reverse=True)
    
    result = {
        "date": target,
        "games_analyzed": len(games),
        "total_opportunities": len(bets),
        "best_bets": bets[:max_bets],
        "min_edge_threshold": min_edge,
        "model_version": f"XGBoost {MODEL_VERSION}",
        "confidence_levels": {
            "High": len([b for b in bets if b.get('confidence') == 'High']),
            "Medium": len([b for b in bets if b.get('confidence') == 'Medium'])
        }
    }
    
    # Cache the result
    with open(cache_file, 'w') as f:
        json.dump(result, f)
    
    return result

# ==================== PARLAY BUILDER ====================
@app.post("/build-parlay")
async def build_parlay(legs: List[ParlayLeg]):
    """Build multi-leg parlay with odds calculation"""
    
    if not legs:
        raise HTTPException(400, "No legs provided")

    total_decimal = 1.0
    details = []

    for leg in legs:
        if leg.player:
            req = PlayerRequest(player_name=leg.player, opponent_abbr="AVG")
            pred = predict_player_ml(req)
            
            base = {
                "points": pred.get("projected_pts", 20.0),
                "rebounds": pred.get("rebounds_per_game", 6.0),
                "assists": pred.get("assists_per_game", 5.0),
            }
            
            projection = base.get(leg.stat, base["points"])
            edge = (projection - leg.line) if leg.over else (leg.line - projection)
            prob = np.clip(0.5 + edge * 0.04, 0.15, 0.85)
            leg_odds = round(max(1.0 / prob, 1.05), 2)
            
            details.append({
                "bet": f"{leg.player} {'OVER' if leg.over else 'UNDER'} {leg.line} {leg.stat}",
                "projection": round(projection, 1),
                "decimal_odds": leg_odds,
                "status": pred.get("status", "Active"),
                "injury_type": pred.get("injury_type", "None")
            })
            total_decimal *= leg_odds

    return {
        "parlay_odds": round(total_decimal, 2),
        "possible_win_per_100php": round((total_decimal - 1) * 100, 2),
        "legs": details,
        "risk_level": "High" if total_decimal > 10 else "Medium" if total_decimal > 5 else "Low"
    }

# ==================== DATA MANAGEMENT ENDPOINTS ====================
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
    """Get current injury reports with optional filtering"""
    
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
        "cache_hit_rate": "N/A",  # Could track this with counters
        "last_cleanup": "Auto every 30 min",
        "injury_update": LAST_INJURY_UPDATE.isoformat()
    }

@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": MODEL_VERSION,
        "players_loaded": len(df_players),
        "teams_loaded": len(df_teams),
        "injuries_tracked": len(INJURY_STATUS),
        "last_injury_update": LAST_INJURY_UPDATE.isoformat(),
        "cache_size": len(prediction_cache._cache)
    }

@app.get("/players")
def list_players(
    limit: int = 50,
    search: Optional[str] = None
):
    """Search players with full accent/diacritic support + fuzzy tolerance"""
    if not search:
        players = df_players['PLAYER_NAME'].head(limit).tolist()
        return {
            "total": len(df_players),
            "players": sorted(players),
            "hint": "Use ?search= to find players (supports accents & typos)"
        }

    query = normalize_name_lower(search.strip())

    # Fast exact or substring match using pre-normalized dict
    matches = []
    for norm_key, official_name in PLAYER_NAME_LOOKUP.items():
        if query in norm_key or norm_key in query:
            matches.append(official_name)
        elif len(matches) >= limit * 3:  # early exit
            break

    # Fallback: fuzzy if no strong matches
    if len(matches) < 3:
        from difflib import get_close_matches
        all_norm = list(PLAYER_NAME_LOOKUP.keys())
        close = get_close_matches(query, all_norm, n=limit, cutoff=0.7)
        matches.extend(PLAYER_NAME_LOOKUP.get(c, c) for c in close)

    matches = sorted(set(matches))[:limit]

    return {
        "query": search,
        "normalized_query": query,
        "total_found": len(matches),
        "players": matches,
        "examples": [
            "Luka Doncic", "Luka Dončić", "Nikola Jokic", "Nikola Jokić",
            "Jose Alvarado", "José Alvarado", "Giannis"
        ]
    }

@app.get("/teams")
def list_teams():
    """List all NBA teams"""
    teams_list = df_teams[['TEAM_ABBREVIATION', 'OFF_RATING', 'DEF_RATING', 'PACE']].to_dict('records')
    return {
        "total": len(teams_list),
        "teams": teams_list
    }

@app.get("/")
def root():
    """API root endpoint"""
    return {
        "service": f"NBA Betting Agent Pro — {MODEL_VERSION}",
        "version": "4.0.0",
        "status": "running",
        "endpoints": {
            "predict": "POST /predict",
            "predict_game": "POST /predict-game",
            "best_bets": "GET /best-bets",
            "build_parlay": "POST /build-parlay",
            "injuries": "GET /injuries",
            "health": "GET /health",
            "players": "GET /players",
            "teams": "GET /teams"
        },
        "injury_tracking": {
            "total": len(INJURY_STATUS),
            "last_update": LAST_INJURY_UPDATE.isoformat(),
            "auto_refresh": "Every 1 hour"
        },
        "cache": {
            "size": len(prediction_cache._cache),
            "prediction_ttl_minutes": 30,
            "game_ttl_hours": 6
        },
        "documentation": "/docs"
    }