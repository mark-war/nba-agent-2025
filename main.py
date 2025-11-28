# main.py — NBA Betting Agent Pro 2025-26 — FULL ML POWERED v3
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ==================== LOAD ML MODELS & DATA ====================
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

print("Loading ML models and data...")

# Real trained models
player_model = joblib.load(MODELS_DIR / "player_model_2025.pkl")
team_model = joblib.load(MODELS_DIR / "team_model_2025.pkl")

# Data
df_players = pd.read_csv(DATA_DIR / "2025_26_players.csv")
df_teams = pd.read_csv(DATA_DIR / "2025_26_teams.csv")
injuries_df = pd.read_csv(DATA_DIR / "injuries.csv")

print(f"ML Models loaded | Players: {len(df_players)} | Teams: {len(df_teams)} | Injuries: {len(injuries_df)}")

# Build injury lookup
INJURY_STATUS = {}
for _, row in injuries_df.iterrows():
    name = str(row['player_name']).strip().title()
    if name and name != "Nan":
        INJURY_STATUS[name] = {
            "status": str(row['status']).strip(),
            "injury_type": str(row.get('injury_type', 'Unknown')).strip(),
            "team": str(row.get('team', 'UNK'))
        }

def get_injury_status(player_name: str) -> Dict:
    key = player_name.strip().title()
    if key in INJURY_STATUS:
        return INJURY_STATUS[key]
    for k in INJURY_STATUS:
        if key in k or k in key:
            return INJURY_STATUS[k]
    return {"status": "Active", "injury_type": "None", "team": ""}

# ==================== FASTAPI APP ====================
app = FastAPI(
    title="NBA Betting Agent Pro 2025-26 — ML EDITION",
    description="True machine learning predictions • Player props • Game outcomes • Best bets",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== REQUEST MODELS ====================
class PlayerRequest(BaseModel):
    player_name: str
    opponent_abbr: str = "AVG"
    pts_line: Optional[float] = None
    reb_line: Optional[float] = None
    ast_line: Optional[float] = None
    stl_line: Optional[float] = None
    blk_line: Optional[float] = None
    three_pm_line: Optional[float] = None

class GamePredictionRequest(BaseModel):
    home_team: str
    away_team: str
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    game_time: Optional[str] = None

class BestBetsRequest(BaseModel):
    date: Optional[str] = None
    min_edge: float = 8.0
    max_bets: int = 10

# ==================== ML PLAYER PROP PREDICTION ====================
@app.post("/predict")
def predict_player_ml(req: PlayerRequest):
    search = req.player_name.strip().lower()
    matches = df_players[df_players['PLAYER_NAME'].str.lower().str.contains(search, na=False)]
    
    if matches.empty:
        return {"error": "Player not found", "player": req.player_name}

    p = matches.iloc[0]
    player_name = p['PLAYER_NAME']
    injury = get_injury_status(player_name)

    if injury['status'] in ['OUT', 'Doubtful']:
        return {
            "player": player_name,
            "status": injury['status'],
            "recommendation": "AVOID ALL PROPS",
            "reason": "Player injured or unlikely to play"
        }

    # Opponent defense lookup
    opp_row = df_teams[df_teams['TEAM_ABBREVIATION'] == req.opponent_abbr.upper()]
    opp_def = opp_row['DEF_RATING'].iloc[0] if not opp_row.empty else 114.0
    opp_pace = opp_row['PACE'].iloc[0] if not opp_row.empty else 100.0

    # Build feature vector (must match training!)
    features = np.array([[
        p.get('MP', 30.0),
        p.get('FG_PCT', 0.45),
        p.get('FG3_PCT', 0.35),
        p.get('FT_PCT', 0.80),
        p.get('OREB', 1.0),
        p.get('DREB', 4.0),
        p.get('AST', 5.0),
        p.get('STL', 1.0),
        p.get('BLK', 0.5),
        p.get('TOV', 2.0),
        p.get('PF', 2.0),
        p.get('PLUS_MINUS', 0),
        opp_def,
        opp_pace,
        2.0,  # rest days
        0,    # not B2B
        1 if p.get('TEAM_ABBREVIATION') != req.opponent_abbr.upper() else 0  # away?
    ]], dtype=np.float32)

    # REAL ML PREDICTION
    raw_preds = player_model.predict(features)[0]
    
    projected = {
        "PTS": round(float(raw_preds[0]), 1),
        "REB": round(float(raw_preds[1]), 1),
        "AST": round(float(raw_preds[2]), 1),
        "STL": round(float(raw_preds[3]), 1),
        "BLK": round(float(raw_preds[4]), 1),
        "FG3M": round(float(raw_preds[5]), 1),
    }

    # Injury adjustment
    if injury['status'] == "Questionable":
        projected = {k: round(v * 0.9, 1) for k, v in projected.items()}

    # Edge calculation
    lines = {
        "PTS": req.pts_line or round(p.get('PTS', 20.0) - 0.5, 1),
        "REB": req.reb_line or round(p.get('REB', 5.0) - 0.3, 1),
        "AST": req.ast_line or round(p.get('AST', 4.0) - 0.4, 1),
        "STL": req.stl_line or 1.0,
        "BLK": req.blk_line or 0.8,
        "FG3M": req.three_pm_line or 1.5,
    }

    bets = {}
    for stat in projected:
        proj = projected[stat]
        line = lines[stat]
        edge = (proj - line) / line * 100 if line > 0 else 0
        bets[stat] = {
            "line": line,
            "projected": proj,
            "edge_percent": round(abs(edge), 1),
            "best_bet": "OVER" if proj > line else "UNDER",
            "confidence": "HIGH" if abs(edge) > 10 else "MEDIUM"
        }

    best_stat = max(bets.items(), key=lambda x: x[1]["edge_percent"])

    return {
        "player": player_name,
        "team": p.get('TEAM_ABBREVIATION', 'UNK'),
        "vs": req.opponent_abbr.upper(),
        "status": injury['status'],
        "injury_type": injury.get('injury_type', 'None'),
        "model": "ML v2 (XGBoost/LGBM)",
        "predictions": bets,
        "top_bet": {"stat": best_stat[0], **best_stat[1]},
        "timestamp": datetime.now().isoformat()
    }

# ==================== ML GAME PREDICTION ====================
@app.post("/predict-game")
def predict_game_ml(req: GamePredictionRequest):
    home = df_teams[df_teams['TEAM_ABBREVIATION'] == req.home_team.upper()]
    away = df_teams[df_teams['TEAM_ABBREVIATION'] == req.away_team.upper()]

    if home.empty or away.empty:
        raise HTTPException(404, f"Team not found")

    h, a = home.iloc[0], away.iloc[0]

    features = np.array([[
        h.get('OFF_RATING', 115), h.get('DEF_RATING', 110), h.get('PACE', 100), h.get('PIE', 0.5),
        a.get('OFF_RATING', 112), a.get('DEF_RATING', 113), a.get('PACE', 99), a.get('PIE', 0.48),
        1,  # home court
        0,  # regular season
    ]], dtype=np.float32)

    pred = team_model.predict(features)[0]

    spread_proj = float(pred[0])  # home margin
    total_proj = float(pred[1])
    win_prob_home = float(pred[2])

    return {
        "game": f"{req.away_team.upper()} @ {req.home_team.upper()}",
        "game_time": req.game_time or "TBD",
        "projected_spread": f"{req.home_team.upper()} {spread_proj:+.1f}",
        "projected_total": round(total_proj, 1),
        "win_probability_home": round(win_prob_home * 100, 1),
        "model": "ML Game Model v2",
        "recommendations": [
            {"bet_type": "SPREAD", "pick": f"{req.home_team} {spread_proj:+.1f}", "edge": "ML"},
            {"bet_type": "TOTAL", "pick": "OVER" if total_proj > (req.total_line or 220) else "UNDER", "edge": "ML"}
        ],
        "timestamp": datetime.now().isoformat()
    }

# ==================== REST OF YOUR ENDPOINTS (unchanged) ====================
# Keep your /todays-games, /best-bets (now uses ML predict-game), /health, etc.
# Just make sure /best-bets calls the new predict_game_ml()

# Example quick /best-bets update (keep your full version, just update the call)
# In get_best_bets_common(), replace predict_game(...) with:
#     pred = predict_game_ml(GamePredictionRequest(...))

# ==================== UTILITY ENDPOINTS ====================
@app.get("/")
def home():
    return {
        "service": "NBA Betting Agent Pro — ML EDITION",
        "status": "LIVE WITH MACHINE LEARNING",
        "models": ["player_model_2025.pkl", "team_model_2025.pkl"],
        "endpoints": ["/predict", "/predict-game", "/best-bets", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ml_models": "loaded",
        "timestamp": datetime.now().isoformat(),
        "players": len(df_players),
        "teams": len(df_teams)
    }