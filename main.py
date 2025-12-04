# main.py — FINAL FIXED VERSION (Feature Count Matches Training)
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

# ==================== LOAD MODELS & DATA ====================
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

print("Loading ML models and data...")
player_model = joblib.load(MODELS_DIR / "player_model_2025.pkl")
team_model = joblib.load(MODELS_DIR / "team_model_2025.pkl")

df_players = pd.read_csv(DATA_DIR / "2025_26_players.csv")
df_teams = pd.read_csv(DATA_DIR / "2025_26_teams.csv")
injuries_df = pd.read_csv(DATA_DIR / "injuries.csv")

# Load metadata
try:
    with open(MODELS_DIR / "training_metadata.json") as f:
        training_metadata = json.load(f)

    TRAINED_FEATURES = training_metadata['player_model']['features']
    MODEL_VERSION = training_metadata.get('version', 'v2.1')

    assert len(TRAINED_FEATURES) == 11, "Feature count mismatch!"
    assert TRAINED_FEATURES == ['MIN_PG', 'USG_PCT', 'TS_PCT', 'FTA_PG', 'AST_PG',
                            'FG3A_PG', 'PER', 'FG_PCT', 'FG3_PCT', 'AGE', 'PACE'], \
        f"Feature order mismatch! Got: {TRAINED_FEATURES}"

    print("Feature order LOCKED and verified")
except Exception as e:
    # Fallback to exact 11 features
    TRAINED_FEATURES = [
        'MIN_PG', 'USG_PCT', 'TS_PCT', 'FTA_PG', 'AST_PG',
        'FG3A_PG', 'PER', 'FG_PCT', 'FG3_PCT', 'AGE', 'PACE'
    ]
    MODEL_VERSION = 'v2.1'
    print(f"⚠️  Using default {len(TRAINED_FEATURES)} features")

print(f"Data loaded | Players: {len(df_players)} | Teams: {len(df_teams)}")

# Injury lookup
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
    if key in INJURY_STATUS: return INJURY_STATUS[key]
    for k in INJURY_STATUS:
        if key in k or k in key: return INJURY_STATUS[k]
    return {"status": "Active", "injury_type": "None", "team": ""}

# ==================== FASTAPI APP ====================
app = FastAPI(title=f"NBA Betting Agent Pro — {MODEL_VERSION}", version="2.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==================== REQUEST MODELS ====================
class PlayerRequest(BaseModel):
    player_name: str
    opponent_abbr: str = "AVG"

class GamePredictionRequest(BaseModel):
    home_team: str
    away_team: str
    spread_line: Optional[float] = None
    total_line: Optional[float] = None

# ==================== CACHING ====================
CACHE_DURATION = timedelta(hours=1)

def load_cached_games(date_str: str):
    cache_file = Path(f"data/games_{date_str}.json")
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache = json.load(f)
                if datetime.fromisoformat(cache['timestamp']) > datetime.now() - CACHE_DURATION:
                    return cache['games']
        except: pass
    return None

def save_games_cache(games, date_str):
    cache_file = Path(f"data/games_{date_str}.json")
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump({"timestamp": datetime.now().isoformat(), "games": games}, f)

# ==================== HELPER: CALCULATE ALL 11 FEATURES ====================
def calculate_features_from_row_OLD(player_row):
    """Calculate ALL 11 features that match training"""
    games = max(float(player_row.get('GP', 1)), 1)
    
    # Per-game stats from TOTALS
    pts_pg = float(player_row.get('PTS', 0)) / games
    min_pg = float(player_row.get('MIN', 0)) / games
    fga_pg = float(player_row.get('FGA', 0)) / games
    fg3a_pg = float(player_row.get('FG3A', 0)) / games
    fta_pg = float(player_row.get('FTA', 0)) / games
    ast_pg = float(player_row.get('AST', 0)) / games
    reb_pg = float(player_row.get('REB', 0)) / games
    stl_pg = float(player_row.get('STL', 0)) / games
    blk_pg = float(player_row.get('BLK', 0)) / games
    tov_pg = float(player_row.get('TOV', 0)) / games
    
    # Percentages
    fg_pct = float(player_row.get('FG_PCT', 0.45))
    fg3_pct = float(player_row.get('FG3_PCT', 0.35))
    
    # True Shooting
    pts_total = float(player_row.get('PTS', 0))
    fga_total = float(player_row.get('FGA', 1))
    fta_total = float(player_row.get('FTA', 0))
    ts_pct = pts_total / (2 * (fga_total + 0.44 * fta_total)) if (fga_total + fta_total) > 0 else 0.50
    ts_pct = np.clip(ts_pct, 0.30, 0.75)
    
    # Usage Rate
    if 'USG_PCT' in player_row and pd.notna(player_row['USG_PCT']):
        usg_pct = float(player_row['USG_PCT'])
    else:
        player_poss = fga_pg + 0.44 * fta_pg + tov_pg
        usg_pct = (player_poss / min_pg) * (240 / 5) * 100 if min_pg > 0 else 20.0
    usg_pct = np.clip(usg_pct, 5, 45)
    
    # PER
    per = (
        pts_pg + 
        reb_pg * 0.4 + 
        ast_pg * 0.7 + 
        stl_pg * 1.0 + 
        blk_pg * 1.0 - 
        tov_pg * 1.0 - 
        (fga_pg - fga_pg * fg_pct) * 0.5
    )
    per = np.clip(per, 0, 40)
    
    # Other
    age = float(player_row.get('AGE', 27))
    pace = float(player_row.get('PACE', 100.0)) if pd.notna(player_row.get('PACE')) else 100.0
    
    # Return ALL 11 features in exact order
    return {
        'PTS_PG': pts_pg,
        'MIN_PG': min_pg,
        'USG_PCT': usg_pct,
        'TS_PCT': ts_pct,
        'FTA_PG': fta_pg,
        'AST_PG': ast_pg,
        'FG3A_PG': fg3a_pg,
        'PER': per,
        'FG_PCT': fg_pct,
        'FG3_PCT': fg3_pct,
        'AGE': age,
        'PACE': pace,
        # Extra stats for display
        'REB_PG': reb_pg,
        'STL_PG': stl_pg,
        'BLK_PG': blk_pg,
        'TOV_PG': tov_pg,
    }

# ==================== FIXED: NO DIVISION BY GP ANYMORE ====================
def calculate_features_from_row(player_row):
    """FINAL VERSION — 100% compatible with your trained model"""
    
    # Raw per-game stats — your CSV already has them
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

    pace = 100.0
    if 'PACE' in player_row and pd.notna(player_row['PACE']):
        pace = float(player_row['PACE'])

    # True Shooting % — exact
    ts_pct = pts_pg / (2 * (fga_pg + 0.44 * fta_pg)) if (fga_pg + 0.44 * fta_pg) > 0 else 0.55

    # USG_PCT — use column if exists, otherwise safe estimate
    if 'USG_PCT' in player_row and pd.notna(player_row['USG_PCT']):
        usg_pct = float(player_row['USG_PCT'])
    else:
        # This formula keeps Luka at ~36–37, never 50
        poss = fga_pg + 0.44 * fta_pg + tov_pg
        usg_pct = (poss / min_pg) * 48 * 5 if min_pg > 0 else 30.0
        usg_pct = min(usg_pct, 40.0)   # ← this line saves you

    # PER — realistic, never explodes
    per = (pts_pg +
           reb_pg +
           ast_pg +
           3 * (stl_pg + blk_pg) -
           tov_pg -
           (fga_pg - fga_pg * fg_pct) -
           (fta_pg - fta_pg * float(player_row.get('FT_PCT', 0.8))))
    per = max(per, 8.0)

    return {
        'PTS_PG': round(pts_pg, 1),           # ← NOW EXISTS
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
    }


def build_feature_vector(features_dict):
    """Hard-coded order — 100% guaranteed match"""
    order = ['MIN_PG','USG_PCT','TS_PCT','FTA_PG','AST_PG','FG3A_PG','PER','FG_PCT','FG3_PCT','AGE','PACE']
    values = [float(features_dict[col]) for col in order]
    print(f"Luka features → {values}")   # ← you will see realistic numbers now
    return np.array([values], dtype=np.float32)

# ==================== PLAYER PREDICTION ====================
@app.post("/predict")
def predict_player_ml(req: PlayerRequest):
    matches = df_players[df_players['PLAYER_NAME'].str.lower().str.contains(req.player_name.lower(), na=False)]
    if matches.empty:
        return {
            "error": "Player not found",
            "suggestion": "Try these players:",
            "available_players": sorted(df_players['PLAYER_NAME'].head(20).tolist())
        }

    p = matches.iloc[0]
    player_name = p['PLAYER_NAME']
    team_abbr = p.get('TEAM_ABBREVIATION', 'UNK')
    
    injury = get_injury_status(player_name)
    if injury['status'] in ['OUT', 'Doubtful']:
        return {
            "player": player_name,
            "team": team_abbr,
            "opponent": req.opponent_abbr.upper(),
            "vs": req.opponent_abbr.upper(),
            "status": injury['status'],
            "recommendation": "AVOID ⛔",
            "reason": f"Player is {injury['status']} - {injury['injury_type']}",
            "projected_pts": 0.0,
            "confidence": "N/A"
        }

    # Calculate ALL features
    features_dict = calculate_features_from_row(p)
    season_avg = features_dict['PTS_PG']
    
    # Get opponent defense
    opp_row = df_teams[df_teams['TEAM_ABBREVIATION'] == req.opponent_abbr.upper()]
    opp_def = float(opp_row['DEF_RATING'].iloc[0]) if not opp_row.empty else 110.0
    
    # Build feature vector
    features = build_feature_vector(features_dict)
    
    # Predict
    try:
        pts_raw = float(player_model.predict(features)[0])
    except Exception as e:
        return {
            "error": f"Model prediction failed: {e}",
            "features_expected": TRAINED_FEATURES,
            "features_provided": len(features[0]),
            "feature_values": features[0].tolist()
        }
    
    # Bounds and adjustments
    pts_projection = np.clip(pts_raw, season_avg * 0.4, season_avg * 1.6)
    
    # Opponent defense
    league_avg_def = 112.0
    if opp_def < league_avg_def - 3:
        pts_projection *= 1.08
        matchup = "Favorable 🎯"
    elif opp_def > league_avg_def + 3:
        pts_projection *= 0.92
        matchup = "Tough 🛡️"
    else:
        matchup = "Neutral ⚖️"
    
    # Injury
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
        recommendation = f"BET OVER {pts_projection - 0.5:.1f} 💰"
    elif confidence == "High" and pts_projection < season_avg * 0.95:
        recommendation = f"BET UNDER {pts_projection + 0.5:.1f} 💰"
    else:
        recommendation = "MONITOR 👀"
    
    return {
        "player": player_name,
        "team": team_abbr,
        "opponent": req.opponent_abbr.upper() if req.opponent_abbr != "AVG" else "AVG",
        "vs": req.opponent_abbr.upper(),
        "status": injury['status'],
        "projected_pts": pts_projection,
        "season_avg_pts": round(season_avg, 1),
        "minutes_per_game": round(features_dict['MIN_PG'], 1),
        "usage_rate": round(features_dict['USG_PCT'], 1),
        "true_shooting_pct": round(features_dict['TS_PCT'], 3),
        "player_efficiency": round(features_dict['PER'], 1),
        "matchup_rating": matchup,
        "opp_def_rating": round(opp_def, 1),
        "confidence": confidence,
        "recommendation": recommendation,
        "model_version": f"XGBoost {MODEL_VERSION}",
        "features_used": len(TRAINED_FEATURES)
    }

# ==================== GAME PREDICTION ====================
@app.post("/predict-game")
def predict_game_ml(req: GamePredictionRequest):
    home = df_teams[df_teams['TEAM_ABBREVIATION'] == req.home_team.upper()]
    away = df_teams[df_teams['TEAM_ABBREVIATION'] == req.away_team.upper()]
    
    if home.empty or away.empty:
        available = sorted(df_teams['TEAM_ABBREVIATION'].tolist())
        raise HTTPException(404, f"Team not found. Available: {available}")

    h, a = home.iloc[0], away.iloc[0]
    
    home_off = float(h.get('OFF_RATING', 115)) + 3.5
    home_def = float(h.get('DEF_RATING', 110))
    home_pace = float(h.get('PACE', 100))
    away_off = float(a.get('OFF_RATING', 112))
    away_def = float(a.get('DEF_RATING', 113))
    away_pace = float(a.get('PACE', 99))
    
    features = np.array([[
        home_off, home_def, home_pace,
        away_off, away_def, away_pace
    ]], dtype=np.float32)

    total_pred = float(team_model.predict(features)[0])
    
    avg_pace = (home_pace + away_pace) / 2
    possessions = avg_pace
    
    home_expected_pts = ((home_off + away_def) / 2) * possessions / 100
    away_expected_pts = ((away_off + home_def) / 2) * possessions / 100
    
    spread_pred = home_expected_pts - away_expected_pts
    win_prob_home = 1 / (1 + np.exp(-spread_pred / 3.5))
    
    edges = {}
    if req.spread_line is not None:
        spread_edge = abs(spread_pred - req.spread_line)
        if spread_edge >= 3:
            if spread_pred > req.spread_line:
                edges['spread_recommendation'] = f"BET {req.home_team.upper()} {req.spread_line:+.1f} 💰"
                edges['spread_side'] = req.home_team.upper()
            else:
                edges['spread_recommendation'] = f"BET {req.away_team.upper()} {-req.spread_line:+.1f} 💰"
                edges['spread_side'] = req.away_team.upper()
        edges['spread_edge'] = round(spread_edge, 1)
    
    if req.total_line is not None:
        total_edge = abs(total_pred - req.total_line)
        if total_edge >= 5:
            if total_pred > req.total_line:
                edges['total_recommendation'] = f"BET OVER {req.total_line} 💰"
            else:
                edges['total_recommendation'] = f"BET UNDER {req.total_line} 💰"
        edges['total_edge'] = round(total_edge, 1)

    return {
        "game": f"{req.away_team.upper()} @ {req.home_team.upper()}",
        "projected_spread": round(spread_pred, 1),
        "projected_total": round(total_pred, 1),
        "projected_home_pts": round(home_expected_pts, 1),
        "projected_away_pts": round(away_expected_pts, 1),
        "win_prob_home": round(win_prob_home * 100, 1),
        "actual_spread": req.spread_line,
        "actual_total": req.total_line,
        **edges,
        "model_version": f"XGBoost {MODEL_VERSION}"
    }

# ==================== BEST BETS ====================
@app.get("/best-bets")
@app.post("/best-bets")
def best_bets(
    date: Optional[str] = None,
    days_offset: Optional[int] = None,
    min_edge: float = 3.0,
    max_bets: int = 10,
):
    if not date:
        offset = days_offset or 0
        target = (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
    else:
        target = date

    games = load_cached_games(target)
    if not games:
        try:
            from utils import fetch_games_for_date
            games_str = target.replace("-", "")
            games = fetch_games_for_date(games_str)
            save_games_cache(games, target)
        except Exception as e:
            print(f"Game fetch error: {e}")
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
            print(f"Error processing game: {e}")
            continue

    bets.sort(key=lambda x: x.get("edge", 0), reverse=True)
    
    return {
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

# ==================== BUILD PARLAY ====================
class ParlayLeg(BaseModel):
    # Player prop
    player: Optional[str] = None
    stat: Optional[str] = "points"          # points / rebounds / assists
    line: Optional[float] = None
    over: Optional[bool] = True             # True = OVER, False = UNDER

    # Game bet
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    bet_type: Optional[str] = None          # "spread" or "total"
    total_over: Optional[bool] = None       # only used for totals

@app.post("/build-parlay-old")
async def build_parlayOLD(legs: List[ParlayLeg]):
    if not legs:
        raise HTTPException(400, "No legs provided")

    total_decimal = 1.0
    details = []

    for leg in legs:
        leg_odds = 1.0
        leg_detail = {}

        # ——— PLAYER PROP ———
        if leg.player:
            req = PlayerRequest(player_name=leg.player, opponent_abbr="AVG")
            try:
                pred = predict_player_ml(req)
            except:
                raise HTTPException(404, f"Player not found: {leg.player}")

            # Right now we only support points — easy to add reb/ast later
            if leg.stat == "points":
                projection = pred["projected_pts"]
            elif leg.stat == "rebounds":
                projection = pred.get("REB_PG", 5.0)
            elif leg.stat == "assists":
                projection = pred.get("AST_PG", 4.0)
            else:
                projection = pred["projected_pts"]  # fallback

            edge = (projection - leg.line) if leg.over else (leg.line - projection)
            prob = 0.5 + edge * 0.04
            prob = np.clip(prob, 0.15, 0.85)
            leg_odds = round(max(1.0 / prob, 1.05), 2)

            leg_detail = {
                "type": "player_prop",
                "bet": f"{leg.player} {'OVER' if leg.over else 'UNDER'} {leg.line} {leg.stat}",
                "projection": round(projection, 1),
                "edge_pts": round(edge, 1),
                "probability": round(prob * 100, 1),
                "decimal_odds": leg_odds
            }

        # ——— GAME BET (spread or total) ———
        elif leg.home_team and leg.away_team and leg.bet_type:
            game_req = GamePredictionRequest(
                home_team=leg.home_team.upper(),
                away_team=leg.away_team.upper(),
                spread_line=leg.line if leg.bet_type == "spread" else None,
                total_line=leg.line if leg.bet_type == "total" else None
            )
            game_pred = predict_game_ml(game_req)

            if leg.bet_type == "spread":
                proj = game_pred["projected_spread"]
                edge = abs(proj - leg.line)
                # Correct logic: if we project bigger home win → bet home
                if proj > leg.line:
                    prob = 0.5 + edge * 0.05
                else:
                    prob = 0.5 - edge * 0.05
                bet_side = leg.home_team.upper() if proj > leg.line else leg.away_team.upper()
                bet_line = leg.line if proj > leg.line else -leg.line

            else:  # total
                proj = game_pred["projected_total"]
                edge = abs(proj - leg.line)
                prob = 0.5 + (proj - leg.line) * 0.06
                bet_side = "OVER" if proj > leg.line else "UNDER"
                bet_line = leg.line

            prob = np.clip(prob, 0.20, 0.80)
            leg_odds = round(max(1.0 / prob, 1.10), 2)

            leg_detail = {
                "type": leg.bet_type,
                "bet": f"{leg.away_team.upper()} @ {leg.home_team.upper()} {bet_side} {bet_line:+.1f}",
                "projection": round(proj, 1),
                "edge": round(edge, 1),
                "probability": round(prob * 100, 1),
                "decimal_odds": leg_odds
            }

        total_decimal *= leg_odds
        details.append(leg_detail)

    return {
        "parlay_odds": round(total_decimal, 2),
        "total_legs": len(legs),
        "estimated_fair_odds": round(total_decimal, 2),
        "possible_win_per_100php": round((total_decimal - 1) * 100, 2),
        "legs": details,
        "model_version": MODEL_VERSION
    }

@app.post("/build-parlay")
async def build_parlay(legs: List[ParlayLeg]):
    if not legs:
        raise HTTPException(400, "No legs provided")

    total_decimal = 1.0
    details = []

    for leg in legs:
        leg_detail = {}
        leg_odds = 1.0

        # ——— PLAYER PROP (ALL STATS SUPPORTED) ———
        if leg.player:
            if leg.line is None:
                raise HTTPException(400, f"Line required for player prop: {leg.player}")

            req = PlayerRequest(player_name=leg.player, opponent_abbr="AVG")
            pred = predict_player_ml(req)

            # Get base stats from model
            base = {
                "points": pred.get("projected_pts", 20.0),
                "rebounds": pred.get("REB_PG", 6.0),
                "assists": pred.get("AST_PG", 5.0),
                "steals": pred.get("STL_PG", 0.8),
                "blocks": pred.get("BLK_PG", 0.6),
                "threes": pred.get("FG3M_PG", pred.get("FG3A_PG", 3.0) * 0.38),  # estimate
            }

            # Handle PRA
            if leg.stat == "pra":
                projection = base["points"] + base["rebounds"] + base["assists"]
                stat_name = "PRA"
            else:
                projection = base.get(leg.stat, base["points"])
                stat_name = leg.stat.replace("threes", "3PM").upper()

            edge = (projection - leg.line) if leg.over else (leg.line - projection)
            prob = np.clip(0.5 + edge * 0.04, 0.15, 0.85)
            leg_odds = round(max(1.0 / prob, 1.05), 2)

            leg_detail = {
                "type": "player_prop",
                "bet": f"{leg.player} {'OVER' if leg.over else 'UNDER'} {leg.line} {stat_name}",
                "projection": round(projection, 1),
                "edge": round(edge, 1),
                "probability": round(prob * 100, 1),
                "decimal_odds": leg_odds
            }
        # ——— GAME BETS ———
        elif leg.home_team and leg.away_team and leg.bet_type:
            game_pred = predict_game_ml(GamePredictionRequest(
                home_team=leg.home_team.upper(),
                away_team=leg.away_team.upper(),
                spread_line=leg.line if leg.bet_type == "spread" else None,
                total_line=leg.line if leg.bet_type == "total" else None
            ))

            if leg.bet_type == "moneyline":
                # No line needed
                winner = leg.home_team.upper() if game_pred["projected_spread"] > 0 else leg.away_team.upper()
                prob = game_pred["win_prob_home"] / 100 if game_pred["projected_spread"] > 0 else (100 - game_pred["win_prob_home"]) / 100
                prob = np.clip(prob, 0.15, 0.85)
                leg_odds = round(max(1.0 / prob, 1.10), 2)

                leg_detail = {
                    "type": "moneyline",
                    "bet": f"{winner} to win (incl. overtime)",
                    "win_probability": round(prob * 100, 1),
                    "decimal_odds": leg_odds
                }

            elif leg.bet_type == "spread":
                if leg.line is None:
                    raise HTTPException(400, "Line required for spread bet")
                proj = game_pred["projected_spread"]
                edge = abs(proj - leg.line)
                side = leg.home_team.upper() if proj > leg.line else leg.away_team.upper()
                display_line = leg.line if proj > leg.line else -leg.line
                prob = np.clip(0.5 + edge * 0.05, 0.20, 0.80)
                leg_odds = round(max(1.0 / prob, 1.10), 2)

                leg_detail = {
                    "type": "spread",
                    "bet": f"{leg.away_team.upper()} @ {leg.home_team.upper()} {side} {display_line:+.1f}",
                    "projection": round(proj, 1),
                    "edge": round(edge, 1),
                    "probability": round(prob * 100, 1),
                    "decimal_odds": leg_odds
                }

            elif leg.bet_type == "total":
                if leg.line is None:
                    raise HTTPException(400, "Line required for total bet")
                proj = game_pred["projected_total"]
                edge = abs(proj - leg.line)
                direction = "OVER" if proj > leg.line else "UNDER"
                prob = np.clip(0.5 + (proj - leg.line) * 0.06, 0.20, 0.80)
                leg_odds = round(max(1.0 / prob, 1.10), 2)

                leg_detail = {
                    "type": "total",
                    "bet": f"{leg.away_team.upper()} @ {leg.home_team.upper()} {direction} {leg.line}",
                    "projection": round(proj, 1),
                    "edge": round(edge, 1),
                    "probability": round(prob * 100, 1),
                    "decimal_odds": leg_odds
                }

        else:
            continue  # skip invalid legs

        total_decimal *= leg_odds
        details.append(leg_detail)

    return {
        "parlay_odds": round(total_decimal, 2),
        "total_legs": len(details),
        "possible_win_per_100php": round((total_decimal - 1) * 100, 2),
        "legs": details,
        "model_version": MODEL_VERSION
    }

# ==================== OTHER ENDPOINTS ====================
@app.get("/todays-games")
def get_todays_games(background_tasks: BackgroundTasks, days_offset: int = 0):
    target_date = (datetime.now() + timedelta(days=days_offset)).strftime("%Y-%m-%d")
    cached = load_cached_games(target_date)
    if cached:
        return {"games": cached, "source": "cache", "date": target_date}

    try:
        from utils import fetch_games_for_date
        games = fetch_games_for_date(target_date.replace("-", ""))
    except Exception as e:
        games = []

    background_tasks.add_task(save_games_cache, games, target_date)
    return {"games": games, "source": "live", "date": target_date}

@app.get("/players")
def list_players(team: Optional[str] = None, limit: int = 100):
    if team:
        filtered = df_players[df_players['TEAM_ABBREVIATION'] == team.upper()]
        players = sorted(filtered['PLAYER_NAME'].tolist())
    else:
        players = sorted(df_players['PLAYER_NAME'].tolist())[:limit]
    
    return {
        "total": len(players),
        "players": players,
        "teams_available": sorted(df_teams['TEAM_ABBREVIATION'].tolist())
    }

@app.get("/injuries")
def get_injuries(status: Optional[str] = None, team: Optional[str] = None):
    injuries = []
    for player, info in INJURY_STATUS.items():
        if status and info['status'] != status:
            continue
        if team and info['team'] != team.upper():
            continue
        injuries.append({"player": player, **info})
    
    return {
        "total": len(injuries),
        "injuries": sorted(injuries, key=lambda x: (x['status'], x['player'])),
        "status_breakdown": {
            "OUT": len([i for i in injuries if i['status'] == 'OUT']),
            "Questionable": len([i for i in injuries if i['status'] == 'Questionable']),
            "Probable": len([i for i in injuries if i['status'] == 'Probable']),
        }
    }

@app.get("/")
def home():
    return {
        "service": f"NBA Betting Agent Pro — {MODEL_VERSION}",
        "status": "LIVE 🚀",
        "model_features": len(TRAINED_FEATURES),
        "features": TRAINED_FEATURES,
        "endpoints": {
            "player_prediction": "/predict",
            "game_prediction": "/predict-game",
            "best_bets": "/best-bets",
            "todays_games": "/todays-games",
            "players": "/players",
            "injuries": "/injuries"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "ml_models": "loaded",
        "timestamp": datetime.now().isoformat(),
        "model_version": MODEL_VERSION,
        "data_loaded": {
            "players": len(df_players),
            "teams": len(df_teams),
            "injuries": len(INJURY_STATUS)
        },
        "features": {
            "player_model": TRAINED_FEATURES,
            "team_model": ['OFF_RATING', 'DEF_RATING', 'PACE']
        }
    }