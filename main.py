from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import joblib
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

from startup import df_players, df_teams, injuries_df

app = FastAPI(
    title="NBA Betting Agent Pro 2025-26",
    description="Complete betting intelligence with live odds, injuries, and predictions",
    version="2.0.0"
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cache settings
CACHE_DURATION = timedelta(hours=1)

def load_cached_games(date_str: str):
    """Load cached games for specific date if fresh"""
    cache_file = Path(f"data/games_{date_str}.json")
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache = json.load(f)
                cache_time = datetime.fromisoformat(cache['timestamp'])
                if datetime.now() - cache_time < CACHE_DURATION:
                    return cache['games']
        except Exception as e:
            print(f"Cache load error: {e}")
    return None

def save_games_cache(games: List[Dict], date_str: str):
    """Save games to cache for specific date"""
    cache_file = Path(f"data/games_{date_str}.json")
    cache = {
        'timestamp': datetime.now().isoformat(),
        'games': games,
        'date': date_str
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Cache save error: {e}")

# Add this near the top with your other cache functions
def load_cached_best_bets(date_str: str):
    cache_file = Path(f"data/bestbets_{date_str}.json")
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cache = json.load(f)
                cache_time = datetime.fromisoformat(cache['timestamp'])
                if datetime.now() - cache_time < CACHE_DURATION:
                    return cache['best_bets']
        except Exception as e:
            print(f"Best bets cache load error: {e}")
    return None

def save_best_bets_cache(best_bets: List[Dict], date_str: str):
    cache_file = Path(f"data/bestbets_{date_str}.json")
    cache = {
        'timestamp': datetime.now().isoformat(),
        'best_bets': best_bets,
        'date': date_str
    }
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Best bets cache save error: {e}")

# Injury status
try:
    INJURY_STATUS = {}
    for _, row in injuries_df.iterrows():
        name = str(row['player_name']).strip()
        status = str(row['status']).strip()
        injury = str(row['injury_type']).strip()
        if name and name != 'nan':
            INJURY_STATUS[name.title()] = {
                "status": status,
                "injury_type": injury,
                "team": str(row.get('team', 'UNK'))
            }
    print(f"Loaded {len(INJURY_STATUS)} injuries")
except Exception as e:
    print(f"Injury load error: {e}")
    INJURY_STATUS = {}

def get_injury_status(player_name: str) -> Dict:
    """Get detailed injury info"""
    name_key = player_name.strip().title()
    if name_key in INJURY_STATUS:
        return INJURY_STATUS[name_key]
    for key in INJURY_STATUS:
        if name_key in key or key in name_key:
            return INJURY_STATUS[key]
    return {"status": "Active", "injury_type": "None", "team": ""}

# === ENHANCED MODELS ===
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
    date: Optional[str] = Field(None, description="YYYY-MM-DD format")
    min_edge: float = Field(8.0, description="Minimum edge percentage")
    max_bets: int = Field(10, description="Max number of bets to return")

# === PLAYER PROPS ===
@app.post("/predict")
def predict_player(req: PlayerRequest):
    """Enhanced player prop prediction with injury awareness"""
    search = req.player_name.strip().lower()
    
    # Check injury FIRST
    injury_info = get_injury_status(req.player_name)
    if injury_info['status'] in ['OUT', 'Doubtful']:
        return {
            "player": req.player_name,
            "status": injury_info['status'],
            "injury_type": injury_info['injury_type'],
            "recommendation": "AVOID ALL PROPS",
            "edge_percent": 95 if injury_info['status'] == 'OUT' else 85,
            "warning": "Player unlikely to play - no reliable projection"
        }
    
    # Find player stats
    row = df_players[df_players['PLAYER_NAME'].str.lower().str.contains(search, case=False, na=False)]
    
    if row.empty:
        return {
            "player": req.player_name,
            "status": injury_info['status'],
            "recommendation": "NO DATA",
            "edge_percent": 0,
            "warning": "Player not found in current season stats"
        }
    
    p = row.iloc[0]
    name = p['PLAYER_NAME']
    
    def safe_float(val, default=0.0):
        try:
            return float(val) if pd.notna(val) else default
        except:
            return default
    
    # Base stats
    base = {
        "PTS": safe_float(p.get('PTS', 20.0)),
        "REB": safe_float(p.get('REB', 5.0)),
        "AST": safe_float(p.get('AST', 4.0)),
        "STL": safe_float(p.get('STL', 0.8)),
        "BLK": safe_float(p.get('BLK', 0.6)),
        "FG3M": safe_float(p.get('FG3M', 1.5)),
    }
    
    # Opponent adjustments (2025-26 defensive ratings)
    opp_adj = {
        "BOS": 0.92, "CLE": 0.94, "OKC": 0.91, "ORL": 0.93, "MIN": 0.95,
        "NYK": 0.96, "MIA": 0.97, "PHI": 0.97, "GSW": 0.98, "DEN": 0.98,
        "LAL": 0.99, "DAL": 1.00, "MIL": 1.00, "PHX": 1.01, "SAC": 1.02,
        "HOU": 1.03, "NOP": 1.04, "LAC": 1.02, "IND": 1.05, "ATL": 1.06,
        "CHI": 1.04, "MEM": 1.03, "SAS": 1.05, "POR": 1.07, "UTA": 1.06,
        "CHA": 1.08, "DET": 1.07, "TOR": 1.05, "BKN": 1.09, "WAS": 1.10
    }
    
    adj = opp_adj.get(req.opponent_abbr.upper(), 1.0)
    
    # Injury adjustment (Questionable = 90% production)
    injury_mult = 0.90 if injury_info['status'] == 'Questionable' else 1.0
    
    # Final projections
    projected = {k: v * adj * 1.03 * injury_mult for k, v in base.items()}
    
    # Market lines (realistic 2025-26)
    lines = {
        "PTS": req.pts_line or round(base["PTS"] - 0.5, 1),
        "REB": req.reb_line or round(base["REB"] - 0.3, 1),
        "AST": req.ast_line or round(base["AST"] - 0.4, 1),
        "STL": req.stl_line or 1.0,
        "BLK": req.blk_line or 0.8,
        "FG3M": req.three_pm_line or 1.5,
    }
    
    bets = {}
    for stat in projected:
        proj = projected[stat]
        line = lines[stat]
        edge = (proj - line) / line * 100 if line > 0 else 0
        
        # Confidence based on sample size
        confidence = "HIGH" if p.get('GP', 0) > 15 else "MEDIUM"
        
        bets[stat] = {
            "line": line,
            "projected": round(proj, 1),
            "best_bet": "OVER" if proj > line else "UNDER",
            "edge_percent": round(abs(edge), 1),
            "confidence": confidence
        }
    
    # Find best bet
    best_bet_stat = max(bets.items(), key=lambda x: x[1]["edge_percent"])
    
    return {
        "player": name,
        "team": p.get('TEAM_ABBREVIATION', 'UNK'),
        "vs": req.opponent_abbr.upper(),
        "status": injury_info['status'],
        "injury_type": injury_info.get('injury_type', 'None'),
        "predictions": bets,
        "top_bet": {
            "stat": best_bet_stat[0],
            **best_bet_stat[1]
        },
        "last_updated": p.get('FETCH_DATE', 'Unknown')
    }

# === GAME PREDICTION ===
TEAM_ID_TO_ABBR = {
    1610612737: "ATL", 1610612738: "BOS", 1610612751: "BKN", 1610612766: "CHA",
    1610612741: "CHI", 1610612739: "CLE", 1610612742: "DAL", 1610612743: "DEN",
    1610612765: "DET", 1610612744: "GSW", 1610612745: "HOU", 1610612754: "IND",
    1610612746: "LAC", 1610612747: "LAL", 1610612763: "MEM", 1610612748: "MIA",
    1610612749: "MIL", 1610612750: "MIN", 1610612740: "NOP", 1610612752: "NYK",
    1610612760: "OKC", 1610612753: "ORL", 1610612755: "PHI", 1610612756: "PHX",
    1610612757: "POR", 1610612758: "SAC", 1610612759: "SAS", 1610612761: "TOR",
    1610612762: "UTA", 1610612764: "WAS"
}

ODDS_API_TEAM_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "GS": "Golden State Warriors",
    "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NO": "New Orleans Pelicans",
    "NYK": "New York Knicks", "NY": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs", "SA": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

@app.post("/predict-game")
def predict_game(req: GamePredictionRequest):
    """Enhanced game prediction with live odds comparison"""
    home_input = req.home_team.strip().upper()
    away_input = req.away_team.strip().upper()

    if 'TEAM_ABBREVIATION' not in df_teams.columns:
        df_teams['TEAM_ABBREVIATION'] = df_teams['TEAM_ID'].map(TEAM_ID_TO_ABBR).fillna("UNK")
    
    h = df_teams[df_teams['TEAM_ABBREVIATION'] == home_input]
    a = df_teams[df_teams['TEAM_ABBREVIATION'] == away_input]
    
    if h.empty or a.empty:
        raise HTTPException(status_code=404, detail=f"Team not found: {home_input} or {away_input}")
    
    h = h.iloc[0]
    a = a.iloc[0]
    
    # Calculate advanced metrics
    def calc_ortg(row):
        poss = row['FGA'] + 0.44 * row['FTA'] - row['OREB'] + row['TOV']
        return (row['PTS'] / poss) * 100 if poss > 0 else 114.0
    
    def calc_drtg(row):
        ortg = calc_ortg(row)
        net = row.get('PLUS_MINUS', 0)
        return ortg - net
    
    def calc_pace(row):
        return row['FGA'] + 0.44 * row['FTA'] - row['OREB'] + row['TOV']
    
    home_ortg = calc_ortg(h) + 3.5  # Home advantage
    home_drtg = calc_drtg(h)
    home_pace = calc_pace(h)
    
    away_ortg = calc_ortg(a)
    away_drtg = calc_drtg(a)
    away_pace = calc_pace(a)
    
    avg_pace = (home_pace + away_pace) / 2
    
    # Projected total
    total_proj = ((home_ortg + away_drtg) / 2 + (away_ortg + home_drtg) / 2) * avg_pace / 100
    
    # Projected spread
    spread_proj = (home_ortg - away_drtg) - (away_ortg - home_drtg)
    winner = home_input if spread_proj > 0 else away_input
    
    # Betting recommendations
    recommendations = []

    # First to 10 Points
    def predict_first_to_10(home_abbr: str, away_abbr: str, home_ortg: float, away_ortg: float, avg_pace: float) -> dict:
        home_off_efficiency = home_ortg / 114.0
        away_off_efficiency = away_ortg / 114.0
        
        burst_factor = {
            "BOS": 1.15, "OKC": 1.18, "CLE": 1.12, "DEN": 1.10, "MIN": 1.08,
            "NYK": 1.07, "LAL": 1.14, "GSW": 1.16, "PHX": 1.13, "MIL": 1.11,
            "IND": 1.20, "SAC": 1.09, "MIA": 1.05, "ORL": 0.92, "PHI": 0.95,
            "NOP": 1.06, "DAL": 1.08, "HOU": 1.04, "ATL": 1.10,
            "CHA": 0.88, "DET": 0.90, "WAS": 0.87, "POR": 0.91, "SAS": 0.93,
            "TOR": 0.94, "BKN": 0.92, "UTA": 0.95, "CHI": 0.96, "MEM": 0.97,
            "LAC": 0.98
        }
        
        home_burst = burst_factor.get(home_abbr, 1.0)
        away_burst = burst_factor.get(away_abbr, 1.0)
        
        home_score = home_off_efficiency * home_burst * avg_pace
        away_score = away_off_efficiency * away_burst * avg_pace
        
        total = home_score + away_score
        home_prob = home_score / total if total > 0 else 0.5
        away_prob = 1 - home_prob
        
        favorite = home_abbr if home_prob > away_prob else away_abbr
        edge = abs(home_prob - 0.5) * 200
        
        return {
            "bet_type": "FIRST_TO_10",
            "pick": f"{favorite} -120" if edge > 8 else f"{favorite} -135",
            "probability": round(max(home_prob, away_prob) * 100, 1),
            "edge_percent": round(edge, 1),
            "confidence": "HIGH" if edge > 12 else "MEDIUM" if edge > 6 else "LOW"
        }

    first_to_10 = predict_first_to_10(home_input, away_input, home_ortg, away_ortg, avg_pace)
    if first_to_10["edge_percent"] >= 6:
        recommendations.append(first_to_10)
    
    if req.total_line:
        total_edge = abs(total_proj - req.total_line) / req.total_line * 100
        recommendations.append({
            "bet_type": "TOTAL",
            "pick": "OVER" if total_proj > req.total_line else "UNDER",
            "line": req.total_line,
            "projection": round(total_proj, 1),
            "edge_percent": round(total_edge, 1)
        })
    
    if req.spread_line:
        spread_edge = abs(abs(spread_proj) - abs(req.spread_line)) / abs(req.spread_line) * 100 if req.spread_line != 0 else 0
        cover_team = home_input if spread_proj + req.spread_line > 0 else away_input
        recommendations.append({
            "bet_type": "SPREAD",
            "pick": f"{cover_team} {req.spread_line:+.1f}",
            "line": req.spread_line,
            "projection": round(spread_proj, 1),
            "edge_percent": round(spread_edge, 1)
        })

    return {
        "game": f"{away_input} @ {home_input}",
        "game_time": req.game_time or "TBD",
        "projected_total": round(total_proj, 1),
        "projected_spread": f"{winner} {abs(spread_proj):.1f}",
        "winner": winner,
        "confidence": "HIGH" if max(h.get('GP', 0), a.get('GP', 0)) > 20 else "MEDIUM",
        "recommendations": sorted(recommendations, key=lambda x: x['edge_percent'], reverse=True),
        "pace_analysis": {
            "home_pace": round(home_pace, 1),
            "away_pace": round(away_pace, 1),
            "expected_pace": round(avg_pace, 1)
        },
        "first_to_10": first_to_10
    }

# === TODAY'S GAMES WITH DATE SUPPORT ===
@app.get("/todays-games")
def get_todays_games(
    background_tasks: BackgroundTasks,
    days_offset: int = 0
):
    """
    Fetch games with odds (cached for 1 hour)
    
    Args:
        days_offset: Days from today (0=today, 1=tomorrow, -1=yesterday)
    """
    # Calculate target date
    target_date = datetime.now() + timedelta(days=days_offset)
    date_str = target_date.strftime("%Y-%m-%d")
    nba_date_str = target_date.strftime("%Y%m%d")
    
    # Check cache first
    cached = load_cached_games(date_str)
    if cached:
        mins_ago = int((datetime.now() - datetime.fromisoformat(
            json.loads(Path(f"data/games_{date_str}.json").read_text())['timestamp']
        )).seconds / 60)
        return {
            "games": cached,
            "source": "cache",
            "updated": f"{mins_ago} min ago" if mins_ago > 0 else "just now",
            "date": date_str
        }
    
    # Fetch fresh data
    from utils import fetch_games_for_date
    
    try:
        games = fetch_games_for_date(nba_date_str)
        background_tasks.add_task(save_games_cache, games, date_str)
        return {
            "games": games,
            "source": "live",
            "updated": "just now",
            "date": date_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch games: {str(e)}")

# === BEST BETS - FULLY DATE-AWARE + CACHED + NO ERRORS ===
def get_best_bets_common(
    date: Optional[str] = None,
    days_offset: Optional[int] = None,
    min_edge: float = 8.0,
    max_bets: int = 10,
    background_tasks: Optional[BackgroundTasks] = None
):
    """Core logic — reused by GET and POST"""
    # Resolve target date
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
            date_str = date
        except:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    else:
        offset = days_offset or 0
        target_date = datetime.now() + timedelta(days=offset)
        date_str = target_date.strftime("%Y-%m-%d")

    nba_date_str = target_date.strftime("%Y%m%d")

    print(f"\n=== BEST BETS REQUEST ===")
    print(f"Date: {date_str} (NBA format: {nba_date_str})")
    print(f"Min Edge: {min_edge}%")

    # Try cache
    cached = load_cached_best_bets(date_str)
    if cached is not None:
        print(f"✓ Returning cached best bets ({len(cached)} bets)")
        try:
            with open(f"data/bestbets_{date_str}.json") as f:
                ts = json.load(f).get('timestamp', '')
                mins_ago = int((datetime.now() - datetime.fromisoformat(ts)).seconds / 60)
                updated_str = f"{mins_ago} min ago" if mins_ago > 0 else "just now"
        except:
            updated_str = "cached"
        
        return {
            "date": date_str,
            "total_opportunities": len(cached),
            "best_bets": cached[:max_bets],
            "min_edge_filter": min_edge,
            "source": "cache",
            "updated": updated_str
        }

    # Fetch games for the date
    print(f"⚙ Fetching fresh games for {nba_date_str}...")
    from utils import fetch_games_with_odds_for_date
    
    try:
        games = fetch_games_with_odds_for_date(nba_date_str)
    except Exception as e:
        print(f"❌ Error fetching games: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch games: {str(e)}")

    if not games:
        print(f"⚠ No games found for {date_str}")
        return {
            "date": date_str,
            "total_opportunities": 0,
            "best_bets": [],
            "min_edge_filter": min_edge,
            "note": "No games scheduled for this date"
        }

    print(f"✓ Found {len(games)} games, analyzing...")
    
    all_bets = []
    abbr_fix = {"NY": "NYK", "NO": "NOP", "GS": "GSW", "SA": "SAS"}
    
    for i, game in enumerate(games, 1):
        home = abbr_fix.get(game.get('home_team'), game.get('home_team'))
        away = abbr_fix.get(game.get('away_team'), game.get('away_team'))
        
        print(f"   [{i}/{len(games)}] {away} @ {home}", end=" ")

        try:
            pred = predict_game(GamePredictionRequest(
                home_team=home,
                away_team=away,
                spread_line=game.get('spread'),
                total_line=game.get('total'),
                game_time=game.get('game_time')
            ))

            game_bets = 0
            for rec in pred.get('recommendations', []):
                if rec['edge_percent'] >= min_edge:
                    all_bets.append({
                        "game": pred['game'],
                        "game_time": game.get('game_time', 'TBD'),
                        "bet_type": rec['bet_type'],
                        "pick": rec['pick'],
                        "line": rec.get('line'),
                        "projection": rec.get('projection'),
                        "edge_percent": round(rec['edge_percent'], 1),
                    })
                    game_bets += 1
            
            print(f"→ {game_bets} bet(s) found" if game_bets > 0 else "→ no value")
            
        except Exception as e:
            print(f"→ ERROR: {e}")
            continue

    print(f"\n✓ Analysis complete: {len(all_bets)} total opportunities")
    
    if len(all_bets) == 0:
        return {
            "date": date_str,
            "total_opportunities": 0,
            "best_bets": [],
            "min_edge_filter": min_edge,
            "note": f"No bets found with {min_edge}% minimum edge"
        }

    # Sort by edge
    all_bets.sort(key=lambda x: x['edge_percent'], reverse=True)
    top_bets = all_bets[:max_bets]

    # Save cache in background
    if background_tasks:
        background_tasks.add_task(save_best_bets_cache, top_bets, date_str)
        print(f"💾 Caching {len(top_bets)} best bets...")

    return {
        "date": date_str,
        "total_opportunities": len(all_bets),
        "best_bets": top_bets,
        "min_edge_filter": min_edge,
        "source": "live",
        "updated": "just now"
    }

# === FINAL ENDPOINTS (No conflicts!) ===
@app.get("/best-bets")
def best_bets_get(
    date: Optional[str] = None,
    days_offset: Optional[int] = None,
    min_edge: float = 8.0,
    max_bets: int = 10,
    background_tasks: BackgroundTasks = None
):
    return get_best_bets_common(date, days_offset, min_edge, max_bets, background_tasks)


@app.post("/best-bets")
def best_bets_post(
    req: BestBetsRequest,
    background_tasks: BackgroundTasks = None
):
    return get_best_bets_common(
        date=req.date,
        days_offset=None,
        min_edge=req.min_edge,
        max_bets=req.max_bets,
        background_tasks=background_tasks
    )

# === UTILITY ENDPOINTS ===
@app.get("/")
def home():
    return {
        "service": "NBA Betting Agent Pro 2025-26",
        "version": "2.0.0",
        "features": [
            "Live player props with injury tracking",
            "Game predictions with odds comparison",
            "Today's games with live odds",
            "Best bets analyzer",
            "Injury status tracking",
            "Date navigation (past/future games)"
        ],
        "endpoints": {
            "player_props": "/predict",
            "game_prediction": "/predict-game",
            "todays_games": "/todays-games?days_offset=0",
            "best_bets": "/best-bets",
            "injuries": "/injuries",
            "players": "/players"
        }
    }

@app.get("/players")
def list_players(team: Optional[str] = None):
    """List all players, optionally filtered by team"""
    if team:
        filtered = df_players[df_players['TEAM_ABBREVIATION'] == team.upper()]
        return {"team": team.upper(), "players": sorted(filtered['PLAYER_NAME'].tolist())}
    return {"total": len(df_players), "players": sorted(df_players['PLAYER_NAME'].tolist())}

@app.get("/injuries")
def get_injuries(status: Optional[str] = None):
    """Get current injury report"""
    injuries = []
    for player, info in INJURY_STATUS.items():
        if status and info['status'] != status:
            continue
        injuries.append({"player": player, **info})
    
    return {
        "total": len(injuries),
        "filter": status,
        "injuries": sorted(injuries, key=lambda x: x['status'])
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": {
            "players": len(df_players),
            "teams": len(df_teams),
            "injuries": len(INJURY_STATUS)
        }
    }

# Add this endpoint to your main.py for debugging:

@app.get("/debug/games")
def debug_games(days_offset: int = 0):
    """Debug endpoint to see what games are being fetched"""
    from utils import fetch_games_with_odds_for_date
    
    target_date = datetime.now() + timedelta(days=days_offset)
    date_str = target_date.strftime("%Y-%m-%d")
    nba_date_str = target_date.strftime("%Y%m%d")
    
    print(f"\n=== DEBUG GAMES ===")
    print(f"Date: {date_str}")
    print(f"NBA Format: {nba_date_str}")
    
    try:
        games = fetch_games_with_odds_for_date(nba_date_str)
        
        return {
            "date": date_str,
            "nba_date": nba_date_str,
            "games_count": len(games),
            "games": games,
            "status": "success"
        }
    except Exception as e:
        import traceback
        return {
            "date": date_str,
            "nba_date": nba_date_str,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }