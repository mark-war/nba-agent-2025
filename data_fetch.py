import requests
from typing import List, Dict

def fetch_nba_roster(season: str = '2025-26') -> List[Dict]:
    """Fetch full NBA roster via free API (e.g., balldontlie or rapidapi)"""
    url = f"https://www.balldontlie.io/api/v1/players?season={season[:4]}&per_page=100"
    response = requests.get(url)
    if response.status_code == 200:
        players = response.json()['data']
        # Map to your expected columns (add defaults for advanced stats)
        return [
            {
                'PLAYER_NAME': p['first_name'] + ' ' + p['last_name'],
                'TEAM_ABBREVIATION': p.get('team', {}).get('abbreviation', 'FA'),
                'MIN': 32.0, 'PTS': 25.0, 'FGA': 18.0, 'FG3A': 7.0, 'FTA': 6.0,
                'AST': 4.0, 'REB': 8.0, 'STL': 1.0, 'BLK': 0.5, 'TOV': 2.5,
                'FG_PCT': 0.47, 'FG3_PCT': 0.38, 'FT_PCT': 0.85, 'AGE': 27,
                'USG_PCT': 30.0, 'PACE': 100.0  # Defaults—update with real stats
            } for p in players
        ]
    return []  # Fallback empty

def fetch_live_injuries() -> List[Dict]:
    """Fetch from ESPN or Rotowire API"""
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    response = requests.get(url)
    if response.status_code == 200:
        injuries = []
        for game in response.json().get('events', []):
            for note in game.get('notes', []):
                if 'injury' in note.get('headline', '').lower():
                    # Parse: e.g., {'player_name': 'Jayson Tatum', 'status': 'Questionable', ...}
                    injuries.append(note)  # Customize parsing
        return injuries
    return []