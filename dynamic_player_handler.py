"""
Dynamic Player Management System
Handles: Active players, inactive players, rookies, injuries, historical data
Updates: Automatically syncs with multiple sources
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ==================== MULTI-SOURCE PLAYER FETCHING ====================

class PlayerDataAggregator:
    """Aggregates player data from multiple sources"""
    
    def __init__(self):
        self.sources = {
            'nba_stats': self.fetch_from_nba_stats,
            'balldontlie': self.fetch_from_balldontlie,
            'sportsdata': self.fetch_from_sportsdata,
            'historical': self.load_historical_data
        }
    
    def fetch_all_players(self, season: str = '2025-26') -> pd.DataFrame:
        """
        Fetch from all sources and merge intelligently
        Priority: NBA Stats API > BallDontLie > Historical
        """
        logger.info("Fetching players from multiple sources...")
        
        all_players = []
        
        # Source 1: NBA Stats API (active players with current stats)
        try:
            nba_players = self.fetch_from_nba_stats(season)
            if nba_players:
                logger.info(f"✓ NBA Stats: {len(nba_players)} players")
                all_players.extend(nba_players)
        except Exception as e:
            logger.error(f"NBA Stats failed: {e}")
        
        # Source 2: BallDontLie (comprehensive roster, includes inactive)
        try:
            bdl_players = self.fetch_from_balldontlie(season)
            if bdl_players:
                logger.info(f"✓ BallDontLie: {len(bdl_players)} players")
                all_players.extend(bdl_players)
        except Exception as e:
            logger.error(f"BallDontLie failed: {e}")
        
        # Source 3: Historical data (for recently inactive players)
        try:
            historical = self.load_historical_data()
            if historical:
                logger.info(f"✓ Historical: {len(historical)} players")
                all_players.extend(historical)
        except Exception as e:
            logger.error(f"Historical load failed: {e}")
        
        # Merge and deduplicate
        if not all_players:
            raise Exception("No player data available from any source")
        
        df = pd.DataFrame(all_players)
        df = self.merge_and_deduplicate(df)
        
        # Mark player status
        df = self.mark_player_status(df)
        
        return df
    
    def fetch_from_nba_stats(self, season: str) -> List[Dict]:
        """Fetch from official NBA Stats API"""
        from utils import fetch_current_season_stats
        
        try:
            df = fetch_current_season_stats()
            
            # CRITICAL FIX: Add SOURCE column
            df['SOURCE'] = 'nba_stats'
            
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"NBA Stats fetch failed: {e}")
            return []
    
    def fetch_from_balldontlie(self, season: str) -> List[Dict]:
        """
        Fetch from BallDontLie API (free, comprehensive)
        Includes all players, even if they haven't played yet
        """
        url = "https://www.balldontlie.io/api/v1/players"
        params = {
            'per_page': 100,
            'season': int(season[:4])  # 2025
        }
        
        all_players = []
        page = 1
        max_pages = 10  # Safety limit
        
        try:
            while page <= max_pages:
                params['page'] = page
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code != 200:
                    break
                
                data = response.json()
                players = data.get('data', [])
                
                if not players:
                    break
                
                # Map to our schema
                for p in players:
                    all_players.append({
                        'PLAYER_NAME': f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
                        'TEAM_ABBREVIATION': p.get('team', {}).get('abbreviation', 'FA'),
                        'POSITION': p.get('position', 'G'),
                        'HEIGHT': p.get('height_feet', 6) * 12 + p.get('height_inches', 0),
                        'WEIGHT': p.get('weight_pounds', 200),
                        # Defaults for stats (will be overridden if they have data)
                        'GP': 0,
                        'MIN': 0.0,
                        'PTS': 0.0,
                        'FGA': 0.0,
                        'FG3A': 0.0,
                        'FTA': 0.0,
                        'AST': 0.0,
                        'REB': 0.0,
                        'STL': 0.0,
                        'BLK': 0.0,
                        'TOV': 0.0,
                        'FG_PCT': 0.0,
                        'FG3_PCT': 0.0,
                        'FT_PCT': 0.0,
                        'AGE': 25,
                        'PACE': 100.0,
                        'SOURCE': 'balldontlie',  # ✓ SOURCE added
                        'STATUS': 'INACTIVE'  # Default, will be updated
                    })
                
                page += 1
            
            return all_players
            
        except Exception as e:
            logger.error(f"BallDontLie fetch failed: {e}")
            return []
    
    def fetch_from_sportsdata(self, season: str) -> List[Dict]:
        """
        Fetch from SportsData.io API (requires key)
        More comprehensive injury and status data
        """
        api_key = os.getenv('SPORTSDATA_API_KEY')
        if not api_key:
            return []
        
        url = f"https://api.sportsdata.io/v3/nba/scores/json/Players"
        headers = {'Ocp-Apim-Subscription-Key': api_key}
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                players = response.json()
                # Add SOURCE column
                mapped_players = self.map_sportsdata_to_schema(players)
                for p in mapped_players:
                    p['SOURCE'] = 'sportsdata'
                return mapped_players
        except Exception as e:
            logger.error(f"SportsData fetch failed: {e}")
        
        return []
    
    def map_sportsdata_to_schema(self, players: List[Dict]) -> List[Dict]:
        """Map SportsData schema to our standard schema"""
        # Add your mapping logic here if you use this API
        return []
    
    def load_historical_data(self) -> List[Dict]:
        """
        Load historical player data for recently inactive players
        This ensures we have data for players who were active last season
        """
        historical_files = [
            DATA_DIR / "2024_25_players.csv",
            DATA_DIR / "2023_24_players.csv",
            DATA_DIR / "historical_players.csv"
        ]
        
        all_historical = []
        
        for file in historical_files:
            if file.exists():
                try:
                    df = pd.read_csv(file)
                    
                    # CRITICAL FIX: Add SOURCE column if missing
                    if 'SOURCE' not in df.columns:
                        df['SOURCE'] = 'historical'
                    
                    if 'STATUS' not in df.columns:
                        df['STATUS'] = 'INACTIVE'  # Assume inactive unless proven otherwise
                    
                    all_historical.extend(df.to_dict('records'))
                    logger.info(f"Loaded {len(df)} players from {file.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file}: {e}")
        
        return all_historical
    
    def merge_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Intelligently merge players from multiple sources
        Priority: NBA Stats > BallDontLie > Historical
        FIXED: Handles missing SOURCE column gracefully
        """
        if df.empty:
            return df
        
        # CRITICAL FIX: Ensure SOURCE column exists
        if 'SOURCE' not in df.columns:
            logger.warning("SOURCE column missing! Adding default...")
            df['SOURCE'] = 'unknown'
        
        # Normalize names for matching
        df['NAME_NORMALIZED'] = df['PLAYER_NAME'].str.lower().str.strip()
        
        # Sort by priority (NBA Stats first)
        source_priority = {
            'nba_stats': 1, 
            'balldontlie': 2, 
            'sportsdata': 3, 
            'historical': 4,
            'unknown': 99
        }
        
        df['SOURCE_PRIORITY'] = df['SOURCE'].map(lambda x: source_priority.get(x, 99))
        df = df.sort_values('SOURCE_PRIORITY')
        
        # Keep first occurrence of each player (highest priority)
        df_deduped = df.drop_duplicates(subset=['NAME_NORMALIZED'], keep='first')
        
        # Clean up
        df_deduped = df_deduped.drop(['NAME_NORMALIZED', 'SOURCE_PRIORITY'], axis=1)
        
        logger.info(f"Merged {len(df)} → {len(df_deduped)} unique players")
        
        return df_deduped
    
    def mark_player_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark player status based on games played and recent activity
        ACTIVE, INACTIVE, ROOKIE, INJURED
        """
        if 'GP' not in df.columns:
            df['GP'] = 0
        
        df['PLAYER_STATUS'] = 'UNKNOWN'
        
        # Active: Playing this season
        df.loc[df['GP'] >= 1, 'PLAYER_STATUS'] = 'ACTIVE'
        
        # Rookie: No historical data, on roster but not played yet
        df.loc[(df['GP'] == 0) & (df['SOURCE'] == 'balldontlie'), 'PLAYER_STATUS'] = 'ROOKIE'
        
        # Inactive: Historical data but no current games
        df.loc[(df['GP'] == 0) & (df['SOURCE'] == 'historical'), 'PLAYER_STATUS'] = 'INACTIVE'
        
        return df

# ==================== INJURY-AWARE TRAINING DATA ====================

class InjuryAwareDataBuilder:
    """
    Builds training data that learns from injury patterns
    Incorporates injury history into feature engineering
    """
    
    def __init__(self):
        self.injury_history_file = DATA_DIR / "injury_history.json"
        self.injury_history = self.load_injury_history()
    
    def load_injury_history(self) -> Dict:
        """Load historical injury data"""
        if self.injury_history_file.exists():
            with open(self.injury_history_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_injury_history(self):
        """Save injury history"""
        with open(self.injury_history_file, 'w') as f:
            json.dump(self.injury_history, f, indent=2)
    
    def update_injury_history(self, current_injuries: List[Dict]):
        """
        Update injury history with current injuries
        Tracks: injury type, duration, recurrence
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        for injury in current_injuries:
            player = injury.get('player_name')
            if not player:
                continue
            
            if player not in self.injury_history:
                self.injury_history[player] = {
                    'injuries': [],
                    'total_days_injured': 0,
                    'injury_count': 0
                }
            
            # Check if this is a new injury or continuation
            history = self.injury_history[player]
            last_injury = history['injuries'][-1] if history['injuries'] else None
            
            if last_injury and last_injury.get('status') == injury.get('status'):
                # Continuation - update duration
                start_date = datetime.fromisoformat(last_injury['start_date'])
                duration = (datetime.now() - start_date).days
                last_injury['duration_days'] = duration
                last_injury['last_updated'] = today
            else:
                # New injury
                history['injuries'].append({
                    'injury_type': injury.get('injury_type', 'Unknown'),
                    'status': injury.get('status', 'Unknown'),
                    'start_date': today,
                    'duration_days': 0,
                    'last_updated': today
                })
                history['injury_count'] += 1
        
        self.save_injury_history()
    
    def calculate_injury_risk_features(self, player_name: str) -> Dict:
        """
        Calculate injury risk features for a player
        These become part of the training data
        """
        if player_name not in self.injury_history:
            return {
                'INJURY_RISK_SCORE': 0.0,
                'DAYS_SINCE_INJURY': 999,
                'INJURY_COUNT_LAST_YEAR': 0,
                'CHRONIC_INJURY_FLAG': 0
            }
        
        history = self.injury_history[player_name]
        injuries = history['injuries']
        
        # Calculate features
        recent_injuries = [
            inj for inj in injuries
            if datetime.fromisoformat(inj['start_date']) > datetime.now() - timedelta(days=365)
        ]
        
        # Days since last injury
        if injuries:
            last_injury_date = datetime.fromisoformat(injuries[-1]['start_date'])
            days_since = (datetime.now() - last_injury_date).days
        else:
            days_since = 999
        
        # Chronic injury detection (same injury type multiple times)
        injury_types = [inj.get('injury_type', 'Unknown') for inj in injuries]
        chronic_flag = 1 if len(injury_types) != len(set(injury_types)) else 0
        
        # Risk score (0-1)
        risk_score = min(1.0, (
            len(recent_injuries) * 0.2 +
            chronic_flag * 0.3 +
            (1.0 - min(days_since / 365, 1.0)) * 0.3
        ))
        
        return {
            'INJURY_RISK_SCORE': round(risk_score, 3),
            'DAYS_SINCE_INJURY': days_since,
            'INJURY_COUNT_LAST_YEAR': len(recent_injuries),
            'CHRONIC_INJURY_FLAG': chronic_flag
        }
    
    def enhance_training_data(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Add injury-aware features to training data
        """
        logger.info("Adding injury-aware features to training data...")
        
        # Calculate injury features for each player
        injury_features = []
        for player in df_players['PLAYER_NAME']:
            features = self.calculate_injury_risk_features(player)
            injury_features.append(features)
        
        # Add to dataframe
        df_injury = pd.DataFrame(injury_features)
        df_enhanced = pd.concat([df_players, df_injury], axis=1)
        
        logger.info(f"Added {len(df_injury.columns)} injury-aware features")
        
        return df_enhanced

# ==================== DYNAMIC PLAYER LOOKUP ====================

class DynamicPlayerLookup:
    """
    Handles player lookups with multiple fallback strategies
    Supports: Active, Inactive, Rookies, Name variations
    """
    
    def __init__(self, df_all_players: pd.DataFrame):
        self.df_all = df_all_players
        self.build_lookup_indices()
    
    def build_lookup_indices(self):
        """Build fast lookup indices"""
        from name_mapper import normalize_name_lower
        
        # Normalized name index
        self.name_index = {}
        for idx, row in self.df_all.iterrows():
            norm_name = normalize_name_lower(row['PLAYER_NAME'])
            self.name_index[norm_name] = idx
        
        logger.info(f"Built lookup index for {len(self.name_index)} players")
    
    def find_player(self, query: str) -> Optional[Dict]:
        """
        Find player with comprehensive fallback strategy
        Returns: player dict with metadata about match quality
        """
        from name_mapper import normalize_name_lower, get_canonical_name
        
        # Step 1: Canonical name resolution
        canonical = get_canonical_name(query)
        norm_query = normalize_name_lower(canonical)
        
        # Step 2: Exact match
        if norm_query in self.name_index:
            idx = self.name_index[norm_query]
            player = self.df_all.iloc[idx].to_dict()
            player['MATCH_TYPE'] = 'EXACT'
            player['MATCH_CONFIDENCE'] = 1.0
            return player
        
        # Step 3: Fuzzy match
        from difflib import get_close_matches
        close_matches = get_close_matches(norm_query, self.name_index.keys(), n=1, cutoff=0.75)
        
        if close_matches:
            matched_name = close_matches[0]
            idx = self.name_index[matched_name]
            player = self.df_all.iloc[idx].to_dict()
            player['MATCH_TYPE'] = 'FUZZY'
            player['MATCH_CONFIDENCE'] = 0.8
            player['ORIGINAL_QUERY'] = query
            return player
        
        # Step 4: Partial name match (first or last name only)
        query_parts = norm_query.split()
        for part in query_parts:
            if len(part) >= 3:  # Minimum 3 chars
                for name, idx in self.name_index.items():
                    if part in name.split():
                        player = self.df_all.iloc[idx].to_dict()
                        player['MATCH_TYPE'] = 'PARTIAL'
                        player['MATCH_CONFIDENCE'] = 0.6
                        player['ORIGINAL_QUERY'] = query
                        return player
        
        return None
    
    def get_player_with_fallback(self, query: str) -> Dict:
        """
        Get player data with intelligent fallback
        Returns data even for inactive/rookie players
        """
        player = self.find_player(query)
        
        if not player:
            return {
                'error': 'PLAYER_NOT_FOUND',
                'query': query,
                'suggestions': self.get_similar_players(query, n=5)
            }
        
        # Check player status
        status = player.get('PLAYER_STATUS', 'UNKNOWN')
        
        if status == 'ACTIVE':
            # Has current season data
            player['DATA_SOURCE'] = 'CURRENT_STATS'
            player['PREDICTION_CONFIDENCE'] = 'HIGH'
        
        elif status == 'ROOKIE':
            # No games yet - use projections or college stats
            player['DATA_SOURCE'] = 'PROJECTED'
            player['PREDICTION_CONFIDENCE'] = 'LOW'
            player['NOTE'] = 'Rookie - limited data available'
        
        elif status == 'INACTIVE':
            # Use last season data
            player['DATA_SOURCE'] = 'HISTORICAL'
            player['PREDICTION_CONFIDENCE'] = 'MEDIUM'
            player['NOTE'] = 'Using historical data - player may be inactive'
        
        return player
    
    def get_similar_players(self, query: str, n: int = 5) -> List[str]:
        """Get similar player names for suggestions"""
        from difflib import get_close_matches
        from name_mapper import normalize_name_lower
        
        norm_query = normalize_name_lower(query)
        all_names = list(self.name_index.keys())
        
        matches = get_close_matches(norm_query, all_names, n=n, cutoff=0.5)
        
        # Convert back to original names
        suggestions = []
        for match in matches:
            idx = self.name_index[match]
            original_name = self.df_all.iloc[idx]['PLAYER_NAME']
            suggestions.append(original_name)
        
        return suggestions

# ==================== MAIN ORCHESTRATOR ====================

def refresh_player_database(season: str = '2025-26'):
    """
    Main function to refresh entire player database
    Call this from train.py or daily_update.py
    """
    logger.info("="*70)
    logger.info("REFRESHING PLAYER DATABASE")
    logger.info("="*70)
    
    # Step 1: Fetch from all sources
    aggregator = PlayerDataAggregator()
    df_all_players = aggregator.fetch_all_players(season)
    
    # Step 2: Update injury history
    from utils import fetch_live_injuries
    try:
        current_injuries = fetch_live_injuries()
    except Exception as e:
        logger.warning(f"Could not fetch injuries: {e}")
        current_injuries = []
    
    injury_builder = InjuryAwareDataBuilder()
    if current_injuries:
        injury_builder.update_injury_history(current_injuries)
    
    # Step 3: Enhance with injury features
    df_enhanced = injury_builder.enhance_training_data(df_all_players)
    
    # Step 4: Save
    df_enhanced.to_csv(DATA_DIR / "2025_26_players_complete.csv", index=False)
    
    # Also save active-only for backward compatibility
    df_active = df_enhanced[df_enhanced['PLAYER_STATUS'] == 'ACTIVE']
    df_active.to_csv(DATA_DIR / "2025_26_players.csv", index=False)
    
    logger.info(f"✓ Saved complete database: {len(df_enhanced)} players")
    logger.info(f"  - Active: {len(df_active)}")
    logger.info(f"  - Rookie: {len(df_enhanced[df_enhanced['PLAYER_STATUS'] == 'ROOKIE'])}")
    logger.info(f"  - Inactive: {len(df_enhanced[df_enhanced['PLAYER_STATUS'] == 'INACTIVE'])}")
    
    return df_enhanced

# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Refresh database
    df = refresh_player_database('2025-26')
    
    # Test dynamic lookup
    lookup = DynamicPlayerLookup(df)
    
    # Test cases
    test_queries = [
        "Luka Doncic",      # Active
        "Jayson Tatum",     # Injured
        "Victor Wembanyama", # Rookie
        "Kyrie Irving",     # May be inactive
        "Luca Doncic",      # Typo
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = lookup.get_player_with_fallback(query)
        print(f"  Result: {result.get('PLAYER_NAME', 'NOT FOUND')}")
        print(f"  Status: {result.get('PLAYER_STATUS', 'N/A')}")
        print(f"  Confidence: {result.get('PREDICTION_CONFIDENCE', 'N/A')}")