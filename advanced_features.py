# advanced_features.py - Enhanced Stats and Predictions
"""
Additional features to add to your NBA Betting Agent:
- Advanced player props
- Combo predictions
- Rest day analysis
- Matchup history
- In-game momentum
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ===== ADVANCED PLAYER PROPS =====

def predict_player_combo_props(player_data: Dict, opponent_defense: Dict) -> Dict:
    """
    Predict combination props like PRA (Points + Rebounds + Assists)
    """
    
    pts = player_data.get('PTS_PG', 20.0)
    reb = player_data.get('REB_PG', 6.0)
    ast = player_data.get('AST_PG', 5.0)
    stl = player_data.get('STL_PG', 1.0)
    blk = player_data.get('BLK_PG', 0.5)
    
    # Calculate combo props
    pra = pts + reb + ast
    pr = pts + reb
    pa = pts + ast
    ra = reb + ast
    sb = stl + blk  # Stocks (steals + blocks)
    
    # Adjust based on opponent defense
    opp_rating = opponent_defense.get('DEF_RATING', 110.0)
    league_avg = 112.0
    
    adjustment = 1.0
    if opp_rating < league_avg - 3:
        adjustment = 1.05  # Easier matchup
    elif opp_rating > league_avg + 3:
        adjustment = 0.95  # Harder matchup
    
    return {
        'PRA': round(pra * adjustment, 1),
        'PR': round(pr * adjustment, 1),
        'PA': round(pa * adjustment, 1),
        'RA': round(ra * adjustment, 1),
        'stocks': round(sb * adjustment, 1),
        'confidence': 'High' if abs(opp_rating - league_avg) > 3 else 'Medium'
    }

def predict_double_double_probability(player_data: Dict) -> float:
    """
    Predict probability of a double-double
    """
    
    pts = player_data.get('PTS_PG', 20.0)
    reb = player_data.get('REB_PG', 6.0)
    ast = player_data.get('AST_PG', 5.0)
    
    # Count how many stats are already close to 10
    close_stats = sum([
        pts >= 8.0,
        reb >= 8.0,
        ast >= 8.0
    ])
    
    if close_stats >= 2:
        base_prob = 0.75
    elif close_stats == 1:
        base_prob = 0.35
    else:
        base_prob = 0.05
    
    # Adjust based on consistency
    pts_std = player_data.get('PTS_STD', 5.0)
    consistency_factor = max(0.7, min(1.3, 1.0 - (pts_std / pts) * 0.5))
    
    final_prob = min(0.95, base_prob * consistency_factor)
    
    return round(final_prob, 3)

def predict_triple_double_probability(player_data: Dict) -> float:
    """
    Predict probability of a triple-double
    """
    
    pts = player_data.get('PTS_PG', 20.0)
    reb = player_data.get('REB_PG', 6.0)
    ast = player_data.get('AST_PG', 5.0)
    
    # Only players averaging 7+ in all three categories have realistic chance
    if pts >= 7 and reb >= 7 and ast >= 7:
        base_prob = 0.15
        
        # Boost for elite playmakers
        if ast >= 9:
            base_prob *= 1.5
        
        return min(0.45, base_prob)
    
    return 0.01

# ===== REST DAY ANALYSIS =====

def calculate_rest_impact(player_data: Dict, rest_days: int) -> float:
    """
    Calculate impact of rest days on performance
    
    Research shows:
    - 0 days (back-to-back): -8% performance
    - 1 day: baseline
    - 2 days: +3% performance
    - 3+ days: +5% performance
    """
    
    if rest_days == 0:
        return 0.92  # Back-to-back penalty
    elif rest_days == 1:
        return 1.0   # Normal
    elif rest_days == 2:
        return 1.03  # Slight boost
    else:
        return 1.05  # Well-rested boost

def analyze_back_to_back_impact(player_data: Dict, is_back_to_back: bool) -> Dict:
    """
    Detailed analysis of back-to-back game impact
    """
    
    age = player_data.get('AGE', 27)
    minutes = player_data.get('MIN_PG', 33.0)
    
    if not is_back_to_back:
        return {'multiplier': 1.0, 'risk': 'Low'}
    
    # Older players and high-minute players affected more
    base_penalty = 0.08
    
    if age > 32:
        base_penalty += 0.03
    
    if minutes > 35:
        base_penalty += 0.02
    
    multiplier = 1.0 - base_penalty
    risk = 'High' if multiplier < 0.90 else 'Medium'
    
    return {
        'multiplier': round(multiplier, 3),
        'risk': risk,
        'recommendation': 'FADE' if multiplier < 0.90 else 'MONITOR'
    }

# ===== MATCHUP HISTORY =====

def analyze_player_vs_opponent_history(
    player_name: str,
    opponent_team: str,
    historical_data: pd.DataFrame
) -> Dict:
    """
    Analyze player's historical performance vs specific opponent
    """
    
    # Filter for this player vs this opponent
    matchups = historical_data[
        (historical_data['PLAYER_NAME'] == player_name) &
        (historical_data['OPPONENT'] == opponent_team)
    ]
    
    if len(matchups) < 3:
        return {
            'sample_size': len(matchups),
            'confidence': 'Low',
            'avg_pts': None
        }
    
    avg_pts = matchups['PTS'].mean()
    std_pts = matchups['PTS'].std()
    recent_trend = matchups.tail(3)['PTS'].mean()
    
    return {
        'sample_size': len(matchups),
        'avg_pts': round(avg_pts, 1),
        'std_pts': round(std_pts, 1),
        'recent_avg': round(recent_trend, 1),
        'confidence': 'High' if len(matchups) >= 5 else 'Medium'
    }

# ===== DEFENSIVE MATCHUP ANALYSIS =====

def analyze_positional_matchup(
    player_position: str,
    opponent_team_defense: Dict
) -> Dict:
    """
    Analyze how opponent defends against specific positions
    """
    
    # Points allowed by position (example data structure)
    position_defense = opponent_team_defense.get(f'{player_position}_PTS_ALLOWED', 25.0)
    league_avg = 25.0
    
    diff = position_defense - league_avg
    
    if diff > 3:
        rating = "Favorable"
        multiplier = 1.08
    elif diff < -3:
        rating = "Tough"
        multiplier = 0.92
    else:
        rating = "Neutral"
        multiplier = 1.0
    
    return {
        'matchup_rating': rating,
        'pts_allowed_to_position': round(position_defense, 1),
        'league_avg': round(league_avg, 1),
        'difference': round(diff, 1),
        'multiplier': multiplier
    }

# ===== USAGE RATE IMPACT =====

def predict_usage_with_injuries(
    player_data: Dict,
    team_injuries: List[str]
) -> Dict:
    """
    Predict how player's usage rate changes with teammate injuries
    """
    
    base_usage = player_data.get('USG_PCT', 25.0)
    
    # Calculate total usage lost from injuries
    # Assume each injured starter = +2% usage for key players
    usage_boost = len(team_injuries) * 2.0
    
    projected_usage = min(40.0, base_usage + usage_boost)
    
    # Higher usage typically means more points but lower efficiency
    pts_boost = (projected_usage - base_usage) * 0.6
    efficiency_penalty = (projected_usage - base_usage) * 0.01
    
    return {
        'base_usage': round(base_usage, 1),
        'projected_usage': round(projected_usage, 1),
        'pts_boost': round(pts_boost, 1),
        'efficiency_impact': round(efficiency_penalty, 3),
        'recommendation': 'BET OVER' if pts_boost > 2 else 'MONITOR'
    }

# ===== PACE IMPACT =====

def analyze_pace_impact(
    player_data: Dict,
    home_pace: float,
    away_pace: float
) -> Dict:
    """
    Analyze how game pace affects player performance
    """
    
    # Calculate expected game pace
    game_pace = (home_pace + away_pace) / 2
    league_avg_pace = 100.0
    
    # Players typically score more in faster games
    pace_diff = game_pace - league_avg_pace
    pts_per_pace_point = 0.15  # Each pace point = ~0.15 ppg
    
    pts_adjustment = pace_diff * pts_per_pace_point
    
    return {
        'expected_pace': round(game_pace, 1),
        'league_avg_pace': round(league_avg_pace, 1),
        'pace_difference': round(pace_diff, 1),
        'pts_adjustment': round(pts_adjustment, 1),
        'game_style': 'Fast' if game_pace > 102 else 'Slow' if game_pace < 98 else 'Average'
    }

# ===== CLUTCH PERFORMANCE =====

def analyze_clutch_performance(player_data: Dict) -> Dict:
    """
    Analyze player's clutch performance (last 5 minutes, close games)
    """
    
    clutch_pts = player_data.get('CLUTCH_PTS', None)
    
    if clutch_pts is None:
        return {
            'clutch_rating': 'Unknown',
            'confidence': 'Low'
        }
    
    regular_pts = player_data.get('PTS_PG', 20.0)
    clutch_usage = player_data.get('CLUTCH_USG', 25.0)
    
    # Calculate clutch factor (pts per minute in clutch vs regular)
    clutch_factor = clutch_pts / (regular_pts / 48) * 5 if regular_pts > 0 else 1.0
    
    if clutch_factor > 1.15:
        rating = "Elite Closer"
    elif clutch_factor > 1.0:
        rating = "Reliable"
    elif clutch_factor > 0.85:
        rating = "Average"
    else:
        rating = "Struggles in Clutch"
    
    return {
        'clutch_rating': rating,
        'clutch_factor': round(clutch_factor, 2),
        'clutch_usage': round(clutch_usage, 1),
        'confidence': 'High'
    }

# ===== HOME/AWAY SPLITS =====

def analyze_home_away_splits(
    player_data: Dict,
    is_home_game: bool
) -> Dict:
    """
    Analyze home/away performance splits
    """
    
    home_pts = player_data.get('HOME_PTS', None)
    away_pts = player_data.get('AWAY_PTS', None)
    
    if home_pts is None or away_pts is None:
        return {
            'split_impact': 0.0,
            'confidence': 'Low'
        }
    
    diff = home_pts - away_pts
    
    if is_home_game:
        adjustment = diff * 0.5  # 50% of the split difference
    else:
        adjustment = -diff * 0.5
    
    return {
        'home_avg': round(home_pts, 1),
        'away_avg': round(away_pts, 1),
        'split_difference': round(diff, 1),
        'adjustment': round(adjustment, 1),
        'confidence': 'High'
    }

# ===== BETTING RECOMMENDATIONS =====

def generate_comprehensive_recommendation(
    base_prediction: float,
    player_data: Dict,
    game_context: Dict,
    injury_status: str
) -> Dict:
    """
    Generate comprehensive betting recommendation using all factors
    """
    
    # Start with base prediction
    final_prediction = base_prediction
    factors = []
    
    # Apply rest day impact
    rest_days = game_context.get('rest_days', 1)
    rest_mult = calculate_rest_impact(player_data, rest_days)
    final_prediction *= rest_mult
    factors.append(f"Rest ({rest_days}d): {rest_mult:.2f}x")
    
    # Apply pace impact
    pace_impact = analyze_pace_impact(
        player_data,
        game_context.get('home_pace', 100),
        game_context.get('away_pace', 100)
    )
    final_prediction += pace_impact['pts_adjustment']
    factors.append(f"Pace: +{pace_impact['pts_adjustment']:.1f}")
    
    # Apply home/away
    location_impact = analyze_home_away_splits(
        player_data,
        game_context.get('is_home', False)
    )
    final_prediction += location_impact['adjustment']
    factors.append(f"Location: {location_impact['adjustment']:+.1f}")
    
    # Determine confidence
    if injury_status in ['OUT', 'Doubtful']:
        confidence = 'Avoid'
        recommendation = 'DO NOT BET ⛔'
    elif injury_status == 'Questionable':
        confidence = 'Low'
        recommendation = 'WAIT FOR INJURY REPORT ⏳'
    else:
        confidence = 'High'
        
        # Compare to market line
        market_line = game_context.get('market_line', final_prediction)
        edge = final_prediction - market_line
        
        if edge > 2.0:
            recommendation = f'BET OVER {market_line} 💰 (Edge: +{edge:.1f})'
        elif edge < -2.0:
            recommendation = f'BET UNDER {market_line} 💰 (Edge: {edge:.1f})'
        else:
            recommendation = 'PASS - No Edge 👀'
    
    return {
        'final_prediction': round(final_prediction, 1),
        'factors_applied': factors,
        'confidence': confidence,
        'recommendation': recommendation,
        'injury_status': injury_status
    }

# ===== EXAMPLE USAGE =====

if __name__ == "__main__":
    # Example player data
    player = {
        'PLAYER_NAME': 'Luka Doncic',
        'PTS_PG': 33.5,
        'REB_PG': 9.1,
        'AST_PG': 9.8,
        'STL_PG': 1.4,
        'BLK_PG': 0.5,
        'USG_PCT': 36.2,
        'AGE': 24,
        'MIN_PG': 37.0
    }
    
    game_context = {
        'rest_days': 1,
        'home_pace': 101.5,
        'away_pace': 99.2,
        'is_home': True,
        'market_line': 32.5
    }
    
    # Generate recommendation
    rec = generate_comprehensive_recommendation(
        base_prediction=33.5,
        player_data=player,
        game_context=game_context,
        injury_status='Active'
    )
    
    print("="*50)
    print(f"Player: {player['PLAYER_NAME']}")
    print(f"Base Prediction: {33.5}")
    print(f"Final Prediction: {rec['final_prediction']}")
    print(f"Confidence: {rec['confidence']}")
    print(f"Recommendation: {rec['recommendation']}")
    print("\nFactors Applied:")
    for factor in rec['factors_applied']:
        print(f"  • {factor}")
    print("="*50)