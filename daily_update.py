#!/usr/bin/env python3
"""
Enhanced Daily Update Script with Scheduler
- Runs automatically at specified times
- Can also be run manually
- Supports incremental model updates

Usage:
  python daily_update.py                    # Run once
  python daily_update.py --schedule         # Run continuously with scheduler
  python daily_update.py --retrain-model    # Include model retraining
"""

from utils import fetch_live_injuries, fetch_todays_games_with_odds
import pandas as pd
import json
from datetime import datetime, time as dtime
from pathlib import Path
import logging
import sys
import schedule
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Path("data").mkdir(exist_ok=True)

def update_injuries():
    """Update injury reports"""
    logger.info("="*60)
    logger.info("UPDATING INJURY REPORTS")
    logger.info("="*60)
    
    try:
        injuries = fetch_live_injuries()
        
        if injuries:
            df_injuries = pd.DataFrame(injuries)
            df_injuries.to_csv("data/injuries.csv", index=False)
            logger.info(f"✓ Updated {len(injuries)} injury reports")
            
            # Statistics
            out_players = [inj for inj in injuries if inj.get('status') == 'OUT']
            questionable = [inj for inj in injuries if inj.get('status') == 'Questionable']
            
            logger.info(f"\nKey Updates:")
            logger.info(f"  • OUT: {len(out_players)} players")
            logger.info(f"  • Questionable: {len(questionable)} players")
            
            if out_players:
                logger.info(f"\nPlayers OUT (Top 10):")
                for player in out_players[:10]:
                    logger.info(f"  - {player.get('player_name')} ({player.get('team')}): {player.get('injury_type', 'Unknown')}")
            
            return len(injuries)
        else:
            logger.warning("No injuries fetched - keeping existing data")
            return 0
            
    except Exception as e:
        logger.error(f"Failed to update injuries: {e}")
        return 0

def update_games():
    """Update today's games with odds"""
    logger.info("\n" + "="*60)
    logger.info("UPDATING TODAY'S GAMES")
    logger.info("="*60)
    
    try:
        games = fetch_todays_games_with_odds()
        
        if games:
            # Save to JSON
            with open("data/todays_games.json", "w") as f:
                json.dump({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "timestamp": datetime.now().isoformat(),
                    "games": games
                }, f, indent=2)
            
            logger.info(f"✓ Updated {len(games)} games")
            
            # Display games
            logger.info(f"\nToday's Games:")
            for game in games:
                spread = game.get('spread', 'N/A')
                total = game.get('total', 'N/A')
                time_str = game.get('game_time', 'TBD')
                logger.info(f"  • {game['away_team']} @ {game['home_team']} | {time_str}")
                if spread != 'N/A' and total != 'N/A':
                    logger.info(f"    Spread: {spread} | Total: {total}")
            
            return len(games)
        else:
            logger.warning("No games found for today")
            return 0
            
    except Exception as e:
        logger.error(f"Failed to update games: {e}")
        return 0

def incremental_model_update():
    """Run incremental model update"""
    logger.info("\n" + "="*60)
    logger.info("RUNNING INCREMENTAL MODEL UPDATE")
    logger.info("="*60)
    
    try:
        from train import incremental_train_player_model, train_team_model
        
        # Update player model incrementally
        player_model, mae, r2, features = incremental_train_player_model(
            retrain_from_scratch=False
        )
        
        logger.info(f"✓ Player model updated: MAE={mae:.2f}, R²={r2:.4f}")
        
        # Update team model (less frequent, but good to keep fresh)
        train_team_model()
        logger.info(f"✓ Team model updated")
        
        return True
    except Exception as e:
        logger.error(f"Failed to update models: {e}")
        return False

def daily_update_job(include_model=False):
    """Main daily update job"""
    logger.info("\n" + "="*60)
    logger.info(f"NBA BETTING AGENT - DAILY UPDATE")
    logger.info(f"Running at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    # Update data
    injuries_count = update_injuries()
    games_count = update_games()
    
    # Optional model update
    model_updated = False
    if include_model:
        model_updated = incremental_model_update()
    
    # Update metadata
    metadata = {
        "last_update": datetime.now().isoformat(),
        "injuries_count": injuries_count,
        "games_count": games_count,
        "model_updated": model_updated,
        "duration_seconds": (datetime.now() - start_time).total_seconds(),
        "next_update_recommended": (datetime.now().replace(hour=8, minute=0) + 
                                     pd.Timedelta(days=1)).isoformat()
    }
    
    with open("data/daily_update_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("✓ DAILY UPDATE COMPLETE!")
    logger.info("="*60)
    logger.info(f"\nData updated:")
    logger.info(f"  • {injuries_count} injuries")
    logger.info(f"  • {games_count} games")
    if model_updated:
        logger.info(f"  • Models retrained")
    logger.info(f"\nDuration: {metadata['duration_seconds']:.1f}s")
    logger.info("="*60)
    
    return metadata

def run_scheduler():
    """Run continuous scheduler"""
    logger.info("="*60)
    logger.info("STARTING SCHEDULED UPDATES")
    logger.info("="*60)
    
    # Schedule updates
    schedule.every().day.at("08:00").do(lambda: daily_update_job(include_model=False))
    schedule.every().day.at("12:00").do(update_injuries)  # Midday injury check
    schedule.every().day.at("16:00").do(update_injuries)  # Pre-game injury check
    schedule.every().day.at("02:00").do(lambda: daily_update_job(include_model=True))  # Nightly model update
    
    logger.info("\nScheduled jobs:")
    logger.info("  • 08:00 - Full data update")
    logger.info("  • 12:00 - Injury check")
    logger.info("  • 16:00 - Injury check")
    logger.info("  • 02:00 - Full update + model retraining")
    logger.info("\nScheduler running... (Press Ctrl+C to stop)")
    
    # Run first update immediately
    daily_update_job(include_model=False)
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    if "--schedule" in sys.argv or "-s" in sys.argv:
        # Run continuously with scheduler
        try:
            run_scheduler()
        except KeyboardInterrupt:
            logger.info("\n\nScheduler stopped by user")
    
    elif "--retrain-model" in sys.argv or "-r" in sys.argv:
        # Run once with model retraining
        daily_update_job(include_model=True)
    
    else:
        # Run once without model retraining
        daily_update_job(include_model=False)
        
        logger.info("\nNext steps:")
        logger.info("  1. Restart your API: uvicorn main:app --reload")
        logger.info("  2. Or run with scheduler: python daily_update.py --schedule")
        logger.info("\nRecommend running with scheduler for automated updates")