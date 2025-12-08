# NBA Betting Agent - Quick Reference

## 🚀 Installation (One-Time Setup)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup environment
cp .env.example .env
# Edit .env and add ODDS_API_KEY

# 3. Initial training
python train.py --full

# 4. Get data
python daily_update.py
```

## 🏃 Running the Agent

### Simple (API Only)
```bash
uvicorn main:app --reload
```

### Recommended (API + Auto Updates)
```bash
# Terminal 1: API
uvicorn main:app --reload

# Terminal 2: Scheduler
python daily_update.py --schedule
```

### Docker (Production)
```bash
docker-compose up -d
```

### With Make
```bash
make setup       # First time
make scheduler   # Run with auto-updates
```

## 📡 Essential API Calls

### Player Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_name":"Luka Doncic","opponent_abbr":"LAL"}'
```

### Game Prediction
```bash
curl -X POST http://localhost:8000/predict-game \
  -H "Content-Type: application/json" \
  -d '{"home_team":"LAL","away_team":"BOS","spread_line":-5.5,"total_line":225.5}'
```

### Best Bets
```bash
curl http://localhost:8000/best-bets?min_edge=3.0&max_bets=10
```

### System Health
```bash
curl http://localhost:8000/health
```

## 🔄 Daily Maintenance

### Update Data (Fast - 30 seconds)
```bash
python daily_update.py
# or
make update
```

### Full Retrain (Slow - 5-10 minutes, run weekly)
```bash
python train.py --full
# or
make train
```

### Manual Injury Refresh
```bash
curl -X POST http://localhost:8000/refresh-injuries
```

## 🐳 Docker Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Rebuild
docker-compose build --no-cache
```

## 🔍 Debugging

### Check if API is running
```bash
curl http://localhost:8000/health
```

### View available players
```bash
curl http://localhost:8000/players
```

### Check injuries
```bash
curl http://localhost:8000/injuries
```

### Check cache
```bash
curl http://localhost:8000/cache/stats
```

### View logs
```bash
tail -f logs/nba_agent.log
# or for Docker
docker-compose logs -f api
```

## 📊 Key Files

- `main.py` - API server
- `train.py` - Model training
- `utils.py` - Data fetching
- `daily_update.py` - Automated updates
- `.env` - Configuration
- `data/injuries.csv` - Current injuries
- `data/todays_games.json` - Today's games
- `models/*.pkl` - Trained models

## ⚙️ Configuration (.env)

```bash
ODDS_API_KEY=your_key_here     # Required
LOG_LEVEL=INFO                  # Optional
CACHE_DURATION_MINUTES=30       # Optional
```

## 🎯 Common Tasks

### Find a player
```bash
curl "http://localhost:8000/players?search=Luka"
```

### Check injury status
```bash
curl "http://localhost:8000/injuries?status=OUT"
```

### Get best bets for tomorrow
```bash
curl "http://localhost:8000/best-bets?days_offset=1"
```

### Build a 3-leg parlay
```bash
curl -X POST http://localhost:8000/build-parlay \
  -H "Content-Type: application/json" \
  -d '[
    {"player":"Luka Doncic","stat":"points","line":30.5,"over":true},
    {"player":"LeBron James","stat":"rebounds","line":7.5,"over":true},
    {"player":"Kevin Durant","stat":"assists","line":5.5,"over":true}
  ]'
```

## 🚨 Troubleshooting Quick Fixes

### "Player not found"
```bash
curl "http://localhost:8000/players?search=<partial_name>"
```

### Stale predictions
```bash
curl -X POST http://localhost:8000/refresh-injuries
# or restart
docker-compose restart api
```

### Missing data files
```bash
python train.py --full
python daily_update.py
```

### Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
# or use different port
uvicorn main:app --port 8001
```

## 📈 Performance Tips

1. **Enable caching** - Responses are cached for 30 min
2. **Use scheduler** - Auto-updates keep data fresh
3. **Batch requests** - Use `/best-bets` instead of multiple calls
4. **Docker** - Better resource management
5. **Incremental training** - Daily updates vs full retrain

## 🔗 Important URLs

- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health
- **Injuries**: http://localhost:8000/injuries
- **Players**: http://localhost:8000/players
- **Teams**: http://localhost:8000/teams

## 💡 Pro Tips

1. Run scheduler in background for auto-updates
2. Use Docker for production (handles restarts)
3. Full retrain weekly, incremental daily
4. Cache is your friend - don't disable it
5. Check `/health` before making predictions
6. Use `/players?search=` to find player names
7. Monitor `/cache/stats` to see cache efficiency

## 📝 Make Commands

```bash
make help          # Show all commands
make install       # Install dependencies
make setup         # Complete setup
make run           # Run API
make dev           # Run with auto-reload
make scheduler     # Run with scheduler
make train         # Full retrain
make update        # Update data
make test          # Run tests
make clean         # Clean cache
make docker-up     # Start Docker
make docker-down   # Stop Docker
```

## 🎓 Next Steps

1. ✅ Get API running
2. 📊 Test endpoints with `/docs`
3. 🔄 Enable scheduler
4. 🎯 Build your frontend
5. 🚀 Deploy to production

---

**Need more help?** Check the full README.md or API docs at http://localhost:8000/docs