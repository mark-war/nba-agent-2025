# 1. Install dependencies
pip install fastapi uvicorn pandas xgboost scikit-learn joblib requests lxml html5lib

# 2. Train models with live data
python train.py

# 3. Start API server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 4. Access API documentation
# Open browser: http://localhost:8000/docs

nba-agent-2025/
├── main.py                      # FastAPI application
├── utils.py                     # Data fetching utilities
├── train.py                     # Model training script
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
├── data/
│   ├── 2025_26_players.csv     # Live player stats
│   ├── 2025_26_teams.csv       # Live team stats
│   ├── injuries.csv            # Current injuries
│   └── todays_games_cache.json # Today's games cache
└── models/
    ├── player_model_2025.pkl   # Trained player model
    ├── team_model_2025.pkl     # Trained team model
    └── training_metadata.json  # Training info


    ------------------------------------------------------------------

    FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

    ------------------------------------------------------------------

docker build -t nba-agent .
docker run -p 8000:8000 -e ODDS_API_KEY=your_key nba-agent