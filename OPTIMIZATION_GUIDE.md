# Install dependencies
pip install -r requirements.txt

# Set environment variables
echo "ODDS_API_KEY=your_key_here" > .env

# Initial training
python train.py --full

# Option 1: Run API only
uvicorn main:app --reload

# Option 2: Run with scheduler (recommended)
# Terminal 1:
uvicorn main:app --reload

# Terminal 2:
python daily_update.py --schedule

# Option 3: Docker (production)
docker-compose up -d