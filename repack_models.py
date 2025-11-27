import joblib

# Load original models (these currently break on Render)
player = joblib.load("models/player_model_2025.pkl")
team   = joblib.load("models/team_model_2025.pkl")

# Re-save with protocol 5 for maximum compatibility
joblib.dump(player, "models/player_model_2025_v2.pkl", protocol=5)
joblib.dump(team,   "models/team_model_2025_v2.pkl", protocol=5)

print("✔ Models repacked successfully — upload the files ending in _v2.pkl")
