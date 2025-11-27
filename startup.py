# utils/load_assets.py  ← put this file in your project

import os
import joblib
import pandas as pd
import requests
from pathlib import Path

# Auto-detect Render disk or local folder
BASE_DIR = Path("/data") if os.path.exists("/data") else Path(".")
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
# REPLACE THESE 5 IDs WITH YOUR REAL GOOGLE DRIVE FILE IDs
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
GDRIVE_IDS = {
    "player_model_2025.pkl":   "1lMJ7z71WUF0r4sWtjAlwClsVvbfeSVfD",
    "team_model_2025.pkl":     "10_jNmyG-buSXxraAhLE5ZiYAhu63qhr3",
    "2025_26_players.csv":     "1wSJJrB7txejOdbzH17GQHJlAqFc1kG1B",
    "2025_26_teams.csv":       "1CLxDviClaHHJbrVGIESHRC8fTENdaitw",
    "injuries.csv":            "1ltr0qPcdsatq_QmcuwgYMD4l7FFZ28SY",
}
# ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

def download_file(filename: str, file_id: str, dest_dir: Path):
    filepath = dest_dir / filename
    if filepath.exists():
        print(f"Found locally: {filename}")
        return
    print(f"Downloading {filename} from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(filepath, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded: {filename}")

# Auto-download any missing files
for filename, file_id in GDRIVE_IDS.items():
    if filename.endswith(".pkl"):
        download_file(filename, file_id, MODELS_DIR)
    else:
        download_file(filename, file_id, DATA_DIR)

# Now load everything — works locally AND on Render
player_model = joblib.load(MODELS_DIR / "player_model_2025_v2.pkl")
team_model   = joblib.load(MODELS_DIR / "team_model_2025_v2.pkl")
df_players   = pd.read_csv(DATA_DIR / "2025_26_players.csv")
df_teams     = pd.read_csv(DATA_DIR / "2025_26_teams.csv")
injuries_df  = pd.read_csv(DATA_DIR / "injuries.csv")

print("All models & data loaded successfully!")