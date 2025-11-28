# === DELETE startup.py ENTIRELY ===
# === Put this instead near the top of main.py (after imports) ===

from pathlib import Path
import pandas as pd

# Load only what your current app actually uses
DATA_DIR = Path("data")
df_players   = pd.read_csv(DATA_DIR / "2025_26_players.csv")
df_teams     = pd.read_csv(DATA_DIR / "2025_26_teams.csv")
injuries_df  = pd.read_csv(DATA_DIR / "injuries.csv")

# Build injury dictionary (same as before)
INJURY_STATUS = {}
for _, row in injuries_df.iterrows():
    name = str(row['player_name']).strip().title()
    if name and name != "Nan":
        INJURY_STATUS[name] = {
            "status": str(row['status']).strip(),
            "injury_type": str(row['injury_type']).strip(),
            "team": str(row.get('team', 'UNK'))
        }

print(f"Data loaded: {len(df_players)} players, {len(df_teams)} teams, {len(INJURY_STATUS)} injuries")