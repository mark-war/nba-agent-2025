#!/usr/bin/env python3
"""
Quick fix for NaN player names error
Run this once to clean your data
"""

import pandas as pd
from pathlib import Path

print("="*60)
print("NBA Agent - Quick Fix for NaN Player Names")
print("="*60)

data_file = Path("data/2025_26_players.csv")

if not data_file.exists():
    print(f"❌ File not found: {data_file}")
    print("Please run: python train.py --full")
    exit(1)

print(f"\n📂 Loading: {data_file}")
df = pd.read_csv(data_file)
print(f"   Original rows: {len(df)}")

# Check for problems
nan_names = df['PLAYER_NAME'].isna().sum()
empty_names = (df['PLAYER_NAME'].astype(str).str.strip() == '').sum()
print(f"\n🔍 Found issues:")
print(f"   NaN names: {nan_names}")
print(f"   Empty names: {empty_names}")

# Clean the data
print(f"\n🧹 Cleaning...")

# Remove NaN player names
df_clean = df[df['PLAYER_NAME'].notna()].copy()
print(f"   After removing NaN: {len(df_clean)} rows")

# Remove empty strings
df_clean = df_clean[df_clean['PLAYER_NAME'].astype(str).str.strip() != ''].copy()
print(f"   After removing empty: {len(df_clean)} rows")

# Remove any remaining junk
df_clean = df_clean[df_clean['PLAYER_NAME'].astype(str).str.len() >= 2].copy()
print(f"   After removing too short: {len(df_clean)} rows")

# Save cleaned data
df_clean.to_csv(data_file, index=False)

print(f"\n✅ Fixed! Cleaned data saved")
print(f"   Removed: {len(df) - len(df_clean)} bad rows")
print(f"   Remaining: {len(df_clean)} valid players")

# Show sample of valid names
print(f"\n📋 Sample of valid player names:")
for name in df_clean['PLAYER_NAME'].head(10):
    print(f"   • {name}")

print("\n" + "="*60)
print("✓ All done! Now run: uvicorn main:app --reload")
print("="*60)