"""
name_mapper.py - Bulletproof player name handling
Handles accents, typos, variations, and data quality issues
"""

import unicodedata
from typing import Optional

# Common name corrections and variations
NAME_CORRECTIONS = {
    # Accent variations
    "Luka Doncic": "Luka Dončić",
    "Nikola Jokic": "Nikola Jokić",
    "Bogdan Bogdanovic": "Bogdan Bogdanović",
    "Bojan Bogdanovic": "Bojan Bogdanović",
    "Dario Saric": "Dario Šarić",
    "Jusuf Nurkic": "Jusuf Nurkić",
    "Nikola Vucevic": "Nikola Vučević",
    "Kristaps Porzingis": "Kristaps Porziņģis",
    "Jonas Valanciunas": "Jonas Valančiūnas",
    "Davis Bertans": "Dāvis Bertāns",
    "Anzejs Pasecniks": "Anžejs Pasečņiks",
    "Goran Dragic": "Goran Dragić",
    "Nemanja Bjelica": "Nemanja Bjelica",
    "Bogdan Bogdanovic": "Bogdan Bogdanović",
    "Jose Alvarado": "José Alvarado",
    "Sandro Mamukelashvili": "Sandro Mamukelašvili",
    "Ömer Yurtseven": "Ömer Yurtseven",
    "Alperen Sengun": "Alperen Şengün",
    "Furkan Korkmaz": "Furkan Korkmaz",
    
    # Common typos/variations
    "Jayson Tatum": "Jayson Tatum",
    "Jason Tatum": "Jayson Tatum",  # Common typo
    "OG Anunoby": "O.G. Anunoby",
    "Og Anunoby": "O.G. Anunoby",
    "PJ Washington": "P.J. Washington",
    "Pj Washington": "P.J. Washington",
    "RJ Barrett": "R.J. Barrett",
    "Rj Barrett": "R.J. Barrett",
    "AJ Griffin": "A.J. Griffin",
    "Aj Griffin": "A.J. Griffin",
    "TJ McConnell": "T.J. McConnell",
    "Tj Mcconnell": "T.J. McConnell",
    "JJ Redick": "J.J. Redick",
    "Jj Redick": "J.J. Redick",
    
    # Name variations
    "Mo Bamba": "Mohamed Bamba",
    "Dennis Schroder": "Dennis Schröder",
    "Dennis Shroder": "Dennis Schröder",
    "Giannis": "Giannis Antetokounmpo",
    "Greek Freak": "Giannis Antetokounmpo",
    
    # Nickname to full name
    "King James": "LeBron James",
    "The Brow": "Anthony Davis",
    "KD": "Kevin Durant",
    "Steph": "Stephen Curry",
    "CP3": "Chris Paul",
}

def normalize_name(name: str) -> str:
    """
    Normalize player name - handles accents, formatting, and edge cases
    
    Args:
        name: Player name (may be messy)
    
    Returns:
        Normalized name or empty string if invalid
    """
    # Safety checks for None, NaN, non-strings
    if name is None:
        return ""
    
    if not isinstance(name, str):
        # Handle float/int (NaN values from pandas)
        try:
            name = str(name)
            if name.lower() in ['nan', 'none', '']:
                return ""
        except:
            return ""
    
    # Strip whitespace
    name = name.strip()
    
    # Empty check
    if not name or len(name) < 2:
        return ""
    
    # Apply known corrections
    name_title = name.title()
    corrected = NAME_CORRECTIONS.get(name_title, name)
    
    # Remove accents/diacritics (NFD normalization)
    try:
        normalized = ''.join(
            c for c in unicodedata.normalize('NFD', corrected)
            if unicodedata.category(c) != 'Mn'
        )
    except Exception:
        # Fallback if normalization fails
        normalized = corrected
    
    return normalized

def normalize_name_lower(name: str) -> str:
    """
    Normalize and lowercase for case-insensitive matching
    
    Args:
        name: Player name
    
    Returns:
        Normalized lowercase name or empty string
    """
    normalized = normalize_name(name)
    return normalized.lower() if normalized else ""

def get_canonical_name(name: str) -> str:
    """
    Get the canonical (official) name for a player
    Handles typos, accents, and variations
    
    Args:
        name: Any variation of player name
    
    Returns:
        Official canonical name or original if not found
    """
    if not name or not isinstance(name, str):
        return ""
    
    # First, normalize
    normalized = normalize_name(name)
    
    if not normalized:
        return name  # Return original if normalization failed
    
    # Check if we have a correction
    name_title = name.strip().title()
    if name_title in NAME_CORRECTIONS:
        return NAME_CORRECTIONS[name_title]
    
    # Return normalized version
    return normalized

def are_names_similar(name1: str, name2: str, threshold: float = 0.80) -> bool:
    """
    Check if two player names are similar (fuzzy match)
    
    Args:
        name1: First name
        name2: Second name
        threshold: Similarity threshold (0-1)
    
    Returns:
        True if names are similar enough
    """
    if not name1 or not name2:
        return False
    
    # Normalize both
    norm1 = normalize_name_lower(name1)
    norm2 = normalize_name_lower(name2)
    
    if not norm1 or not norm2:
        return False
    
    # Exact match
    if norm1 == norm2:
        return True
    
    # Substring match (one name contains the other)
    if norm1 in norm2 or norm2 in norm1:
        return True
    
    # Fuzzy match using difflib
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    
    return similarity >= threshold

# Validation helper
def is_valid_player_name(name: str) -> bool:
    """
    Check if a player name is valid
    
    Args:
        name: Player name to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not name or not isinstance(name, str):
        return False
    
    name = name.strip()
    
    # Must have at least 2 characters
    if len(name) < 2:
        return False
    
    # Must not be common invalid values
    invalid_values = ['nan', 'none', 'null', 'n/a', '', '--', 'unknown']
    if name.lower() in invalid_values:
        return False
    
    # Must contain at least one letter
    if not any(c.isalpha() for c in name):
        return False
    
    return True

# Helper for batch processing
def normalize_names_batch(names: list) -> list:
    """
    Normalize a list of names efficiently
    
    Args:
        names: List of player names
    
    Returns:
        List of normalized names
    """
    return [normalize_name(name) for name in names if is_valid_player_name(name)]

# Testing helpers
if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Luka Doncic",
        "Luka Dončić",
        "luka doncic",
        "LUKA DONCIC",
        "Jayson Tatum",
        "Jason Tatum",  # Typo
        "Jose Alvarado",
        "José Alvarado",
        "",
        None,
        float('nan'),
        "   ",
        "Giannis",
        "Nikola Jokić",
        "Nikola Jokic",
    ]
    
    print("Testing name_mapper.py\n")
    print("="*60)
    
    for test in test_cases:
        try:
            normalized = normalize_name(test)
            canonical = get_canonical_name(test)
            valid = is_valid_player_name(test)
            
            print(f"Input:      {repr(test)}")
            print(f"Normalized: {repr(normalized)}")
            print(f"Canonical:  {repr(canonical)}")
            print(f"Valid:      {valid}")
            print("-"*60)
        except Exception as e:
            print(f"Input:      {repr(test)}")
            print(f"ERROR:      {e}")
            print("-"*60)