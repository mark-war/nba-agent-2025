import unicodedata
from typing import Dict

# Manual overrides for the most problematic or frequently mis-transliterated names
NAME_CORRECTIONS: Dict[str, str] = {
    # Common NBA players with diacritics or Greek/Cyrillic origins
    "Luka Doncic": "Luka Dončić",
    "Nikola Jokic": "Nikola Jokić",
    "Nikola Vucevic": "Nikola Vučević",
    "Dario Saric": "Dario Šarić",
    "Bojan Bogdanovic": "Bojan Bogdanović",
    "Ivica Zubac": "Ivica Zubac",  # usually fine
    "Goran Dragic": "Goran Dragić",
    "Domantas Sabonis": "Domantas Sabonis",  # usually correct
    "Jonas Valanciunas": "Jonas Valančiūnas",
    "Deni Avdija": "Deni Avdija",
    "Aleksej Pokusevski": "Aleksej Pokuševski",
    "Alperen Sengun": "Alperen Şengün",
    "Jose Alvarado": "José Alvarado",
    "Rudy Fernandez": "Rudy Fernández",
    "Nemanja Bjelica": "Nemanja Bjelica",
    "Boban Marjanovic": "Boban Marjanović",
    "Ognjen Dobric": "Ognjen Dobrić",
    "Nikola Jovic": "Nikola Jović",
    "Vasilije Micic": "Vasilije Mičić",
    "Giannis Antetokounmpo": "Giannis Antetokounmpo",  # Greek → Latin standard
    "Thanasis Antetokounmpo": "Thanasis Antetokounmpo",
    # Add more as needed!
}

# Reverse mapping: accented → clean (for lookup)
CLEAN_TO_ORIGINAL = {v: k for k, v in NAME_CORRECTIONS.items()}
ORIGINAL_TO_CLEAN = NAME_CORRECTIONS.copy()


def normalize_name(name: str) -> str:
    """
    Normalize name by:
    - Converting to ASCII (remove/replace diacritics)
    - Lowercase
    - Strip whitespace
    """
    if not name:
        return ""
    
    # First apply manual corrections
    corrected = NAME_CORRECTIONS.get(name.strip().title(), name)
    
    # Then normalize diacritics
    normalized = unicodedata.normalize('NFD', corrected)
    ascii_name = normalized.encode('ascii', 'ignore').decode('ascii')
    
    return ascii_name.strip().title()


def normalize_name_lower(name: str) -> str:
    """For fuzzy matching keys"""
    return normalize_name(name).lower()


# Pre-build lookup dictionaries for speed
NORMALIZED_TO_ORIGINAL: Dict[str, str] = {}
for original, clean in NAME_CORRECTIONS.items():
    norm_key = normalize_name_lower(clean)
    NORMALIZED_TO_ORIGINAL[norm_key] = original

# Also map from original forms
for orig in NAME_CORRECTIONS.values():
    norm_key = normalize_name_lower(orig)
    if norm_key not in NORMALIZED_TO_ORIGINAL:
        NORMALIZED_TO_ORIGINAL[norm_key] = orig


def get_canonical_name(input_name: str) -> str:
    """
    Takes any variation (with/without accents) and returns the official name used in your dataset
    """
    if not input_name:
        return input_name

    key = normalize_name_lower(input_name)
    
    # Direct match in our correction table
    if key in NORMALIZED_TO_ORIGINAL:
        return NORMALIZED_TO_ORIGINAL[key]
    
    # If not found, return normalized version (best effort)
    return normalize_name(input_name)