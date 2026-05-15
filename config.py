
"""
Configuration parameters for the AcoustiGuard pipeline.
"""

MODE = "home"

PATHS = {
    "home": {"raw": "data/raw/home", "out": "data/processed/home"},
    "classroom": {"raw": "data/raw/classroom", "out": "data/processed/classroom"},
    "masked": {"raw": "data/raw/classroom_masked", "out": "data/processed/masked"},
    "home_masked": {"raw": "data/raw/home_masked", "out": "data/processed/home_masked"} 
}

TIER_MAP = {
    'Tier1_BottomRow_Modifiers': ['space', 'enter', 'lshift', 'rshift', 'backspace', 'z', 'x', 'c', 'v', 'b', 'n', 'm', 'comma', 'period', 'slash'],
    'Tier2_HomeRow': ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'semicolon', 'apostrophe'],
    'Tier3_TopRow': ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', 'bracketleft', 'bracketright', 'backslash'],
    'Tier4_NumberRow': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'backtick', 'hyphen', 'equals']
}