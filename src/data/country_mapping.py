# # src/data/country_mapping.py
# import pandas as pd
# import numpy as np
# import pycountry
# from pathlib import Path
# import warnings
# warnings.filterwarnings('ignore')

# class CountryMapper:
#     def __init__(self):
#         # Manual mappings for common variations
#         def __init__(self):
#             self.manual_map = {
#                 'USA': 'USA', 'United States': 'USA', 'United States of America': 'USA',
#                 'Korea, Rep.': 'KOR', 'South Korea': 'KOR',
#                 'Czech Republic': 'CZE', 'Czechia': 'CZE',
#                 'Russia': 'RUS', 'Russian Federation': 'RUS',
#                 'Iran': 'IRN', 'Iran, Islamic Rep.': 'IRN',
#                 'Venezuela': 'VEN', 'Venezuela, RB': 'VEN',
#                 'Bolivia': 'BOL', 'Vietnam': 'VNM', 'Viet Nam': 'VNM',
#                 'Tanzania': 'TZA', 'Syria': 'SYR', 'Syrian Arab Republic': 'SYR',
#                 'Laos': 'LAO', 'Moldova': 'MDA',
#                 'Egypt, Arab Rep.': 'EGY', 'Egypt': 'EGY',
#                 'Yemen, Rep.': 'YEM', 'Yemen': 'YEM',
#                 'Congo, Dem. Rep.': 'COD', 'Congo, Rep.': 'COG',
#                 'Hong Kong SAR, China': 'HKG', 'Hong Kong': 'HKG',
#                 'Macao SAR, China': 'MAC', 'Macau': 'MAC',
#                 'West Bank and Gaza': 'PSE', 'Palestine': 'PSE',
#                 'Slovak Republic': 'SVK', 'Slovakia': 'SVK',
#                 'Türkiye': 'TUR', 'Turkey': 'TUR',
#             }
            
#     def get_iso3(self, country_name):
#         """Convert country name to ISO3 code"""
#         if pd.isna(country_name):
#             return None
        
#         country_name = str(country_name).strip()
        
#         # Check manual map first
#         if country_name in self.manual_map:
#             return self.manual_map[country_name]
        
#         # Try pycountry lookup
#         try:
#             country = pycountry.countries.search_fuzzy(country_name)[0]
#             return country.alpha_3
#         except:
#             print(f"Warning: Could not map '{country_name}'")
#             return None
    
#     def create_mapping_table(self, country_names):
#         """Create a DataFrame of country name -> ISO3"""
#         unique_names = list(set(country_names))
#         mappings = [(name, self.get_iso3(name)) for name in unique_names]
#         return pd.DataFrame(mappings, columns=['country_name', 'iso3'])

# # Usage
# mapper = CountryMapper()


"""Country name to ISO3 code mapping."""

import pycountry
from typing import Dict, Optional

# Manual mappings for common variations
MANUAL_MAPPINGS = {
    # Basic variations
    "Czech Republic": "CZE",
    "Korea, Rep.": "KOR",
    "Korea, Democratic People's Republic of": "PRK",
    "United States of America": "USA",
    "United States": "USA",
    "United Kingdom": "GBR",
    "Viet Nam": "VNM",
    "Vietnam": "VNM",
    "Russian Federation": "RUS",
    "Russia": "RUS",
    "Iran, Islamic Rep.": "IRN",
    "Egypt, Arab Rep.": "EGY",
    "Venezuela, RB": "VEN",
    "Yemen, Rep.": "YEM",
    "Slovak Republic": "SVK",
    "Slovakia": "SVK",
    "Lao PDR": "LAO",
    "Kyrgyz Republic": "KGZ",
    "Syria": "SYR",
    "Syrian Arab Republic": "SYR",
    "Congo, Dem. Rep.": "COD",
    "Congo, Rep.": "COG",
    "Democratic Republic of the Congo": "COD",
    "Turkiye": "TUR",
    "Turkey": "TUR",
    "Hong Kong SAR, China": "HKG",
    "Hong Kong": "HKG",
    "Macao SAR, China": "MAC",
    "Macau": "MAC",
    
    # Special territories & disputed regions
    "Chinese Taipei": "TWN",
    "Taiwan": "TWN",
    "Kosovo (under UNSC res. 1244)": "XKX",  # Temporary code for Kosovo
    "Faeroe Islands": "FRO",
    "Faroe Islands": "FRO",
    "Falkland Islands (Islas Malvinas)": "FLK",
    "Netherlands Antilles": "ANT",
    "with respect to Aruba": "ABW",
    "British Overseas Territory of Saint Helena": "SHN",
    "Wallis and Futuna Islands": "WLF",
    
    # Economic blocs - Map to largest member or skip
    # Note: These are trading blocs, not countries, so we skip them
    "European Union": None,
    "European Free Trade Association (EFTA)": None,
    "Gulf Cooperation Council (GCC)": None,
    "ASEAN Free Trade Area (AFTA)": None,
    "Southern African Customs Union (SACU)": None,
    "Central American Common Market (CACM)": None,
    "Southern Common Market (MERCOSUR)": None,
    "Eurasian Economic Union (EAEU)": None,
    
    # Numeric codes (from your error)
    "251": None,  # Unknown code
    "490": None,  # Unknown code
    "757": None,  # Unknown code
    "842": None,  # Unknown code

    "CÃ´te d'Ivoire": "CIV",       # Côte d'Ivoire
    "CuraÃ§ao": "CUW",             # Curaçao
    "TÃ¼rkiye": "TUR",             # Türkiye
    "China, Hong Kong SAR": "HKG",
    "China, Macao SAR": "MAC",
    "Lao People's Dem. Rep.": "LAO",

    # ——— Straightforward matches ———
    "Rep. of Korea": "KOR",       # South Korea
    "Dem. People's Rep. of Korea": "PRK",  # North Korea
    "Dem. Rep. of the Congo": "COD",
    "Central African Rep.": "CAF",
    "Dominican Rep.": "DOM",
    "Bolivia (Plurinational State of)": "BOL",
    "United Rep. of Tanzania": "TZA",
    "Bosnia Herzegovina": "BIH",
    "Rep. of Moldova": "MDA",
    "FS Micronesia": "FSM",
    "Marshall Isds": "MHL",
    "Solomon Isds": "SLB",
    "Cook Isds": "COK",
    "Cayman Isds": "CYM",
    "Br. Virgin Isds": "VGB",
    "Turks and Caicos Isds": "TCA",
    "N. Mariana Isds": "MNP",
    "Norfolk Isds": "NFK",
    "Christmas Isds": "CXR",
    "Cocos Isds": "CCK",
    "Faroe Isds": "FRO",
    "Falkland Isds (Malvinas)": "FLK",
    "Wallis and Futuna Isds": "WLF",

    # ——— Special aggregate/non-country regions ———
    # These are NOT countries → Map to None to exclude from graph
    "World": None,
    "Areas, nes": None,
    "Other Asia, nes": None,

    # --- Regional groups / artificial aggregates ---
    "LAIA, nes": None,   # Latin American Integration Association grouping
    "Oceania, nes": None,
    "Other Africa, nes": None,
    "Other Europe, nes": None,
    "North America and Central America, nes": None,

    # --- Zones, customs areas, free trade zones ---
    "Free Zones": None,
    "Special Categories": None,
    "Areas, nes": None,

    # --- Non-country operational codes ---
    "Bunkers": None,             # Fuel bunkers used in shipping
    "Bunkers (fuel)": None,

    # --- Territories with no permanent population ---
    "Fr. South Antarctic Terr.": "ATF",   # French Southern Territories (ISO exists)
    "Br. Indian Ocean Terr.": "IOT",      # British Indian Ocean Territory

    # --- Small territories (already unmapped in your warnings) ---
    "Saint Barthélemy": "BLM",

    # If encoding issues occur:
    "Saint BarthÃ©lemy": "BLM",
    "Fr. South Antarctic Terr": "ATF",

    "Mayotte (Overseas France)": "MYT",
    "Mayotte": "MYT",
}


def get_iso3(country_name: str) -> Optional[str]:
    """
    Convert country name to ISO3 code.
    
    Args:
        country_name: Country name string
    
    Returns:
        ISO3 code or None if not found
    """
    if not country_name or str(country_name).lower() == 'nan':
        return None
    
    country_name = str(country_name).strip()
    
    # Check manual mappings first
    if country_name in MANUAL_MAPPINGS:
        return MANUAL_MAPPINGS[country_name]
    
    # Try pycountry lookup
    try:
        country = pycountry.countries.lookup(country_name)
        return country.alpha_3
    except LookupError:
        pass
    
    # Try fuzzy matching
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        return country.alpha_3
    except:
        print(f"Warning: Could not map country: {country_name}")
        return None


def build_country_mapping_table() -> Dict[str, str]:
    """Build complete mapping table."""
    mapping = MANUAL_MAPPINGS.copy()
    
    for country in pycountry.countries:
        mapping[country.name] = country.alpha_3
        if hasattr(country, 'official_name'):
            mapping[country.official_name] = country.alpha_3
    
    return mapping

# Reverse of MANUAL_MAPPINGS — ISO3 code → display name for API queries.
# Prefer the shortest/most-recognised name when multiple names share a code.
ISO3_TO_NAME: Dict[str, str] = {
    "IND": "India", "USA": "United States", "CHN": "China",
    "ARE": "UAE", "DEU": "Germany", "GBR": "United Kingdom",
    "JPN": "Japan", "SGP": "Singapore", "HKG": "Hong Kong",
    "SAU": "Saudi Arabia", "NLD": "Netherlands", "BEL": "Belgium",
    "FRA": "France", "ITA": "Italy", "KOR": "South Korea",
    "MYS": "Malaysia", "IDN": "Indonesia", "THA": "Thailand",
    "VNM": "Vietnam", "BGD": "Bangladesh", "NPL": "Nepal",
    "RUS": "Russia", "BRA": "Brazil", "ZAF": "South Africa",
    "AUS": "Australia", "CAN": "Canada", "MEX": "Mexico",
    "ESP": "Spain", "POL": "Poland", "TUR": "Turkey",
}
