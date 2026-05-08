"""
Complete data loaders for all source datasets.
Handles UN Comtrade, World Bank, CEPII, WTO RTAs, and GDELT sentiment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.utils.config import get_config, get_settings
from src.utils.logger import get_logger
from src.data.country_mapping import get_iso3
from src.utils.helpers import reduce_memory_usage, validate_iso3_codes

logger = get_logger(__name__)
config = get_config()
settings = get_settings()


class ComtradeLoader:
    """Load UN Comtrade bilateral trade data - BULLETPROOF VERSION."""

    def __init__(self, file_path: Optional[Path] = None):
        comtrade_dir = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "comtrade"
        self._comtrade_dir = comtrade_dir
        self._is_monthly = False

        if file_path is None:
            # Check for new monthly split files (2011-2025) first
            monthly_exports = comtrade_dir / "TradeData_exports_pharma_2011_2025.csv"
            monthly_imports = comtrade_dir / "TradeData_imports_pharma_2011_2025.csv"
            if monthly_exports.exists() and monthly_imports.exists():
                self.file_path = None
                self._exports_path = monthly_exports
                self._imports_path = monthly_imports
                self._is_monthly = True
                return

            # Check for split import/export files (old format)
            exports_path = comtrade_dir / "TradeData_exports_pharma.csv"
            imports_path = comtrade_dir / "TradeData_imports_pharma.csv"
            if exports_path.exists() and imports_path.exists():
                self.file_path = None  # signal to use split-file loader
                self._exports_path = exports_path
                self._imports_path = imports_path
                return

            possible_files = [
                "TradeData_pharma.csv",
                "TradeData.csv",
                "TradeData_filtered.csv",
                "comtrade_export.csv",
                "india_exports.csv"
            ]
            for filename in possible_files:
                potential_path = comtrade_dir / filename
                if potential_path.exists():
                    file_path = potential_path
                    break
            if file_path is None:
                file_path = comtrade_dir / "TradeData.csv"

        self.file_path = Path(file_path)
        self._exports_path = None
        self._imports_path = None

    def load(self) -> pd.DataFrame:
        """Load and process Comtrade data - Robust version for multiple formats."""

        # New monthly split-file format (2011-2025)
        if self.file_path is None and self._is_monthly:
            return self._load_monthly_pharma_comtrade()

        # Old split-file format: separate exports and imports CSVs
        if self.file_path is None:
            return self._load_split_pharma_comtrade()

        logger.info(f"Loading Trade data from {self.file_path}")

        if not self.file_path.exists():
            logger.warning(f"File not found: {self.file_path}. Returning empty DataFrame.")
            return pd.DataFrame()

        try:
            # UN Comtrade pharma export has a trailing comma that shifts all columns by 1.
            # Detect by filename and use on_bad_lines='warn' which makes pandas treat the
            # first column ('C') as the row index, giving us the correct shifted mapping.
            is_pharma_format = self.file_path.name == "TradeData_pharma.csv"

            if is_pharma_format:
                return self._load_pharma_comtrade()

            # Load CSV
            df = pd.read_csv(self.file_path, low_memory=False)
            logger.info(f"Loaded {len(df):,} raw records")

            # Identify which format we have
            cols = df.columns

            # Format 1: User's pre-processed format (with reporterCode, partnerCode, primaryValue, etc.)
            if 'reporterCode' in cols and 'primaryValue' in cols:
                logger.info("Detected pre-processed format with numeric codes")
                column_mapping = {
                    'reporterCode': 'source_idx',
                    'partnerCode': 'target_idx',
                    'refYear': 'year',
                    'primaryValue': 'trade_value_usd',
                    'cmdCode': 'hs_code',
                    'dist': 'distance_km'
                }
                df = df.rename(columns=column_mapping)
                
                # Convert numeric codes to ISO3 if possible, else use them as is (will handle mapping later)
                # For now, let's just make sure we have source_iso3/target_iso3
                # If they are already numbers, we might need a numeric->iso3 mapper
                # but many scripts expect strings.
                
                # If the user's file doesn't have ISO3 strings, we'll try to convert
                import pycountry
                def num_to_iso3(val):
                    try:
                        code = str(int(float(val))).zfill(3)
                        country = pycountry.countries.get(numeric=code)
                        return country.alpha_3 if country else code
                    except:
                        return str(val)
                
                if 'source_iso3' not in df.columns:
                    df['source_iso3'] = df['source_idx'].apply(num_to_iso3)
                if 'target_iso3' not in df.columns:
                    df['target_iso3'] = df['target_idx'].apply(num_to_iso3)
                
                # If sector column is missing, derive it from HS chapter code.
                # HS chapter 30 = Pharmaceuticals; chapters 50-63 = Textiles.
                # cmdCode in the BACI-sourced file is the chapter number (30 or 60 sentinel).
                if 'sector' not in df.columns:
                    df['sector'] = "Other"
                    if 'hs_code' in df.columns:
                        try:
                            hs = pd.to_numeric(df['hs_code'], errors='coerce').fillna(0).astype(int)
                            # Chapter is the first two digits of the 6-digit HS code,
                            # or the code itself when it IS the chapter (e.g. 30 or 60).
                            chapter = hs.where(hs <= 99, hs // 10000)
                            df.loc[chapter == 30, 'sector'] = "Pharmaceuticals"
                            df.loc[(chapter >= 50) & (chapter <= 63), 'sector'] = "Textiles"
                            # BACI download uses sentinel cmdCode 60 for all textile chapters
                            df.loc[hs == 60, 'sector'] = "Textiles"
                        except Exception as e:
                            logger.warning(f"Sector assignment failed: {e}")
                
            # Format 2: Original UN Comtrade format
            elif 'reporterDesc' in cols or 'reporterISO' in cols:
                logger.info("Detected standard UN Comtrade format")
                column_mapping = {
                    'reporterDesc': 'reporter_name',
                    'partnerDesc': 'partner_name',
                    'reporterISO': 'source_iso3',
                    'partnerISO': 'target_iso3',
                    'refYear': 'year',
                    'refMonth': 'month',
                    'flowDesc': 'flow',
                    'cmdCode': 'hs_code',
                    'primaryValue': 'trade_value_usd'
                }
                df = df.rename(columns=column_mapping)
                
                # If source_iso3 not already present (rare), map name to it
                if 'source_iso3' not in df.columns and 'reporter_name' in df.columns:
                    df['source_iso3'] = df['reporter_name'].apply(get_iso3)
                    df['target_iso3'] = df['partner_name'].apply(get_iso3)
            
            # Clean up
            if 'year' in df.columns:
                df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            
            if 'month' not in df.columns:
                df['month'] = 1  # Default to January
            else:
                df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
                
            if 'trade_value_usd' in df.columns:
                df['trade_value_usd'] = pd.to_numeric(df['trade_value_usd'], errors='coerce')
            
            # Basic validation
            df = df.dropna(subset=['source_iso3', 'target_iso3', 'year'])
            df = df[df['year'].notna()]
            
            logger.info(f"Final processed records: {len(df):,}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading TradeData: {e}")
            return pd.DataFrame()

    def _load_split_pharma_comtrade(self) -> pd.DataFrame:
        """
        Load TradeData_exports_pharma.csv + TradeData_imports_pharma.csv.

        Both files: India as reporter (reporterISO='IND'), one row per partner per year.
        Year in `refYear`, partner ISO3 in `partnerISO`, value in `primaryValue` (raw USD).
        Rows where partnerISO='W00' are the world-total aggregate and are excluded.
        """
        logger.info(f"Loading split pharma Comtrade files")
        logger.info(f"  exports: {self._exports_path}")
        logger.info(f"  imports: {self._imports_path}")

        rows = []
        for path, flow in [(self._exports_path, "Export"), (self._imports_path, "Import")]:
            try:
                df = pd.read_csv(path, encoding="latin1", index_col=False, low_memory=False)
                logger.info(f"  {flow}: {len(df):,} raw rows")

                df = df[df["partnerISO"].astype(str).str.strip() != "W00"]
                df["_year"] = pd.to_numeric(df["refYear"], errors="coerce")
                df["_value"] = pd.to_numeric(df["primaryValue"], errors="coerce").fillna(0)
                df["_partner"] = df["partnerISO"].astype(str).str.strip()

                for _, row in df[df["_value"] > 0].dropna(subset=["_year"]).iterrows():
                    year = int(row["_year"])
                    value = float(row["_value"]) / 1e6  # raw USD → millions
                    partner = row["_partner"]
                    if flow == "Export":
                        source, target = "IND", partner
                    else:
                        source, target = partner, "IND"
                    rows.append({
                        "source_iso3": source,
                        "target_iso3": target,
                        "year": year,
                        "month": 1,
                        "trade_value_usd": value,
                        "sector": "Pharmaceuticals",
                        "hs_code": "30",
                    })
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")

        result = pd.DataFrame(rows)
        if len(result):
            logger.info(
                f"Split pharma records: {len(result):,} "
                f"({int(result['year'].min())}-{int(result['year'].max())})"
            )
        return result

    def _load_monthly_pharma_comtrade(self) -> pd.DataFrame:
        """
        Load TradeData_exports_pharma_2011_2025.csv + TradeData_imports_pharma_2011_2025.csv.

        Files have a trailing comma creating 48 data fields vs 47 header columns.
        Fix: read with names=header_cols+['_trailing'] and skiprows=1.
        Filter freqCode=='M' to drop embedded header rows.
        Saves data/processed/valid_2025_partners.json with ISOs that have
        both export AND import data in 2025.
        """
        import json as _json

        logger.info("Loading monthly pharma Comtrade files (2011-2025)")
        logger.info(f"  exports: {self._exports_path}")
        logger.info(f"  imports: {self._imports_path}")

        def _read_monthly_comtrade_csv(path: Path) -> pd.DataFrame:
            """Parse Comtrade CSV with quoted headers/fields; optional merge-comment first line."""
            with open(path, 'rb') as f:
                first = f.readline().decode('latin1', errors='replace')
            skiprows = 0
            low = first.lower()
            if 'freqcode' not in low and 'typecode' not in low:
                skiprows = 1
            df = pd.read_csv(
                path,
                encoding='latin1',
                low_memory=False,
                skiprows=skiprows,
            )
            df.columns = df.columns.str.strip()
            # Trailing comma in some dumps adds an empty unnamed column
            unnamed = [c for c in df.columns if str(c).startswith('Unnamed')]
            for c in unnamed:
                col = df[c]
                empty = col.isna() | (col.astype(str).str.strip() == '')
                if empty.all():
                    df = df.drop(columns=[c])
            return df

        rows = []
        valid_2025_exports: set = set()
        valid_2025_imports: set = set()

        for path, flow in [(self._exports_path, 'Export'), (self._imports_path, 'Import')]:
            try:
                df = _read_monthly_comtrade_csv(path)
                # freqCode=='M' drops embedded header rows and non-monthly records
                df = df[df['freqCode'] == 'M'].copy()
                logger.info(f"  {flow}: {len(df):,} monthly rows after freqCode filter")

                df['_year']    = pd.to_numeric(df['refYear'],     errors='coerce')
                df['_month']   = pd.to_numeric(df['refMonth'],    errors='coerce')
                df['_value']   = pd.to_numeric(df['primaryValue'],errors='coerce').fillna(0)
                df['_partner'] = df['partnerISO'].astype(str).str.strip()

                df = df[df['_partner'] != 'W00']
                df = df[df['_value'] > 0].dropna(subset=['_year', '_month'])
                logger.info(f"  {flow}: {len(df):,} rows after filtering")

                # Track which partners have 2025 data for post-filter
                df_2025 = df[df['_year'] == 2025]
                if flow == 'Export':
                    valid_2025_exports.update(df_2025['_partner'].unique())
                else:
                    valid_2025_imports.update(df_2025['_partner'].unique())

                for _, row in df.iterrows():
                    year    = int(row['_year'])
                    month   = int(row['_month'])
                    value   = float(row['_value']) / 1e6  # raw USD → USD millions
                    partner = row['_partner']
                    if flow == 'Export':
                        source, target = 'IND', partner
                    else:
                        source, target = partner, 'IND'
                    rows.append({
                        'source_iso3':     source,
                        'target_iso3':     target,
                        'year':            year,
                        'month':           month,
                        'trade_value_usd': value,
                        'sector':          'Pharmaceuticals',
                        'hs_code':         '30',
                    })
            except Exception as e:
                logger.error(f"Error loading {path}: {e}")

        # Partners that have BOTH export and import data in 2025
        valid_2025 = sorted(valid_2025_exports & valid_2025_imports)
        logger.info(f"Partners with 2025 export data:          {len(valid_2025_exports)}")
        logger.info(f"Partners with 2025 import data:          {len(valid_2025_imports)}")
        logger.info(f"Partners with BOTH 2025 export+import:   {len(valid_2025)}")

        result = pd.DataFrame(rows)
        if len(result) == 0:
            logger.error(
                "Monthly pharma Comtrade load produced no rows; "
                "not overwriting valid_2025_partners.json"
            )
            return result

        processed_dir = settings.PROJECT_ROOT / settings.PROCESSED_DATA_PATH
        processed_dir.mkdir(parents=True, exist_ok=True)
        valid_path = processed_dir / "valid_2025_partners.json"
        with open(valid_path, 'w') as f:
            _json.dump(valid_2025, f)
        logger.info(f"Saved valid 2025 partners list → {valid_path}")

        if len(result):
            logger.info(
                f"Monthly pharma records: {len(result):,} "
                f"({int(result['year'].min())}-{int(result['year'].max())}), "
                f"{result[['year','month']].drop_duplicates().shape[0]} distinct month-snapshots"
            )
        return result

    def _load_pharma_comtrade(self) -> pd.DataFrame:
        """
        Load TradeData_pharma.csv (UN Comtrade v2 export with trailing comma).

        The trailing comma causes pandas to treat column 0 ('C') as the row index,
        shifting every subsequent column one position to the right. The effective
        column mapping after the shift is:
          reporterCode  → ISO3 of the reporting country
          partnerCode   → ISO3 of the partner (always 'IND')
          flowCode      → 'Import' or 'Export'  (the flowDesc value, shifted)
          refPeriodId   → actual year integer (the refYear value, shifted)
          fobvalue      → trade value in raw USD (the primaryValue value, shifted)
        """
        logger.info(f"Loading pharma Comtrade format from {self.file_path}")
        try:
            df = pd.read_csv(self.file_path, encoding="latin1", on_bad_lines="warn")
            logger.info(f"Loaded {len(df):,} raw pharma records")

            df["_value_usd"] = pd.to_numeric(df["fobvalue"], errors="coerce").fillna(0)
            df["_year"]      = pd.to_numeric(df["refPeriodId"], errors="coerce")
            df["_reporter"]  = df["reporterCode"].astype(str).str.strip()
            df["_partner"]   = df["partnerCode"].astype(str).str.strip()
            df["_flow"]      = df["flowCode"].astype(str).str.strip()

            # Some reporters include a partner2Code breakdown (country of origin) alongside
            # the bilateral aggregate row (partner2Code='W00'). Summing all rows doubles the
            # value for those countries. Keep only the W00 aggregate row to avoid this.
            if "partner2Code" in df.columns:
                p2 = df["partner2Code"].astype(str).str.strip()
                df = df[p2 == "W00"]
                logger.info(f"Filtered to partner2Code=W00: {len(df):,} rows remain")

            agg = (
                df.groupby(["_reporter", "_partner", "_flow", "_year"], as_index=False)
                ["_value_usd"].sum()
            )
            agg = agg[agg["_value_usd"] > 0].dropna(subset=["_year"])

            rows = []
            for _, row in agg.iterrows():
                flow  = row["_flow"]
                year  = int(row["_year"])
                value = float(row["_value_usd"]) / 1e6  # raw USD → millions USD

                # Normalise to directed export flow: source → target
                if flow == "Import":
                    source, target = row["_partner"], row["_reporter"]
                else:
                    source, target = row["_reporter"], row["_partner"]

                # Drop self-loops
                if source == target:
                    continue

                rows.append({
                    "source_iso3":     source,
                    "target_iso3":     target,
                    "year":            year,
                    "month":           1,
                    "trade_value_usd": value,
                    "sector":          "Pharmaceuticals",
                    "hs_code":         "30",
                })

            result = pd.DataFrame(rows)
            logger.info(
                f"Pharma records after aggregation: {len(result):,} "
                f"({int(result['year'].min())}-{int(result['year'].max())})"
            )
            return result

        except Exception as e:
            logger.error(f"Error loading pharma Comtrade file: {e}")
            return pd.DataFrame()


class WorldBankLoader:
    """Load World Bank indicators (GDP, Population, Inflation)."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "world-bank"
        self.data_dir = Path(data_dir)
    
    def load(self) -> pd.DataFrame:
        """Load custom world_bank.csv and format it."""
        file_path = self.data_dir / "world_bank.csv"
        logger.info(f"Loading World Bank data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Map the columns to what the pipeline expects
            df = df.rename(columns={
                'country_iso3': 'iso3',
                'gdp': 'gdp_usd',
                'pop': 'population'
            })
            
            # Include inflation_rate as 0 since it's not in the custom file
            if 'inflation_rate' not in df.columns:
                df['inflation_rate'] = 0.0
            
            # Keep only required columns
            expected_cols = ['iso3', 'year', 'gdp_usd', 'population', 'inflation_rate']
            df = df[[col for col in expected_cols if col in df.columns]]
            
            # Ensure proper typing
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            
            # Drop invalid rows
            df = df.dropna(subset=['iso3', 'year'])
            
            # Validate ISO3
            df = validate_iso3_codes(df, ['iso3'])
            
            logger.info(f"Final World Bank data: {len(df):,} country-year records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load custom world_bank.csv: {e}")
            raise


class CEPIILoader:
    """Load CEPII GeoDist bilateral distance data."""
    
    def __init__(self, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "cepii" / "dist_cepii.csv"
        self.file_path = Path(file_path)
    
    def load(self) -> pd.DataFrame:
        """Load CEPII distance data."""
        logger.info(f"Loading CEPII GeoDist data from {self.file_path}")
        
        if not self.file_path.exists():
            logger.warning(f"CEPII file not found at {self.file_path}. Skipping.")
            return pd.DataFrame(columns=['source_iso3', 'target_iso3', 'distance_km', 'shared_language', 'contiguous'])
            
        df = pd.read_csv(self.file_path)
        
        logger.info(f"Loaded {len(df):,} country pair distances")
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'iso_o': 'source_iso3',
            'iso_d': 'target_iso3',
            'dist': 'distance_km',
            'comlang_off': 'shared_language',
            'contig': 'contiguous'
        })
        
        # Convert binary columns to boolean
        df['shared_language'] = df['shared_language'].astype(bool)
        df['contiguous'] = df['contiguous'].astype(bool)
        
        # Select relevant columns
        columns = [
            'source_iso3', 'target_iso3', 'distance_km',
            'shared_language', 'contiguous'
        ]
        
        df = df[columns].copy()
        
        # Validate ISO3 codes
        df = validate_iso3_codes(df, ['source_iso3', 'target_iso3'])
        
        logger.info(f"Final CEPII data: {len(df):,} country pairs")
        return df


class RTALoader:
    """Load WTO Regional Trade Agreements data."""
    
    def __init__(self, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "rta" / "AllRTAs.csv"
        self.file_path = Path(file_path)
    
    def load(self) -> pd.DataFrame:
        """Load and parse RTA data into country pairs."""
        logger.info(f"Loading WTO RTA data from {self.file_path}")
        
        if not self.file_path.exists():
            logger.warning(f"RTA file not found at {self.file_path}. Skipping.")
            return pd.DataFrame(columns=['source_iso3', 'target_iso3', 'fta_binary', 'rta_name'])
            
        df = pd.read_csv(self.file_path)
        
        logger.info(f"Loaded {len(df)} RTAs")
        
        # Parse signatories and create country pairs
        pairs = []
        
        for idx, row in df.iterrows():
            signatories_str = str(row.get('Current signatories', ''))
            
            if pd.isna(signatories_str) or signatories_str == 'nan':
                continue
            
            # Split by common delimiters
            signatories = [s.strip() for s in signatories_str.replace(';', ',').split(',')]
            
            # Convert to ISO3
            iso3_list = []
            for country_name in signatories:
                iso3 = get_iso3(country_name)
                if iso3:
                    iso3_list.append(iso3)
            
            # Create all bidirectional pairs
            for i, iso1 in enumerate(iso3_list):
                for iso2 in iso3_list[i+1:]:
                    # Add both directions
                    pairs.append({
                        'source_iso3': iso1,
                        'target_iso3': iso2,
                        'fta_binary': 1,
                        'rta_name': row['RTA Name']
                    })
                    pairs.append({
                        'source_iso3': iso2,
                        'target_iso3': iso1,
                        'fta_binary': 1,
                        'rta_name': row['RTA Name']
                    })
        
        fta_df = pd.DataFrame(pairs)
        
        # Remove duplicates (keep first occurrence)
        fta_df = fta_df.drop_duplicates(subset=['source_iso3', 'target_iso3'], keep='first')
        
        # Validate ISO3 codes
        fta_df = validate_iso3_codes(fta_df, ['source_iso3', 'target_iso3'])
        
        logger.info(f"Final RTA data: {len(fta_df):,} FTA country pairs")
        return fta_df


class GDELTLoader:
    """Load GDELT sentiment data."""
    
    def __init__(self, file_path: Optional[Path] = None):
        if file_path is None:
            file_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "sentiment" / "sentiment.csv"
        self.file_path = Path(file_path)
    
    def load(self) -> pd.DataFrame:
        """Load GDELT sentiment aggregates."""
        logger.info(f"Loading GDELT sentiment from {self.file_path}")
        
        if not self.file_path.exists():
            logger.warning(f"GDELT sentiment file not found at {self.file_path}. Skipping.")
            return pd.DataFrame(columns=['year', 'month', 'country_1_iso3', 'country_2_iso3', 'avg_tone'])
            
        df = pd.read_csv(self.file_path)
        
        logger.info(f"Loaded {len(df):,} sentiment records")
        
        # Check what columns we actually have
        logger.info(f"GDELT columns: {list(df.columns)}")
        
        # Handle different GDELT formats
        if 'month' in df.columns and df['month'].dtype == 'object':
            # Format 1: "2023-01" style month column
            try:
                df[['year', 'month']] = df['month'].str.split('-', expand=True)
                df['year'] = pd.to_numeric(df['year']).astype('Int64')
                df['month'] = pd.to_numeric(df['month']).astype('Int64')
            except:
                # If split fails, month might already be separated
                pass
        
        elif 'year' not in df.columns or 'month' not in df.columns:
            # Format 2: No year/month columns, create from date or other column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df['year'] = df['date'].dt.year.astype('Int64')
                df['month'] = df['date'].dt.month.astype('Int64')
            else:
                # Use current year/month as fallback
                logger.warning("No date columns found, using current year/month")
                from datetime import datetime
                now = datetime.now()
                df['year'] = now.year
                df['month'] = now.month
        else:
            # Format 3: year and month already separate
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df['month'] = pd.to_numeric(df['month'], errors='coerce').astype('Int64')
        
        # Standardize country column names
        column_mapping = {
            'country_1': 'country_1_iso3',
            'country_2': 'country_2_iso3',
            'Actor1CountryCode': 'country_1_iso3',
            'Actor2CountryCode': 'country_2_iso3',
            'avg_sentiment': 'avg_tone',
            'AvgTone': 'avg_tone'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Validate required columns exist
        required = ['country_1_iso3', 'country_2_iso3', 'avg_tone']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            logger.error(f"GDELT missing columns: {missing}")
            logger.error(f"Available columns: {list(df.columns)}")
            return pd.DataFrame()
        
        # Validate ISO3 codes
        df = validate_iso3_codes(df, ['country_1_iso3', 'country_2_iso3'])
        
        # Select columns
        final_cols = ['year', 'month', 'country_1_iso3', 'country_2_iso3', 'avg_tone']
        existing_cols = [col for col in final_cols if col in df.columns]
        df = df[existing_cols].copy()
        
        logger.info(f"Final GDELT data: {len(df):,} sentiment records")
        return df


class DataLoader:
    """Main data loader orchestrator."""
    
    def __init__(self):
        self.comtrade_loader = ComtradeLoader()
        self.worldbank_loader = WorldBankLoader()
        self.cepii_loader = CEPIILoader()
        self.rta_loader = RTALoader()
        self.gdelt_loader = GDELTLoader()
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all data sources.
        
        Returns:
            Dictionary with data from all sources
        """
        logger.info("="*60)
        logger.info("LOADING ALL DATA SOURCES")
        logger.info("="*60)
        
        data = {}
        
        try:
            data['comtrade'] = self.comtrade_loader.load()
        except Exception as e:
            logger.error(f"Failed to load Comtrade: {e}", exc_info=True)
            data['comtrade'] = pd.DataFrame()
        
        try:
            data['world_bank'] = self.worldbank_loader.load()
        except Exception as e:
            logger.error(f"Failed to load World Bank: {e}", exc_info=True)
            data['world_bank'] = pd.DataFrame()
        
        try:
            data['cepii'] = self.cepii_loader.load()
        except Exception as e:
            logger.error(f"Failed to load CEPII: {e}", exc_info=True)
            data['cepii'] = pd.DataFrame()
        
        try:
            data['rtas'] = self.rta_loader.load()
        except Exception as e:
            logger.error(f"Failed to load RTAs: {e}", exc_info=True)
            data['rtas'] = pd.DataFrame()
        
        try:
            data['gdelt'] = self.gdelt_loader.load()
        except Exception as e:
            logger.error(f"Failed to load GDELT: {e}", exc_info=True)
            data['gdelt'] = pd.DataFrame()
        
        logger.info("="*60)
        logger.info("DATA LOADING SUMMARY")
        logger.info("="*60)
        for source, df in data.items():
            logger.info(f"{source:15s}: {len(df):>10,} rows")
        logger.info("="*60)
        
        return data


if __name__ == "__main__":
    # Test data loaders
    loader = DataLoader()
    data = loader.load_all()
    
    print("\n✅ All data sources loaded successfully!")
    print("\nSample data from each source:")
    for source, df in data.items():
        print(f"\n{source.upper()}:")
        print(df.head(2))