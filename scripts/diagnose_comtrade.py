"""
Diagnostic script to analyze your Comtrade CSV file.
Helps identify why data is being filtered to 0 rows.

Usage:
    python scripts/diagnose_comtrade.py
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_settings

settings = get_settings()


def diagnose_comtrade():
    """Analyze Comtrade data to diagnose issues."""
    
    print("\n" + "="*60)
    print("🔍 COMTRADE DATA DIAGNOSTIC")
    print("="*60)
    
    # Find Comtrade file
    comtrade_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "comtrade" / "TradeData.csv"
    
    if not comtrade_path.exists():
        print(f"\n❌ File not found: {comtrade_path}")
        print("\nPlease ensure your file is at:")
        print(f"  {comtrade_path}")
        return
    
    print(f"\n📁 File found: {comtrade_path}")
    print(f"File size: {comtrade_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load CSV
    print("\n📊 Loading CSV...")
    try:
        df = pd.read_csv(comtrade_path, encoding='latin1', low_memory=False)
        print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return
    
    # Show columns
    print("\n📋 Columns in file:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Check critical columns
    print("\n🔍 Checking critical columns...")
    
    critical_cols = {
        'Flow indicator': ['flowCode', 'flowDesc', 'flow'],
        'HS Code': ['cmdCode', 'hs_code'],
        'Trade value': ['primaryValue', 'trade_value'],
        'Reporter': ['reporterDesc', 'reporterISO', 'reporter_name'],
        'Partner': ['partnerDesc', 'partnerISO', 'partner_name']
    }
    
    found_cols = {}
    for purpose, options in critical_cols.items():
        found = [col for col in options if col in df.columns]
        if found:
            found_cols[purpose] = found[0]
            print(f"  ✅ {purpose}: {found[0]}")
        else:
            print(f"  ❌ {purpose}: NOT FOUND (expected one of: {', '.join(options)})")
    
    # Analyze flow column
    if 'Flow indicator' in found_cols:
        flow_col = found_cols['Flow indicator']
        print(f"\n🔄 Flow distribution ({flow_col}):")
        print(df[flow_col].value_counts().head(10))
        
        # Check for exports
        export_keywords = ['Export', 'X', 'Exports', 'export', 'x']
        export_count = df[df[flow_col].isin(export_keywords)].shape[0]
        print(f"\n  → Exports found: {export_count:,} rows")
        
        if export_count == 0:
            print("  ⚠️  WARNING: No exports found!")
            print("  Check what values are in flowCode/flowDesc")
    
    # Analyze HS codes
    if 'HS Code' in found_cols:
        hs_col = found_cols['HS Code']
        print(f"\n📦 HS Code analysis ({hs_col}):")
        
        # Convert to string for analysis
        df[hs_col] = df[hs_col].astype(str).str.lower()
        
        # Top HS codes
        print("\nTop 20 HS codes/descriptions:")
        print(df[hs_col].value_counts().head(20))
        
        # UPDATED LOGIC: Check for TEXT descriptions, not just numeric codes
        print("\n🔍 Checking if HS codes are TEXT DESCRIPTIONS or NUMERIC CODES...")
        
        # Check if HS codes are text descriptions
        sample_codes = df[hs_col].head(100)
        avg_length = sample_codes.str.len().mean()
        
        if avg_length > 10:  # Likely text descriptions
            print(f"  ✓ Detected TEXT DESCRIPTIONS (avg length: {avg_length:.0f} chars)")
            
            # Check for pharma keywords
            pharma_keywords = ['pharmaceutical', 'pharma', 'medicament', 'medicine', 'drug']
            pharma_codes = df[df[hs_col].str.contains('|'.join(pharma_keywords), na=False, case=False)]
            print(f"\n  → Pharma items (by text): {len(pharma_codes):,} rows")
            
            if len(pharma_codes) > 0:
                print("    Sample pharma descriptions:")
                print(pharma_codes[hs_col].value_counts().head(5))
            
            # Check for textile keywords
            textile_keywords = ['textile', 'fabric', 'apparel', 'clothing', 'silk', 'cotton', 'wool', 'knitted']
            textile_codes = df[df[hs_col].str.contains('|'.join(textile_keywords), na=False, case=False)]
            print(f"\n  → Textile items (by text): {len(textile_codes):,} rows")
            
            if len(textile_codes) > 0:
                print("    Sample textile descriptions:")
                print(textile_codes[hs_col].value_counts().head(5))
            
            # Combined pharma + textiles
            combined = len(pharma_codes) + len(textile_codes)
            print(f"\n  → TOTAL Pharma + Textiles (by text): {combined:,} rows")
            
        else:  # Numeric codes
            print(f"  ✓ Detected NUMERIC CODES (avg length: {avg_length:.0f} chars)")
            
            # Check for pharma codes
            pharma_codes = df[df[hs_col].str.startswith(('30', '3'), na=False)]
            print(f"\n  → Pharma codes (starting with '3' or '30'): {len(pharma_codes):,} rows")
            
            if len(pharma_codes) > 0:
                print("    Sample pharma codes:")
                print(pharma_codes[hs_col].value_counts().head(10))
            
            # Check for textile codes
            textile_prefixes = [str(i) for i in range(50, 64)]
            textile_codes = df[df[hs_col].str.startswith(tuple(textile_prefixes), na=False)]
            print(f"\n  → Textile codes (50-63): {len(textile_codes):,} rows")
            
            if len(textile_codes) > 0:
                print("    Sample textile codes:")
                print(textile_codes[hs_col].value_counts().head(10))
            
            # Combined pharma + textiles
            combined = len(pharma_codes) + len(textile_codes)
            print(f"\n  → TOTAL Pharma + Textiles: {combined:,} rows")
        
        if combined == 0:
            print("  ⚠️  WARNING: No pharma or textile items found!")
            print("  Please check the HS code format in your data.")
    
    # Analyze countries
    if 'Reporter' in found_cols:
        reporter_col = found_cols['Reporter']
        print(f"\n🌍 Reporter countries ({reporter_col}):")
        print(df[reporter_col].value_counts().head(10))
        
        # UPDATED: Check if reporter column has codes or names
        sample_values = df[reporter_col].head(100).unique()
        max_len = df[reporter_col].str.len().max()
        
        if max_len <= 3:
            print(f"\n  ⚠️  WARNING: Reporter column contains CODES (max length: {max_len}), not country names!")
            print("  You should use reporterISO column instead.")
            
            # Check reporterISO column
            if 'reporterISO' in df.columns:
                print(f"\n  ✓ Using reporterISO column:")
                print(df['reporterISO'].value_counts().head(10))
                
                # Check for India in ISO column
                india_rows = df[df['reporterISO'].str.contains('IND', case=False, na=False)]
                print(f"\n  → India (IND) as reporter: {len(india_rows):,} rows")
        else:
            # Check for India in name column
            india_rows = df[df[reporter_col].str.contains('India', case=False, na=False)]
            print(f"\n  → India as reporter: {len(india_rows):,} rows")
            
            if len(india_rows) == 0:
                print("  ⚠️  WARNING: India not found as reporter!")
    
    # Simulate filtering
    print("\n" + "="*60)
    print("🧪 SIMULATING FILTERING PROCESS")
    print("="*60)
    
    steps = [
        ("Initial data", len(df)),
    ]
    
    current_df = df.copy()
    
    # Step 1: Flow filtering
    if 'Flow indicator' in found_cols:
        flow_col = found_cols['Flow indicator']
        export_keywords = ['Export', 'X', 'Exports', 'export', 'x']
        current_df = current_df[current_df[flow_col].isin(export_keywords)]
        steps.append(("After filtering to Exports", len(current_df)))
    
    # Step 2: HS code filtering
    if 'HS Code' in found_cols and len(current_df) > 0:
        hs_col = found_cols['HS Code']
        current_df[hs_col] = current_df[hs_col].astype(str).str.lower()
        
        # UPDATED: Check if text descriptions or numeric codes
        avg_length = current_df[hs_col].str.len().mean()
        
        if avg_length > 10:  # Text descriptions
            print("  (Using TEXT DESCRIPTION matching)")
            pharma_keywords = ['pharmaceutical', 'pharma', 'medicament', 'medicine', 'drug']
            textile_keywords = ['textile', 'fabric', 'apparel', 'clothing', 'silk', 'cotton', 'wool', 'knitted']
            
            pharma = current_df[hs_col].str.contains('|'.join(pharma_keywords), na=False, case=False)
            textiles = current_df[hs_col].str.contains('|'.join(textile_keywords), na=False, case=False)
            
            current_df = current_df[pharma | textiles]
        else:  # Numeric codes
            print("  (Using NUMERIC CODE matching)")
            pharma = current_df[hs_col].str.startswith(('30', '3'), na=False)
            textile_prefixes = tuple([str(i) for i in range(50, 64)])
            textiles = current_df[hs_col].str.startswith(textile_prefixes, na=False)
            
            current_df = current_df[pharma | textiles]
        
        steps.append(("After filtering to Pharma/Textiles", len(current_df)))
    
    # Print steps
    print("\nFiltering steps:")
    for step, count in steps:
        pct = (count / steps[0][1] * 100) if steps[0][1] > 0 else 0
        print(f"  {step:40s}: {count:>10,} rows ({pct:>5.1f}%)")
    
    # Final verdict
    print("\n" + "="*60)
    if steps[-1][1] > 0:
        print(f"✅ SUCCESS: {steps[-1][1]:,} rows would remain after filtering")
        print("\nYour data should work! If preprocessing still fails,")
        print("there might be an issue with:")
        print("  1. Country name mapping (check ISO3 codes)")
        print("  2. Column name mismatches")
        print("  3. Data type conversions")
    else:
        print("❌ PROBLEM: 0 rows remain after filtering")
        print("\nPossible issues:")
        print("  1. Flow column doesn't contain 'Export' or 'X'")
        print("  2. HS codes don't start with expected prefixes")
        print("  3. Data might need different filtering logic")
        print("\n💡 Recommendation:")
        print("  Share sample rows from your CSV so I can fix the loader!")
    print("="*60 + "\n")


if __name__ == "__main__":
    diagnose_comtrade()