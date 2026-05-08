# scripts/convert_excel_to_csv.py
import pandas as pd
from pathlib import Path

def convert_cepii():
    xls = pd.ExcelFile('data/raw/cepii/dist_cepii.xls')
    print(f"Available sheets: {xls.sheet_names}")
    
    # Usually the data is in first sheet
    df = pd.read_excel(xls, sheet_name=0)
    df.to_csv('data/raw/cepii/dist_cepii.csv', index=False)
    print(f"Converted CEPII: {len(df)} rows")

def convert_rta():
    df = pd.read_excel('data/raw/rta/AllRTAs.xlsx', sheet_name=0)
    df2 = pd.read_excel('data/raw/rta/AllRTAs.xlsx', sheet_name=1)
    df.to_csv('data/raw/rta/AllRTAs.csv', index=False)
    df2.to_csv('data/raw/rta/RTA-Changes.csv', index=False)
    print(f"Converted AllRTAs: {len(df)} rows")

def convert_comtrade():
    df = pd.read_excel('data/raw/comtrade/TradeData-final.xlsx', sheet_name=0)
    df.to_csv('data/raw/comtrade/TradeData.csv', index=False)
    print(f"Converted Trade data: {len(df)} rows")

if __name__ == "__main__":
    convert_comtrade()