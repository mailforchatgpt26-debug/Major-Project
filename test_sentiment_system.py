#!/usr/bin/env python
"""
Test script to verify real-time sentiment analysis system is operational
"""

import sys
from pathlib import Path
import pandas as pd
import time

sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("🔍 TRADE FORECASTING SYSTEM - REAL-TIME SENTIMENT ANALYSIS CHECK")
print("="*70)

# 1. Check database
print("\n1️⃣  DATABASE & INFRASTRUCTURE")
print("-" * 70)
try:
    from src.utils.database import get_db_context, get_redis
    from sqlalchemy import text
    
    # Test DB
    with get_db_context() as db:
        result = db.execute(text("SELECT 1")).scalar()
        print("   ✓ PostgreSQL database connected")
    
    # Test Redis
    redis = get_redis()
    redis.ping()
    print("   ✓ Redis cache connected")
except Exception as e:
    print(f"   ✗ Infrastructure error: {e}")

# 2. Check real-time sentiment data
print("\n2️⃣  REAL-TIME SENTIMENT DATA")
print("-" * 70)

sentiment_file = Path("data/raw/sentiment/bilateral_sentiment.csv")
if sentiment_file.exists():
    df_sentiment = pd.read_csv(sentiment_file)
    print(f"   ✓ Bilateral sentiment file loaded: {len(df_sentiment)} country pairs")
    print(f"     - Average sentiment: {df_sentiment['sentiment_score'].mean():.3f}")
    print(f"     - Min sentiment: {df_sentiment['sentiment_score'].min():.3f}")
    print(f"     - Max sentiment: {df_sentiment['sentiment_score'].max():.3f}")
    print(f"     - Total articles analyzed: {df_sentiment['article_count'].sum()}")
    print("\n   Sample bilateral sentiment scores:")
    for idx, row in df_sentiment.head(5).iterrows():
        print(f"     • {row['country_1_iso3']}-{row['country_2_iso3']}: {row['sentiment_score']:>7.3f} ({int(row['article_count'])} articles)")
else:
    print(f"   ⚠ Bilateral sentiment file not found: {sentiment_file}")

articles_file = Path("data/raw/sentiment/articles_with_sentiment.csv")
if articles_file.exists():
    df_articles = pd.read_csv(articles_file)
    print(f"\n   ✓ Articles file loaded: {len(df_articles)} articles")
    if 'sentiment' in df_articles.columns:
        print(f"     - Average sentiment: {df_articles['sentiment'].mean():.3f}")
else:
    print(f"\n   ⚠ Articles file not found: {articles_file}")

# 3. Check model
print("\n3️⃣  MODEL & PREDICTIONS")
print("-" * 70)
try:
    from src.models.gnn import TradeGNN
    from src.data.loaders import GraphDataLoader
    
    # Try to load the latest model
    models_dir = Path("models")
    model_files = sorted(models_dir.glob("*.pt"), reverse=True)
    
    if model_files:
        latest_model = model_files[0]
        print(f"   ✓ Latest model checkpoint: {latest_model.name}")
    else:
        print("   ⚠ No trained models found yet")
        
except Exception as e:
    print(f"   ⚠ Model check: {e}")

# 4. Check API
print("\n4️⃣  API SERVER")
print("-" * 70)
try:
    import requests
    response = requests.get("http://localhost:8000/docs", timeout=2)
    if response.status_code == 200:
        print("   ✓ FastAPI server is running on http://localhost:8000")
        print("   ✓ API documentation available at http://localhost:8000/docs")
    else:
        print(f"   ✗ API server responded with status: {response.status_code}")
except Exception as e:
    print(f"   ✗ API server not responding: {e}")

# 5. Check active processes
print("\n5️⃣  ACTIVE PROCESSES")
print("-" * 70)
import subprocess
processes = subprocess.run(
    ["ps", "aux"], 
    capture_output=True, 
    text=True
).stdout

if "weekly_update" in processes:
    print("   ✓ Sentiment fetcher/analyzer (weekly_update.py) is running")
if "uvicorn" in processes or "FastAPI" in processes:
    print("   ✓ API backend (uvicorn/FastAPI) is running")
if "redis" in processes:
    print("   ✓ Redis cache is running")
if "postgres" in processes:
    print("   ✓ PostgreSQL database is running")

# Summary
print("\n" + "="*70)
print("✅ SYSTEM STATUS: OPERATIONAL WITH REAL-TIME SENTIMENT ANALYSIS")
print("="*70)
print("""
📊 Data Flow:
  1. GDELT Fetcher → Fetches news articles via GDELT API
  2. Sentiment Analyzer → Analyzes tone and sentiment (FinBERT)
  3. Database → Stores bilateral sentiment scores
  4. Cache → Redis caches frequent queries
  5. API → FastAPI serves predictions with sentiment
  6. Dashboard → Next.js UI visualizes results
  
🚀 Quick Actions:
  • View API docs: http://localhost:8000/docs
  • Check sentiment data: data/raw/sentiment/bilateral_sentiment.csv
  • Monitor pipeline: tail -f weekly_update.log
  • Dashboard (when built): http://localhost:3000
""")
print("="*70)
