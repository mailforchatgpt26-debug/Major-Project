"""
Script to fetch latest GDELT data from BigQuery.

Usage:
    # Fetch last 7 days
    python scripts/fetch_gdelt_data.py
    
    # Fetch last 30 days
    python scripts/fetch_gdelt_data.py --days 30
    
    # Skip article fetching (faster)
    python scripts/fetch_gdelt_data.py --sentiment-only
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.gdelt_fetcher import GDELTFetcher
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(
    name="gdelt_fetch",
    log_level="INFO",
    log_file="gdelt_fetch.log",
    console=True
)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Fetch GDELT sentiment and news data from BigQuery'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to look back (default: 7)'
    )
    parser.add_argument(
        '--sentiment-only',
        action='store_true',
        help='Only fetch sentiment aggregates, skip articles'
    )
    parser.add_argument(
        '--max-articles',
        type=int,
        default=5000,
        help='Maximum number of articles to fetch (default: 5000)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🌐 GDELT DATA FETCHER")
    print("="*60)
    print(f"Mode: {'Sentiment only' if args.sentiment_only else 'Sentiment + Articles'}")
    print(f"Lookback period: {args.days} days")
    if not args.sentiment_only:
        print(f"Max articles: {args.max_articles}")
    print("="*60 + "\n")
    
    try:
        # Initialize fetcher
        fetcher = GDELTFetcher()
        
        if not fetcher.client:
            print("\n❌ ERROR: BigQuery client not initialized")
            print("\n📝 Setup instructions:")
            print("="*60)
            print("1. Go to Google Cloud Console: https://console.cloud.google.com")
            print("2. Create a new project (or use existing)")
            print("3. Enable BigQuery API")
            print("4. Create a service account:")
            print("   - IAM & Admin → Service Accounts → Create")
            print("   - Grant 'BigQuery Data Viewer' role")
            print("   - Download JSON key file")
            print("5. Update .env file:")
            print("   GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json")
            print("   GCP_PROJECT_ID=your-project-id")
            print("="*60)
            return 1
        
        # Fetch data
        logger.info("Starting GDELT data fetch...")
        
        sentiment_df, articles_df = fetcher.fetch_latest(
            days=args.days,
            fetch_articles=not args.sentiment_only,
            max_articles=args.max_articles
        )
        
        # Save to files
        fetcher.save_to_csv(sentiment_df, articles_df if not args.sentiment_only else None)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ FETCH COMPLETE")
        print("="*60)
        print(f"Sentiment records: {len(sentiment_df):,}")
        if not args.sentiment_only:
            print(f"Article records: {len(articles_df):,}")
        print(f"\n📁 Output location: data/raw/sentiment/")
        print(f"  • sentiment.csv - Aggregated sentiment scores")
        if not args.sentiment_only:
            print(f"  • articles.csv - Raw article metadata")
        print("\n🎯 Next step: Run preprocessing")
        print("  python scripts/preprocess_data.py")
        print("="*60 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ GDELT fetch failed: {e}", exc_info=True)
        print(f"\n❌ ERROR: {e}")
        print("Check logs/gdelt_fetch.log for details")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)