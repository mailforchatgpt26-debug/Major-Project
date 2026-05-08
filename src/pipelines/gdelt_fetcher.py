"""
Real-time GDELT data fetcher using Google BigQuery.
Fetches sentiment scores and raw article metadata.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False
    print("Warning: google-cloud-bigquery not installed. Install with: pip install google-cloud-bigquery")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.utils.config import get_settings, get_config
from src.utils.helpers import normalize_sentiment, save_dataframe, ensure_directory

logger = get_logger(__name__)
settings = get_settings()
config = get_config()


class GDELTFetcher:
    """
    Fetch GDELT data from BigQuery for real-time sentiment analysis.
    
    Usage:
        fetcher = GDELTFetcher()
        sentiment_df, articles_df = fetcher.fetch_latest(days=7)
    """
    
    def __init__(self):
        """Initialize GDELT fetcher with BigQuery client."""
        self.project_id = settings.GCP_PROJECT_ID
        self.credentials_path = settings.GOOGLE_APPLICATION_CREDENTIALS
        self.output_path = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "sentiment"
        ensure_directory(self.output_path)
        
        # Initialize BigQuery client
        self.client = None
        if BIGQUERY_AVAILABLE:
            self._init_bigquery_client()
    
    def _init_bigquery_client(self):
        """Initialize BigQuery client with credentials."""
        try:
            if self.credentials_path and Path(self.credentials_path).exists():
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=["https://www.googleapis.com/auth/bigquery"]
                )
                self.client = bigquery.Client(
                    credentials=credentials,
                    project=self.project_id
                )
            elif self.project_id:
                # Try default credentials
                self.client = bigquery.Client(project=self.project_id)
            else:
                logger.warning("No GCP credentials configured. GDELT fetching will be disabled.")
                return
            
            logger.info(f"✅ BigQuery client initialized for project: {self.project_id}")
        
        except Exception as e:
            logger.error(f"Failed to initialize BigQuery client: {e}")
            self.client = None
    
    def fetch_sentiment_aggregates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Fetch aggregated sentiment scores from GDELT Events table.
        
        Args:
            start_date: Start date (YYYY-MM-DD) or None for auto
            end_date: End date (YYYY-MM-DD) or None for today
            days_back: Days to look back if start_date not specified
        
        Returns:
            DataFrame with columns: [year, month, country_1_iso3, country_2_iso3, avg_tone, article_count]
        """
        if not self.client:
            logger.error("BigQuery client not initialized")
            return pd.DataFrame()
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching GDELT sentiment from {start_date} to {end_date}")
        
        # SQL Query for sentiment aggregates
        query = f"""
        SELECT
            EXTRACT(YEAR FROM PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS year,
            EXTRACT(MONTH FROM PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING))) AS month,
            Actor1CountryCode AS country_1_iso3,
            Actor2CountryCode AS country_2_iso3,
            AVG(AvgTone) AS avg_tone,
            COUNT(*) AS article_count,
            AVG(GoldsteinScale) AS goldstein_scale_avg
        FROM 
            `gdelt-bq.gdeltv2.events`
        WHERE 
            SQLDATE >= {start_date.replace('-', '')}
            AND SQLDATE <= {end_date.replace('-', '')}
            AND Actor1CountryCode IS NOT NULL
            AND Actor2CountryCode IS NOT NULL
            AND Actor1CountryCode != ''
            AND Actor2CountryCode != ''
            AND AvgTone IS NOT NULL
        GROUP BY 
            year, month, country_1_iso3, country_2_iso3
        ORDER BY 
            year DESC, month DESC
        """
        
        try:
            logger.info("Executing BigQuery query for sentiment aggregates...")
            df = self.client.query(query).to_dataframe()
            
            logger.info(f"Fetched {len(df):,} sentiment aggregates")
            
            # Clean and validate
            df = df.dropna(subset=['country_1_iso3', 'country_2_iso3', 'avg_tone'])
            df = df[df['country_1_iso3'].str.len() == 3]
            df = df[df['country_2_iso3'].str.len() == 3]
            
            logger.info(f"After cleaning: {len(df):,} aggregates")
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch sentiment aggregates: {e}", exc_info=True)
            return pd.DataFrame()
    
    def fetch_news_articles(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days_back: int = 7,
        max_articles: int = 10000
    ) -> pd.DataFrame:
        """
        Fetch raw news article metadata from GDELT GKG table.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            days_back: Days to look back
            max_articles: Maximum articles to fetch
        
        Returns:
            DataFrame with article URLs, tones, themes, persons, organizations
        """
        if not self.client:
            logger.error("BigQuery client not initialized")
            return pd.DataFrame()
        
        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching GDELT articles from {start_date} to {end_date}")
        
        # SQL Query for articles (with LIMIT to avoid huge queries)
        query = f"""
        SELECT
            DocumentIdentifier AS url,
            DATE(PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING))) AS date,
            V2Tone AS tone_scores,
            V2Persons AS persons,
            V2Organizations AS organizations,
            V2Themes AS themes,
            V2Locations AS locations
        FROM 
            `gdelt-bq.gdeltv2.gkg`
        WHERE 
            DATE >= {start_date.replace('-', '')}000000
            AND DATE <= {end_date.replace('-', '')}235959
            AND DocumentIdentifier IS NOT NULL
            AND DocumentIdentifier != ''
            AND V2Tone IS NOT NULL
        LIMIT {max_articles}
        """
        
        try:
            logger.info("Executing BigQuery query for news articles...")
            df = self.client.query(query).to_dataframe()
            
            logger.info(f"Fetched {len(df):,} news articles")
            
            # Parse tone (first value is avg tone)
            if 'tone_scores' in df.columns:
                df['avg_tone'] = df['tone_scores'].apply(
                    lambda x: float(str(x).split(',')[0]) if pd.notna(x) and str(x) else 0.0
                )
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch news articles: {e}", exc_info=True)
            return pd.DataFrame()
    
    def fetch_latest(
        self, 
        days: int = 7,
        fetch_articles: bool = True,
        max_articles: int = 5000
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch latest GDELT data (aggregates + articles).
        
        Args:
            days: Number of days to look back
            fetch_articles: Whether to fetch article metadata
            max_articles: Max articles to fetch
        
        Returns:
            Tuple of (sentiment_df, articles_df)
        """
        logger.info(f"Fetching latest GDELT data ({days} days back)")
        
        # Fetch sentiment aggregates
        sentiment_df = self.fetch_sentiment_aggregates(days_back=days)
        
        # Fetch articles if requested
        articles_df = pd.DataFrame()
        if fetch_articles:
            articles_df = self.fetch_news_articles(
                days_back=days,
                max_articles=max_articles
            )
        
        return sentiment_df, articles_df
    
    def save_to_csv(
        self,
        sentiment_df: pd.DataFrame,
        articles_df: Optional[pd.DataFrame] = None
    ):
        """Save fetched data to CSV files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save sentiment aggregates
        if len(sentiment_df) > 0:
            sentiment_path = self.output_path / f"sentiment_{timestamp}.csv"
            save_dataframe(sentiment_df, sentiment_path)
            logger.info(f"Saved sentiment data to {sentiment_path}")
            
            # Also save as latest
            latest_path = self.output_path / "sentiment.csv"
            save_dataframe(sentiment_df, latest_path)
            logger.info(f"Updated {latest_path}")
        
        # Save articles
        if articles_df is not None and len(articles_df) > 0:
            articles_path = self.output_path / f"articles_{timestamp}.csv"
            save_dataframe(articles_df, articles_path)
            logger.info(f"Saved articles data to {articles_path}")
            
            # Also save as latest
            latest_articles_path = self.output_path / "articles.csv"
            save_dataframe(articles_df, latest_articles_path)
            logger.info(f"Updated {latest_articles_path}")
    
    def update_and_save(self, days: int = 7) -> bool:
        """
        Fetch latest data and save to files.
        
        Args:
            days: Days to look back
        
        Returns:
            True if successful, False otherwise
        """
        try:
            sentiment_df, articles_df = self.fetch_latest(days=days)
            
            if len(sentiment_df) > 0:
                self.save_to_csv(sentiment_df, articles_df)
                logger.info("✅ GDELT data updated successfully")
                return True
            else:
                logger.warning("No GDELT data fetched")
                return False
        
        except Exception as e:
            logger.error(f"Failed to update GDELT data: {e}", exc_info=True)
            return False


def main():
    """Command-line interface for GDELT fetcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch GDELT data from BigQuery')
    parser.add_argument('--days', type=int, default=7, help='Days to look back')
    parser.add_argument('--no-articles', action='store_true', help='Skip article fetching')
    parser.add_argument('--max-articles', type=int, default=5000, help='Max articles to fetch')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("GDELT DATA FETCHER")
    logger.info("="*60)
    
    fetcher = GDELTFetcher()
    
    if not fetcher.client:
        logger.error("❌ Cannot fetch data: BigQuery client not initialized")
        logger.error("Please set up Google Cloud credentials:")
        logger.error("  1. Create a service account in GCP Console")
        logger.error("  2. Download JSON key file")
        logger.error("  3. Set GOOGLE_APPLICATION_CREDENTIALS in .env")
        logger.error("  4. Set GCP_PROJECT_ID in .env")
        return 1
    
    success = fetcher.update_and_save(days=args.days)
    
    if success:
        logger.info("="*60)
        logger.info("✅ GDELT DATA FETCH COMPLETE")
        logger.info("="*60)
        return 0
    else:
        logger.error("❌ GDELT data fetch failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())