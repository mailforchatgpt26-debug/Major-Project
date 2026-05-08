"""
Load analyzed articles into PostgreSQL database
Run after sentiment_analyzer.py to populate news_articles table
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.api.postgres_db import PostgresDB

logger = get_logger(__name__)


def load_articles_to_db(csv_path: Path):
    """Load articles from CSV into news_articles table"""
    logger.info(f"Loading articles from {csv_path}")

    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Found {len(df)} articles in CSV")

    # Initialize DB
    db = PostgresDB()
    if not db.enabled:
        logger.error("Database not available")
        return

    # Prepare data for insertion
    articles_data = []
    for _, row in df.iterrows():
        try:
            # Parse published_at
            published_at = pd.to_datetime(row['date']).to_pydatetime()

            articles_data.append((
                row['url'],
                row.get('title', ''),
                row.get('text', ''),
                published_at,
                row['country_code'],
                float(row['sentiment_score']),
                float(row.get('sentiment_confidence', 0.8)),
                row.get('source', 'GDELT')
            ))
        except Exception as e:
            logger.warning(f"Skipping invalid row: {e}")
            continue

    logger.info(f"Prepared {len(articles_data)} articles for insertion")

    # Insert in batches
    batch_size = 1000
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(articles_data), batch_size):
                batch = articles_data[i:i+batch_size]
                try:
                    cur.executemany("""
                        INSERT INTO news_articles
                        (url, title, content, published_at, country_code,
                         sentiment_score, sentiment_confidence, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url) DO UPDATE SET
                            sentiment_score = EXCLUDED.sentiment_score,
                            sentiment_confidence = EXCLUDED.sentiment_confidence,
                            updated_at = CURRENT_TIMESTAMP
                    """, batch)
                    logger.info(f"Inserted batch {i//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                    continue

    logger.info("✅ Articles loaded to database")


def main():
    """Main function"""
    # Path to articles with sentiment
    articles_csv = Path("data/raw/sentiment/articles_with_sentiment.csv")

    if not articles_csv.exists():
        logger.error(f"Articles CSV not found: {articles_csv}")
        logger.info("Run sentiment_analyzer.py first")
        return

    load_articles_to_db(articles_csv)


if __name__ == "__main__":
    main()