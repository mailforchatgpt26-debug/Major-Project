"""
Simple news fetcher for real-time sentiment
Fetches articles from RSS feeds and saves to CSV
"""

import sys
from pathlib import Path
import pandas as pd
import requests
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class SimpleNewsFetcher:
    """Fetch news from RSS feeds"""

    def __init__(self):
        self.output_dir = Path("data/raw/sentiment")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.articles_file = self.output_dir / "articles.csv"

        # RSS feeds for trade/economy news
        self.rss_feeds = [
            "https://feeds.bbci.co.uk/news/business/rss.xml",
            "https://feeds.reuters.com/Reuters/businessNews",
            "https://rss.cnn.com/rss/edition_business.rss",
            "https://feeds.npr.org/1006/rss.xml",  # NPR Business
        ]

    def fetch_articles(self, max_articles: int = 50) -> pd.DataFrame:
        """Fetch articles from RSS feeds"""
        all_articles = []

        for feed_url in self.rss_feeds:
            try:
                logger.info(f"Fetching from {feed_url}")
                feed = feedparser.parse(feed_url)

                for entry in feed.entries[:max_articles//len(self.rss_feeds)]:
                    article = {
                        'url': entry.link,
                        'title': entry.title,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'country_code': 'USA',  # Default to USA
                        'source': feed.feed.title if hasattr(feed.feed, 'title') else 'RSS'
                    }
                    all_articles.append(article)

            except Exception as e:
                logger.warning(f"Failed to fetch from {feed_url}: {e}")
                continue

        df = pd.DataFrame(all_articles)
        df = df.drop_duplicates(subset=['url'])
        logger.info(f"Fetched {len(df)} articles")
        return df

    def save_articles(self, df: pd.DataFrame):
        """Save articles to CSV"""
        df.to_csv(self.articles_file, index=False)
        logger.info(f"Saved {len(df)} articles to {self.articles_file}")


def main():
    fetcher = SimpleNewsFetcher()
    df = fetcher.fetch_articles(max_articles=100)
    fetcher.save_articles(df)
    print(f"✅ Fetched and saved {len(df)} articles")


if __name__ == "__main__":
    main()