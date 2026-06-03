"""
Automated GDELT article fetcher using the gdeltdoc library.
Runs as a background service, fetching articles periodically.

Usage:
    python src/pipelines/gdelt_article_scheduler.py --once
    python src/pipelines/gdelt_article_scheduler.py --daemon
"""

import sys
from pathlib import Path
import pandas as pd
import time
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import schedule
import requests

try:
    from gdeltdoc import GdeltDoc, Filters
    from gdeltdoc.errors import RateLimitError, BadRequestError, ServerError
    import gdeltdoc.api_client as _gdelt_api_client

    # gdeltdoc calls requests.get with no timeout, which hangs when GDELT is slow.
    # Patch it here so every call gets a (connect=5s, read=25s) budget.
    # Use force-assign (not setdefault) because some environments pre-set a lower value.
    _real_get = _gdelt_api_client.requests.get
    def _get_with_timeout(*args, **kwargs):
        kwargs["timeout"] = (5, 25)
        return _real_get(*args, **kwargs)
    _gdelt_api_client.requests.get = _get_with_timeout

    GDELTDOC_AVAILABLE = True
except ImportError:
    GDELTDOC_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import get_settings
from src.utils.helpers import ensure_directory
from src.data.country_mapping import ISO3_TO_NAME
from src.pipelines.sentiment_analyzer import FinancialSentimentAnalyzer, BilateralSentimentAggregator

logger = get_logger(__name__)
settings = get_settings()

_MAX_FETCH_ATTEMPTS = 2
_RATE_LIMIT_BACKOFF_S = 2   # wait between attempts on 429
_RATE_LIMIT_COOLDOWN_S = 10  # skip re-fetching a pair for this long after exhausted retries
_DIRECT_CONNECT_TIMEOUT_S = 15
_DIRECT_READ_TIMEOUT_S = 60
_DEFAULT_TIMESPAN = "2weeks"  # GDELT DOC API window for live news
_NEWS_LOOKBACK_DAYS = 14          # GDELT live fetch window
_HISTORICAL_LOOKBACK_DAYS = 56    # local CSV corpus (batch refresh can lag)
_MIN_DIRECT_INTERVAL_S = 5.5  # GDELT DOC API: one request per 5 seconds per IP
_MAX_DIRECT_ATTEMPTS = 5


def _format_gdelt_date(raw: Optional[object]) -> str:
    """Normalize GDELT seendate (YYYYMMDD or YYYYMMDDHHMMSS) to YYYY-MM-DD."""
    if raw is None:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    if pd.notna(ts):
        return ts.strftime("%Y-%m-%d")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _lookback_cutoff_date(days: int = _NEWS_LOOKBACK_DAYS) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


def _article_within_lookback(date_value: Optional[object], days: int = _NEWS_LOOKBACK_DAYS) -> bool:
    if date_value is None:
        return False
    ts = pd.to_datetime(str(date_value), errors="coerce", utc=True)
    if pd.isna(ts):
        return False
    return ts.to_pydatetime() >= _lookback_cutoff_date(days)


def _filter_last_n_days(articles: List[Dict], days: int = _NEWS_LOOKBACK_DAYS) -> List[Dict]:
    """Keep articles from the last N days; if none match, return the input list."""
    recent = [a for a in articles if _article_within_lookback(a.get("date"), days=days)]
    return recent if recent else articles


class GDELTArticleFetcher:
    """Fetch GDELT article metadata without BigQuery using the gdeltdoc library."""

    # {cache_key: (timestamp, articles)} — shared across instances
    _cache: Dict[str, Tuple[float, List[Dict]]] = {}
    # {cache_key: timestamp} — tracks when a pair last hit a rate limit ceiling
    _rate_limited: Dict[str, float] = {}
    _direct_api_lock = threading.Lock()
    _fetch_lock = threading.Lock()
    _last_direct_request_at: float = 0.0

    CACHE_TTL = 1800  # 30 minutes

    def __init__(self):
        self.output_dir = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "sentiment"
        ensure_directory(self.output_dir)
        self.articles_file = self.output_dir / "articles.csv"
        self._gd = GdeltDoc() if GDELTDOC_AVAILABLE else None

    def _normalize_articles(self, rows: List[Dict], country1: str, country2: str) -> List[Dict]:
        return [
            {
                "url":            row.get("url", ""),
                "title":          row.get("title", ""),
                "date":           _format_gdelt_date(row.get("seendate") or row.get("date")),
                "domain":         row.get("domain", ""),
                "language":       row.get("language", "en"),
                "country_1_iso3": country1,
                "country_2_iso3": country2,
                "sentiment":      0.0,
                "fetched_at":     datetime.now(timezone.utc).isoformat(),
            }
            for row in rows
        ]

    def _throttle_direct_api(self) -> None:
        elapsed = time.time() - GDELTArticleFetcher._last_direct_request_at
        if elapsed < _MIN_DIRECT_INTERVAL_S:
            time.sleep(_MIN_DIRECT_INTERVAL_S - elapsed)

    def _direct_gdelt_artlist(
        self,
        query: str,
        max_articles: int,
        timespan: str = _DEFAULT_TIMESPAN,
    ) -> List[Dict]:
        """Fetch via GDELT DOC API (works without gdeltdoc / BigQuery)."""
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(min(max_articles, 250)),
            "sort": "datedesc",
            "timespan": timespan,
        }

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_DIRECT_ATTEMPTS):
            try:
                with GDELTArticleFetcher._direct_api_lock:
                    self._throttle_direct_api()
                    with requests.Session() as session:
                        resp = session.request(
                            "GET",
                            "https://api.gdeltproject.org/api/v2/doc/doc",
                            params=params,
                            headers={"User-Agent": "trade-forecast-news-fetcher/1.0"},
                            timeout=_DIRECT_READ_TIMEOUT_S,
                        )
                    GDELTArticleFetcher._last_direct_request_at = time.time()
            except requests.exceptions.Timeout as exc:
                last_error = exc
                logger.warning(
                    f"GDELT request timed out (attempt {attempt + 1}/{_MAX_DIRECT_ATTEMPTS})"
                )
                time.sleep(_MIN_DIRECT_INTERVAL_S * (attempt + 1))
                continue

            body = (resp.text or "").strip()
            if resp.status_code == 429 or "Please limit requests" in body:
                last_error = RuntimeError("GDELT rate limit")
                logger.warning(
                    f"GDELT rate limit hit (attempt {attempt + 1}/{_MAX_DIRECT_ATTEMPTS}), waiting..."
                )
                time.sleep(max(12.0, _MIN_DIRECT_INTERVAL_S * (attempt + 2)))
                continue

            resp.raise_for_status()
            if not body.startswith("{"):
                last_error = ValueError(f"Unexpected GDELT response: {body[:120]}")
                time.sleep(_MIN_DIRECT_INTERVAL_S)
                continue

            payload = resp.json()
            articles = payload.get("articles", [])
            return articles if isinstance(articles, list) else []

        if last_error:
            raise last_error
        return []

    def _fetch_via_direct_api(
        self,
        query: str,
        country1: str,
        country2: str,
        max_articles: int,
        cache_key: str,
        timespan: str = _DEFAULT_TIMESPAN,
    ) -> List[Dict]:
        raw_articles = self._direct_gdelt_artlist(query, max_articles, timespan=timespan)
        articles = _filter_last_n_days(
            self._normalize_articles(raw_articles, country1, country2)
        )
        GDELTArticleFetcher._cache[cache_key] = (time.time(), articles)
        logger.info(
            f"Direct GDELT API returned {len(articles)} articles for {country1}-{country2} "
            f"(timespan={timespan})"
        )
        return articles

    def fetch_articles_for_country_pair(
        self,
        country1: str,
        country2: str,
        max_articles: int = 10,
        timespan: str = _DEFAULT_TIMESPAN,
    ) -> List[Dict]:
        """
        Fetch recent news articles (default: last two weeks) mentioning both countries.
        Uses gdeltdoc when installed; otherwise the public GDELT DOC API.
        """
        cache_key = f"{country1}-{country2}-{max_articles}-{timespan}"

        with GDELTArticleFetcher._fetch_lock:
            return self._fetch_articles_for_country_pair_locked(
                country1, country2, max_articles, timespan, cache_key
            )

    def _fetch_articles_for_country_pair_locked(
        self,
        country1: str,
        country2: str,
        max_articles: int,
        timespan: str,
        cache_key: str,
    ) -> List[Dict]:
        # Skip pairs that recently exhausted rate-limit retries
        backoff_ts = GDELTArticleFetcher._rate_limited.get(cache_key)
        if backoff_ts and time.time() - backoff_ts < _RATE_LIMIT_COOLDOWN_S:
            logger.debug(f"Rate-limit cooldown active for {country1}-{country2}")
            return []

        # Return cached result if still fresh
        cached = GDELTArticleFetcher._cache.get(cache_key)
        if cached:
            ts, articles = cached
            if time.time() - ts < self.CACHE_TTL:
                logger.debug(f"Cache hit for {country1}-{country2}")
                return articles

        c1_name = ISO3_TO_NAME.get(country1, country1)
        c2_name = ISO3_TO_NAME.get(country2, country2)
        # GDELT DOC syntax: parentheses only around OR groups; terms are ANDed by space.
        query = (
            f'"{c1_name}" "{c2_name}" '
            "(pharma OR pharmaceutical OR medicine OR drug) "
            "(trade OR export OR import)"
        )

        if not GDELTDOC_AVAILABLE or self._gd is None:
            try:
                return self._fetch_via_direct_api(
                    query, country1, country2, max_articles, cache_key, timespan=timespan
                )
            except Exception as e:
                logger.warning(
                    f"Direct GDELT API failed for {country1}-{country2}: {type(e).__name__}: {e}"
                )
                return []

        for attempt in range(_MAX_FETCH_ATTEMPTS):
            try:
                f = Filters(timespan=timespan, num_records=min(max_articles, 250), language="English")
                # Insert raw boolean query directly — keyword= wraps the whole string in
                # extra outer quotes, turning it into an exact-phrase match that returns 0 results.
                f.query_params.insert(0, f"{query} ")
                df = self._gd.article_search(f)
                articles = _filter_last_n_days(
                    self._normalize_articles(df.to_dict(orient="records"), country1, country2)
                )

                GDELTArticleFetcher._cache[cache_key] = (time.time(), articles)
                return articles

            except RateLimitError:
                if attempt < _MAX_FETCH_ATTEMPTS - 1:
                    logger.warning(f"Rate limited for {country1}-{country2}, retrying in {_RATE_LIMIT_BACKOFF_S}s...")
                    time.sleep(_RATE_LIMIT_BACKOFF_S)
                else:
                    logger.warning(f"Rate limited after {_MAX_FETCH_ATTEMPTS} attempts for {country1}-{country2}")
                    GDELTArticleFetcher._rate_limited[cache_key] = time.time()
                    return []

            except (BadRequestError, ServerError) as e:
                logger.warning(f"GDELT error for {country1}-{country2}: {type(e).__name__}: {e}")
                return []

            except Exception as e:
                logger.warning(f"gdeltdoc error for {country1}-{country2}: {type(e).__name__}: {e} — trying direct GDELT API")
                try:
                    return self._fetch_via_direct_api(
                        query, country1, country2, max_articles, cache_key, timespan=timespan
                    )
                except Exception as fallback_error:
                    logger.warning(
                        f"Direct GDELT fallback failed for {country1}-{country2}: "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )
                    return []

        return []

    def fetch_general_trade_articles(
        self,
        max_articles: int = 20,
        timespan: str = _DEFAULT_TIMESPAN,
    ) -> List[Dict]:
        """Fetch India trade/pharma news from the last two weeks (no single partner filter)."""
        cache_key = f"IND-GENERAL-{max_articles}-{timespan}"
        with GDELTArticleFetcher._fetch_lock:
            return self._fetch_general_trade_articles_locked(
                max_articles, timespan, cache_key
            )

    def _fetch_general_trade_articles_locked(
        self,
        max_articles: int,
        timespan: str,
        cache_key: str,
    ) -> List[Dict]:
        query = (
            '"India" '
            "(pharma OR pharmaceutical OR medicine OR drug) "
            "(trade OR export OR import)"
        )
        cached = GDELTArticleFetcher._cache.get(cache_key)
        if cached:
            ts, articles = cached
            if time.time() - ts < self.CACHE_TTL:
                logger.debug("Cache hit for IND general trade feed")
                return articles

        if not GDELTDOC_AVAILABLE or self._gd is None:
            try:
                return self._fetch_via_direct_api(
                    query, "IND", "WLD", max_articles, cache_key, timespan=timespan
                )
            except Exception as e:
                logger.warning(f"Direct GDELT API failed for IND general feed: {type(e).__name__}: {e}")
                return []

        for attempt in range(_MAX_FETCH_ATTEMPTS):
            try:
                f = Filters(timespan=timespan, num_records=min(max_articles, 250), language="English")
                f.query_params.insert(
                    0,
                    query
                )
                df = self._gd.article_search(f)
                articles = _filter_last_n_days(
                    self._normalize_articles(df.to_dict(orient="records"), "IND", "WLD")
                )
                GDELTArticleFetcher._cache[cache_key] = (time.time(), articles)
                return articles
            except RateLimitError:
                if attempt < _MAX_FETCH_ATTEMPTS - 1:
                    logger.warning(f"Rate limited for IND general feed, retrying in {_RATE_LIMIT_BACKOFF_S}s...")
                    time.sleep(_RATE_LIMIT_BACKOFF_S)
                else:
                    logger.warning("Rate limited after retries for IND general feed")
                    return []
            except (BadRequestError, ServerError) as e:
                logger.warning(f"GDELT error for IND general feed: {type(e).__name__}: {e}")
                return []
            except Exception as e:
                logger.warning(f"gdeltdoc error for IND general feed: {type(e).__name__}: {e} — trying direct GDELT API")
                try:
                    return self._fetch_via_direct_api(
                        query, "IND", "WLD", max_articles, cache_key, timespan=timespan
                    )
                except Exception as fallback_error:
                    logger.warning(
                        "Direct GDELT fallback failed for IND general feed: "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )
                    return []

        return []

    def get_priority_country_pairs(self) -> List[tuple]:
        # Top 37 India pharma export partners by historical trade value (edges.csv)
        india_partners = [
            "USA", "ZAF", "GBR", "RUS", "NGA", "BRA", "FRA", "KEN",
            "CAN", "AUS", "DEU", "NLD", "PHL", "BEL", "LKA", "TZA",
            "NPL", "MMR", "VNM", "UGA", "GHA", "ETH", "ARE", "THA",
            "MOZ", "UKR", "CHL", "MEX", "ZMB", "ZWE",
            "CHN", "JPN", "KOR", "SGP", "HKG", "IDN", "MYS",
        ]
        return [("IND", p) for p in india_partners]

    def fetch_all_articles(self, max_per_pair: int = 10) -> pd.DataFrame:
        logger.info("Fetching articles for priority country pairs...")
        pairs = self.get_priority_country_pairs()
        all_articles: List[Dict] = []

        for i, (c1, c2) in enumerate(pairs, 1):
            logger.info(f"  [{i}/{len(pairs)}] Fetching {c1}-{c2}...")
            all_articles.extend(self.fetch_articles_for_country_pair(c1, c2, max_per_pair))
            time.sleep(2)

        if not all_articles:
            logger.warning("No articles fetched!")
            return pd.DataFrame()

        df = pd.DataFrame(all_articles)
        logger.info(f"Fetched {len(df):,} total articles")
        return df

    def save_articles(self, df: pd.DataFrame):
        if df.empty:
            return

        df = df.drop_duplicates(subset=["url"], keep="last")

        if self.articles_file.exists():
            try:
                existing = pd.read_csv(self.articles_file)
                combined = (
                    pd.concat([existing, df], ignore_index=True)
                    .drop_duplicates(subset=["url"], keep="last")
                    .sort_values("date", ascending=False)
                    .head(1000)
                )
                combined.to_csv(self.articles_file, index=False)
                logger.info(f"Updated {self.articles_file} ({len(combined):,} total articles)")
            except Exception as e:
                logger.error(f"Failed to merge with existing: {e}")
                df.to_csv(self.articles_file, index=False)
        else:
            df.to_csv(self.articles_file, index=False)
            logger.info(f"Created {self.articles_file} ({len(df):,} articles)")

    def score_and_update_sentiment(self, new_articles: pd.DataFrame) -> None:
        """Run FinBERT on any articles not yet in articles_with_sentiment.csv,
        append results, and regenerate bilateral_sentiment.csv."""
        if new_articles.empty:
            return

        sentiment_file = self.output_dir / "articles_with_sentiment.csv"
        bilateral_file = self.output_dir / "bilateral_sentiment.csv"

        # Load already-scored articles
        if sentiment_file.exists():
            scored = pd.read_csv(sentiment_file)
            already_scored_urls = set(scored["url"].dropna())
        else:
            scored = pd.DataFrame()
            already_scored_urls = set()

        # Find articles that haven't been scored yet
        to_score = new_articles[~new_articles["url"].isin(already_scored_urls)].copy()
        if to_score.empty:
            logger.info("No new articles to score — all URLs already in articles_with_sentiment.csv")
            # Still regenerate bilateral_sentiment.csv from the full scored corpus
            if not scored.empty:
                try:
                    aggregator = BilateralSentimentAggregator()
                    bilateral = aggregator.aggregate_all_pairs(scored)
                    bilateral.to_csv(bilateral_file, index=False)
                    logger.info(f"✓ bilateral_sentiment.csv regenerated ({len(bilateral)} pairs)")
                except Exception as e:
                    logger.error(f"Failed to regenerate bilateral sentiment: {e}")
            return

        logger.info(f"Scoring {len(to_score)} new articles with FinBERT...")
        try:
            analyzer = FinancialSentimentAnalyzer()
            aggregator = BilateralSentimentAggregator()
        except Exception as e:
            logger.error(f"Failed to initialise FinBERT — skipping sentiment update: {e}")
            return

        rows = []
        for _, row in to_score.iterrows():
            title = str(row.get("title", "")).strip()
            try:
                result = analyzer.analyze_text(title, max_length=128)
            except Exception as e:
                logger.warning(f"FinBERT failed on '{title[:60]}': {e}")
                result = {"score": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}

            trade_rel = aggregator.calculate_trade_relevance(title)
            rows.append({
                "url":               row.get("url", ""),
                "title":             title,
                "date":              row.get("date", ""),
                "domain":            row.get("domain", ""),
                "country_1_iso3":    row.get("country_1_iso3", ""),
                "country_2_iso3":    row.get("country_2_iso3", ""),
                "sentiment_score":   result["score"],
                "sentiment_positive":result["positive"],
                "sentiment_negative":result["negative"],
                "sentiment_neutral": result["neutral"],
                "trade_relevance":   trade_rel,
                "fetched_at":        row.get("fetched_at", datetime.now().isoformat()),
            })

        newly_scored = pd.DataFrame(rows)
        logger.info(f"✓ FinBERT scored {len(newly_scored)} articles "
                    f"(avg score: {newly_scored['sentiment_score'].mean():.3f})")

        # Append to articles_with_sentiment.csv
        combined = (
            pd.concat([scored, newly_scored], ignore_index=True)
            .drop_duplicates(subset=["url"], keep="last")
        )
        combined.to_csv(sentiment_file, index=False)
        logger.info(f"✓ articles_with_sentiment.csv updated ({len(combined)} total rows)")

        try:
            import subprocess

            project_root = Path(__file__).resolve().parents[2]
            script = project_root / "scripts" / "sync_mock_news_to_dashboard.py"
            if script.exists():
                subprocess.run(
                    [sys.executable, str(script)],
                    cwd=str(project_root),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                logger.info("✓ Dashboard mock news fallback synced from articles CSV")
        except Exception as e:
            logger.warning(f"Mock news sync skipped: {e}")

        # Regenerate bilateral_sentiment.csv from all scored articles
        try:
            bilateral = aggregator.aggregate_all_pairs(combined)
            bilateral.to_csv(bilateral_file, index=False)
            logger.info(f"✓ bilateral_sentiment.csv regenerated ({len(bilateral)} country pairs)")
        except Exception as e:
            logger.error(f"Failed to regenerate bilateral sentiment: {e}")

    def run_once(self):
        logger.info("=" * 60)
        logger.info("GDELT ARTICLE FETCHER - Single Run")
        logger.info("=" * 60)
        df = self.fetch_all_articles(max_per_pair=10)
        self.save_articles(df)
        self.score_and_update_sentiment(df)
        logger.info("=" * 60)
        logger.info("Article fetch complete!")
        logger.info("=" * 60)


def run_scheduled_fetch():
    fetcher = GDELTArticleFetcher()
    try:
        fetcher.run_once()
    except Exception as e:
        logger.error(f"Scheduled fetch failed: {e}", exc_info=True)


def run_daemon():
    logger.info("=" * 60)
    logger.info("GDELT ARTICLE FETCHER - Daemon Mode (hourly)")
    logger.info("=" * 60)
    schedule.every().hour.do(run_scheduled_fetch)
    run_scheduled_fetch()
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Daemon stopped")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch GDELT articles without BigQuery quota limits")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (fetch every hour)")
    parser.add_argument("--max-per-pair", type=int, default=10, help="Max articles per country pair")
    args = parser.parse_args()

    if args.daemon:
        run_daemon()
    else:
        fetcher = GDELTArticleFetcher()
        df = fetcher.fetch_all_articles(max_per_pair=args.max_per_pair)
        fetcher.save_articles(df)
        fetcher.score_and_update_sentiment(df)
        print(f"\nArticles saved: {len(df):,}\nOutput: {fetcher.articles_file}\n")


if __name__ == "__main__":
    main()
