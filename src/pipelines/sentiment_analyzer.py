"""
Advanced Sentiment Analysis for Trade News
Uses FinBERT for financial sentiment + bilateral context weighting
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class NewsContentExtractor:
    """Extract article content from URLs"""
    
    def __init__(self):
        self.timeout = 10
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def extract_content(self, url: str) -> Dict[str, str]:
        """
        Extract article title and text from URL
        
        Returns:
            {'title': str, 'text': str, 'summary': str}
        """
        try:
            # Try newspaper3k first (best for news articles)
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 100:
                return {
                    'title': article.title or '',
                    'text': article.text,
                    'summary': article.text[:500]  # First 500 chars
                }
        except Exception as e:
            logger.debug(f"Newspaper3k failed for {url}: {e}")
        
        try:
            # Fallback: BeautifulSoup
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Get title
            title = ''
            if soup.title:
                title = soup.title.string or ''
            
            if len(text) > 100:
                return {
                    'title': title,
                    'text': text,
                    'summary': text[:500]
                }
        except Exception as e:
            logger.debug(f"BeautifulSoup failed for {url}: {e}")
        
        return {'title': '', 'text': '', 'summary': ''}


# class FinancialSentimentAnalyzer:
#     """
#     Sentiment analyzer using FinBERT
#     Specialized for financial/economic news
#     """
    
#     def __init__(self, model_name: str = "ProsusAI/finbert"):
#         """
#         Initialize FinBERT model
        
#         Args:
#             model_name: HuggingFace model (default: FinBERT)
#         """
#         logger.info(f"Loading sentiment model: {model_name}")
        
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         logger.info(f"Using device: {self.device}")
        
#         try:
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#             self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
#             self.model.to(self.device)
#             self.model.eval()
#             logger.info("✓ FinBERT model loaded successfully")
#         except Exception as e:
#             logger.error(f"Failed to load FinBERT: {e}")
#             logger.info("Falling back to distilbert-base-uncased-finetuned-sst-2-english")
#             # Fallback to general sentiment model
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 "distilbert-base-uncased-finetuned-sst-2-english"
#             )
#             self.model = AutoModelForSequenceClassification.from_pretrained(
#                 "distilbert-base-uncased-finetuned-sst-2-english"
#             )
#             self.model.to(self.device)
#             self.model.eval()
        
#         # Label mapping (FinBERT: negative, neutral, positive)
#         self.labels = ['negative', 'neutral', 'positive']
    
#     def analyze_text(self, text: str, max_length: int = 512) -> Dict[str, float]:
#         """
#         Analyze sentiment of text
        
#         Args:
#             text: Input text
#             max_length: Max tokens
        
#         Returns:
#             {'negative': float, 'neutral': float, 'positive': float, 'score': float}
#         """
#         if not text or len(text.strip()) < 10:
#             return {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0, 'score': 0.0}
        
#         try:
#             # Tokenize
#             inputs = self.tokenizer(
#                 text,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=max_length,
#                 padding=True
#             ).to(self.device)
            
#             # Get predictions
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
#             # Convert to dict
#             probs = predictions[0].cpu().numpy()
#             result = {label: float(prob) for label, prob in zip(self.labels, probs)}
            
#             # Calculate compound score: -1 (negative) to +1 (positive)
#             result['score'] = (result['positive'] - result['negative'])
            
#             return result
            
#         except Exception as e:
#             logger.error(f"Sentiment analysis failed: {e}")
#             return {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0, 'score': 0.0}
    
#     def analyze_article(self, title: str, text: str) -> Dict[str, float]:
#         """
#         Analyze sentiment of news article
#         Gives more weight to title
        
#         Args:
#             title: Article title
#             text: Article text
        
#         Returns:
#             Sentiment scores
#         """
#         # Analyze title (more important)
#         title_sentiment = self.analyze_text(title, max_length=128)
        
#         # Analyze text (first 512 tokens)
#         text_sentiment = self.analyze_text(text[:2000], max_length=512)
        
#         # Weighted average (title: 0.4, text: 0.6)
#         combined = {
#             'negative': 0.4 * title_sentiment['negative'] + 0.6 * text_sentiment['negative'],
#             'neutral': 0.4 * title_sentiment['neutral'] + 0.6 * text_sentiment['neutral'],
#             'positive': 0.4 * title_sentiment['positive'] + 0.6 * text_sentiment['positive'],
#         }
#         combined['score'] = 0.4 * title_sentiment['score'] + 0.6 * text_sentiment['score']
        
#         return combined
class FinancialSentimentAnalyzer:
    """
    Sentiment analyzer using FinBERT when available,
    falls back to DistilBERT and normalizes output.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        logger.info(f"Loading sentiment model: {model_name}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.is_finbert = True  # track which mode we're in
        
        try:
            # Try FinBERT
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            # FinBERT: 3 labels: negative, neutral, positive
            self.labels = ['negative', 'neutral', 'positive']
            logger.info("✓ FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            logger.info("Falling back to distilbert-base-uncased-finetuned-sst-2-english")
            
            self.is_finbert = False
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.model.to(self.device)
            self.model.eval()
            # We'll derive labels from model.config in analyze_text
            self.labels = None
    
    def analyze_text(self, text: str, max_length: int = 512) -> Dict[str, float]:
        """
        Analyze sentiment of text.

        Returns a dict with keys:
        negative, neutral, positive, score
        """
        if not text or len(text.strip()) < 10:
            return {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0, 'score': 0.0}
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            
            if self.is_finbert:
                # 3-way output: negative / neutral / positive
                result = {label: float(prob) for label, prob in zip(self.labels, probs)}
                # Make sure all keys exist
                for k in ['negative', 'neutral', 'positive']:
                    result.setdefault(k, 0.0)
                result['score'] = result['positive'] - result['negative']
                return result
            else:
                # DistilBERT SST-2 or similar: 2 labels like NEGATIVE / POSITIVE
                id2label = self.model.config.id2label
                labels = [id2label[i].upper() for i in range(len(probs))]
                
                pos_prob = 0.0
                neg_prob = 0.0
                neu_prob = 0.0  # may stay 0 for pure binary models
                
                for lbl, prob in zip(labels, probs):
                    if 'POSITIVE' in lbl:
                        pos_prob += float(prob)
                    elif 'NEGATIVE' in lbl:
                        neg_prob += float(prob)
                    else:
                        neu_prob += float(prob)
                
                total = pos_prob + neg_prob + neu_prob
                if total > 0:
                    pos_prob /= total
                    neg_prob /= total
                    neu_prob /= total
                
                result = {
                    'positive': pos_prob,
                    'negative': neg_prob,
                    'neutral': neu_prob,
                    'score': pos_prob - neg_prob,
                }
                return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0, 'score': 0.0}
    
    def analyze_article(self, title: str, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of news article
        Gives more weight to title.
        """
        title_sentiment = self.analyze_text(title, max_length=128)
        text_sentiment = self.analyze_text(text[:2000], max_length=512)
        
        combined = {
            'negative': 0.4 * title_sentiment['negative'] + 0.6 * text_sentiment['negative'],
            'neutral':  0.4 * title_sentiment['neutral']  + 0.6 * text_sentiment['neutral'],
            'positive': 0.4 * title_sentiment['positive'] + 0.6 * text_sentiment['positive'],
        }
        combined['score'] = 0.4 * title_sentiment['score'] + 0.6 * text_sentiment['score']
        return combined


class BilateralSentimentAggregator:
    """
    Aggregate sentiment scores for country pairs
    Uses trade-specific keyword weighting
    """
    
    def __init__(self):
        # Trade-related keywords for context weighting
        self.trade_keywords = {
            'positive': [
                'trade agreement', 'export growth', 'partnership', 'cooperation',
                'investment', 'deal', 'expansion', 'opportunity', 'boost',
                'increase', 'strengthen', 'collaboration', 'alliance'
            ],
            'negative': [
                'tariff', 'sanction', 'dispute', 'restriction', 'ban',
                'decline', 'reduction', 'conflict', 'war', 'tension',
                'embargo', 'penalty', 'violation', 'investigation'
            ]
        }
    
    def calculate_trade_relevance(self, text: str) -> float:
        """
        Calculate how relevant article is to trade
        
        Returns:
            Relevance score 0-1
        """
        text_lower = text.lower()
        
        # Count trade keywords
        positive_count = sum(1 for kw in self.trade_keywords['positive'] if kw in text_lower)
        negative_count = sum(1 for kw in self.trade_keywords['negative'] if kw in text_lower)
        
        total_keywords = positive_count + negative_count
        
        # Relevance based on keyword density
        if total_keywords > 0:
            return min(1.0, total_keywords / 5.0)  # Normalize
        
        # Check for generic trade terms
        generic_trade = ['trade', 'export', 'import', 'commerce', 'goods']
        generic_count = sum(1 for term in generic_trade if term in text_lower)
        
        return min(0.5, generic_count / 10.0)
    
    def calculate_bilateral_sentiment(
        self,
        articles: pd.DataFrame,
        country_1: str,
        country_2: str
    ) -> Dict[str, float]:
        """
        Calculate aggregated sentiment for country pair
        
        Args:
            articles: DataFrame with sentiment scores
            country_1, country_2: ISO3 codes
        
        Returns:
            {'sentiment_score': float, 'confidence': float, 'article_count': int}
        """
        # Filter to this country pair (bidirectional)
        pair_articles = articles[
            ((articles['country_1_iso3'] == country_1) & (articles['country_2_iso3'] == country_2)) |
            ((articles['country_1_iso3'] == country_2) & (articles['country_2_iso3'] == country_1))
        ].copy()
        
        if len(pair_articles) == 0:
            return {
                'sentiment_score': 0.0,
                'sentiment_positive': 0.0,
                'sentiment_negative': 0.0,
                'sentiment_neutral': 1.0,
                'confidence': 0.0,
                'article_count': 0,
                'trade_relevance': 0.0
            }
        
        # Weight by trade relevance and recency
        weights = []
        for _, row in pair_articles.iterrows():
            relevance = row.get('trade_relevance', 0.5)
            # Time decay: more recent = higher weight
            recency = 1.0  # Can add time-based decay if needed
            weight = relevance * recency
            weights.append(weight)
        
        weights = np.array(weights)
        if weights.sum() == 0:
            weights = np.ones(len(weights))
        weights = weights / weights.sum()
        
        # Weighted average sentiment
        sentiment_score = np.average(pair_articles['sentiment_score'], weights=weights)
        sentiment_pos = np.average(pair_articles['sentiment_positive'], weights=weights)
        sentiment_neg = np.average(pair_articles['sentiment_negative'], weights=weights)
        sentiment_neu = np.average(pair_articles['sentiment_neutral'], weights=weights)
        
        # Confidence based on article count and agreement
        article_count = len(pair_articles)
        sentiment_std = np.std(pair_articles['sentiment_score'])
        
        # High confidence if: many articles + low disagreement
        confidence = min(1.0, (article_count / 10.0) * (1.0 - sentiment_std))
        
        # Average trade relevance
        avg_relevance = pair_articles['trade_relevance'].mean()
        
        return {
            'sentiment_score': float(sentiment_score),
            'sentiment_positive': float(sentiment_pos),
            'sentiment_negative': float(sentiment_neg),
            'sentiment_neutral': float(sentiment_neu),
            'confidence': float(confidence),
            'article_count': int(article_count),
            'trade_relevance': float(avg_relevance)
        }
    
    def aggregate_all_pairs(self, articles: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sentiment for all country pairs
        
        Args:
            articles: DataFrame with analyzed articles
        
        Returns:
            DataFrame with bilateral sentiment scores
        """
        # Get unique country pairs
        pairs = set()
        for _, row in articles.iterrows():
            c1, c2 = row['country_1_iso3'], row['country_2_iso3']
            if pd.notna(c1) and pd.notna(c2):
                # Store in canonical order (alphabetically)
                pairs.add(tuple(sorted([c1, c2])))
        
        logger.info(f"Aggregating sentiment for {len(pairs)} country pairs")
        
        results = []
        for c1, c2 in pairs:
            sentiment = self.calculate_bilateral_sentiment(articles, c1, c2)
            results.append({
                'country_1_iso3': c1,
                'country_2_iso3': c2,
                **sentiment
            })
        
        return pd.DataFrame(results)


class CompleteSentimentPipeline:
    """
    End-to-end sentiment analysis pipeline
    """
    
    def __init__(self):
        self.extractor = NewsContentExtractor()
        self.analyzer = FinancialSentimentAnalyzer()
        self.aggregator = BilateralSentimentAggregator()
        self.output_dir = settings.PROJECT_ROOT / settings.RAW_DATA_PATH / "sentiment"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_articles(
        self,
        articles_df: pd.DataFrame,
        extract_content: bool = True,
        max_articles: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process articles: extract content → analyze sentiment
        
        Args:
            articles_df: DataFrame with article URLs
            extract_content: Whether to fetch article content
            max_articles: Limit number of articles (for testing)
        
        Returns:
            DataFrame with sentiment scores
        """
        if max_articles:
            articles_df = articles_df.head(max_articles)
        
        logger.info(f"Processing {len(articles_df)} articles...")
        
        results = []
        
        for idx, row in articles_df.iterrows():
            try:
                url = row.get('url', '')
                title = row.get('title', '')
                
                # Extract content if needed
                if extract_content and url:
                    logger.debug(f"[{idx+1}/{len(articles_df)}] Extracting: {url[:50]}...")
                    content = self.extractor.extract_content(url)
                    if content['text']:
                        title = content['title'] or title
                        text = content['text']
                    else:
                        text = title  # Fallback to title only
                else:
                    text = title
                
                # Calculate trade relevance
                trade_relevance = self.aggregator.calculate_trade_relevance(title + ' ' + text)
                
                # Analyze sentiment
                sentiment = self.analyzer.analyze_article(title, text)
                
                # Combine with original data
                result = {
                    'url': url,
                    'title': title,
                    'date': row.get('date', ''),
                    'domain': row.get('domain', ''),
                    'country_1_iso3': row.get('country_1_iso3', ''),
                    'country_2_iso3': row.get('country_2_iso3', ''),
                    'sentiment_score': sentiment['score'],
                    'sentiment_positive': sentiment['positive'],
                    'sentiment_negative': sentiment['negative'],
                    'sentiment_neutral': sentiment['neutral'],
                    'trade_relevance': trade_relevance,
                    'fetched_at': row.get('fetched_at', pd.Timestamp.now().isoformat())
                }
                
                results.append(result)
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"  Processed {idx + 1}/{len(articles_df)} articles")
                
            except Exception as e:
                logger.error(f"Error processing article {idx}: {e}")
                continue
        
        result_df = pd.DataFrame(results)
        logger.info(f"✓ Processed {len(result_df)} articles successfully")
        
        return result_df
    
    def run_full_pipeline(
        self,
        articles_csv: Optional[Path] = None,
        extract_content: bool = True,
        save_results: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete sentiment analysis pipeline
        
        Args:
            articles_csv: Path to articles CSV (uses default if None)
            extract_content: Whether to extract article content
            save_results: Whether to save results
        
        Returns:
            (articles_with_sentiment, bilateral_sentiment)
        """
        logger.info("="*60)
        logger.info("🎯 RUNNING SENTIMENT ANALYSIS PIPELINE")
        logger.info("="*60)
        
        # Load articles
        if articles_csv is None:
            articles_csv = self.output_dir / "articles.csv"
        
        if not articles_csv.exists():
            logger.error(f"Articles file not found: {articles_csv}")
            logger.info("Run: python scripts/setup_gdelt_articles.py first")
            return pd.DataFrame(), pd.DataFrame()
        
        logger.info(f"Loading articles from: {articles_csv}")
        articles_df = pd.read_csv(articles_csv)
        logger.info(f"Loaded {len(articles_df)} articles")
        
        # Process articles
        articles_with_sentiment = self.process_articles(
            articles_df,
            extract_content=extract_content
        )
        
        # Aggregate bilateral sentiment
        logger.info("Aggregating bilateral sentiment...")
        bilateral_sentiment = self.aggregator.aggregate_all_pairs(articles_with_sentiment)
        logger.info(f"✓ Created {len(bilateral_sentiment)} bilateral sentiment scores")
        
        # Save results
        if save_results:
            # Save articles with sentiment
            articles_output = self.output_dir / "articles_with_sentiment.csv"
            articles_with_sentiment.to_csv(articles_output, index=False)
            logger.info(f"✓ Saved: {articles_output}")
            
            # Save bilateral sentiment
            bilateral_output = self.output_dir / "bilateral_sentiment.csv"
            bilateral_sentiment.to_csv(bilateral_output, index=False)
            logger.info(f"✓ Saved: {bilateral_output}")
        
        logger.info("="*60)
        logger.info("✅ SENTIMENT ANALYSIS COMPLETE")
        logger.info("="*60)
        
        return articles_with_sentiment, bilateral_sentiment


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze sentiment of trade news')
    parser.add_argument('--no-extract', action='store_true', 
                       help='Skip content extraction (faster, less accurate)')
    parser.add_argument('--max-articles', type=int, 
                       help='Limit number of articles (for testing)')
    
    args = parser.parse_args()
    
    pipeline = CompleteSentimentPipeline()
    articles, bilateral = pipeline.run_full_pipeline(
        extract_content=not args.no_extract
    )
    
    # Print summary
    if len(bilateral) > 0:
        print("\n📊 Sentiment Summary:")
        print(f"  Total country pairs: {len(bilateral)}")
        print(f"  Avg sentiment: {bilateral['sentiment_score'].mean():.3f}")
        print(f"  Avg confidence: {bilateral['confidence'].mean():.3f}")
        print("\n🔝 Most positive:")
        print(bilateral.nlargest(5, 'sentiment_score')[
            ['country_1_iso3', 'country_2_iso3', 'sentiment_score', 'article_count']
        ].to_string(index=False))
        print("\n🔻 Most negative:")
        print(bilateral.nsmallest(5, 'sentiment_score')[
            ['country_1_iso3', 'country_2_iso3', 'sentiment_score', 'article_count']
        ].to_string(index=False))


if __name__ == "__main__":
    main()