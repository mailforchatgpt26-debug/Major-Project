"""
Feature builder that integrates sentiment into GNN node features
Place this at: src/features/sentiment_features.py
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class SentimentFeatureBuilder:
    """
    Builds GNN node features that include sentiment analysis data
    """
    
    def __init__(self, db_connection):
        """
        Initialize feature builder
        
        Args:
            db_connection: Database connection
        """
        self.db = db_connection
        self.feature_names = []
        self.feature_dim = 0
    
    def get_node_features_with_sentiment(self, 
                                        country_code: str,
                                        date: datetime,
                                        include_trade: bool = True,
                                        include_sentiment: bool = True) -> torch.Tensor:
        """
        Get complete node features including sentiment
        
        Args:
            country_code: Country identifier
            date: Reference date for features
            include_trade: Whether to include trade features
            include_sentiment: Whether to include sentiment features
            
        Returns:
            Feature tensor
        """
        features = []
        
        if include_trade:
            trade_features = self.get_trade_features(country_code, date)
            features.append(trade_features)
        
        if include_sentiment:
            sentiment_features = self.get_sentiment_features(country_code, date)
            features.append(sentiment_features)
        
        # Concatenate all feature types
        all_features = np.concatenate(features)
        
        return torch.FloatTensor(all_features)
    
    def get_sentiment_features(self, 
                              country_code: str,
                              date: datetime,
                              lookback_days: int = 30) -> np.ndarray:
        """
        Extract comprehensive sentiment-based features
        
        Args:
            country_code: Country identifier
            date: Reference date
            lookback_days: Number of days to look back
            
        Returns:
            Numpy array of sentiment features
        """
        features = []
        
        # 1. Recent sentiment (last 7 days weighted average)
        recent_sentiment = self._get_weighted_sentiment(
            country_code, date, days_back=7, decay_rate=0.1
        )
        features.append(recent_sentiment)
        
        # 2. Medium-term sentiment (8-30 days)
        medium_sentiment = self._get_avg_sentiment(
            country_code, date - timedelta(days=7), days_back=23
        )
        features.append(medium_sentiment)
        
        # 3. Sentiment trend (recent vs previous period)
        prev_sentiment = self._get_avg_sentiment(
            country_code, date - timedelta(days=14), days_back=7
        )
        sentiment_trend = recent_sentiment - prev_sentiment
        features.append(sentiment_trend)
        
        # 4. Sentiment volatility (standard deviation)
        sentiment_volatility = self._get_sentiment_volatility(
            country_code, date, days_back=lookback_days
        )
        features.append(sentiment_volatility)
        
        # 5. Positive/negative ratio
        pos_neg_ratio = self._get_positive_negative_ratio(
            country_code, date, days_back=7
        )
        features.append(pos_neg_ratio)
        
        # 6. News volume (normalized)
        news_volume = self._get_news_volume(
            country_code, date, days_back=7
        )
        features.append(news_volume)
        
        # 7. Sentiment momentum (rate of change)
        sentiment_momentum = self._calculate_sentiment_momentum(
            country_code, date
        )
        features.append(sentiment_momentum)
        
        # 8. Extreme sentiment indicator (binary: 1 if extreme, 0 otherwise)
        is_extreme = 1.0 if abs(recent_sentiment) > 0.5 else 0.0
        features.append(is_extreme)
        
        return np.array(features, dtype=np.float32)
    
    def _get_weighted_sentiment(self, 
                                country_code: str,
                                date: datetime,
                                days_back: int,
                                decay_rate: float = 0.1) -> float:
        """
        Get sentiment with exponential time decay weighting
        More recent news has more weight
        
        Args:
            country_code: Country identifier
            date: Reference date
            days_back: Number of days to look back
            decay_rate: Exponential decay rate (higher = faster decay)
            
        Returns:
            Weighted average sentiment score
        """
        try:
            cursor = self.db.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                sentiment_score,
                EXTRACT(EPOCH FROM (%s - published_at)) / 86400.0 as days_ago
            FROM news_articles
            WHERE country_code = %s
            AND published_at BETWEEN %s AND %s
            AND sentiment_confidence > 0.5
            ORDER BY published_at DESC
            """
            
            start_date = date - timedelta(days=days_back)
            cursor.execute(query, (date, country_code, start_date, date))
            results = cursor.fetchall()
            
            if not results:
                return 0.0
            
            # Calculate weighted average
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for row in results:
                days_ago = row['days_ago']
                weight = np.exp(-decay_rate * days_ago)  # Exponential decay
                weighted_sum += row['sentiment_score'] * weight
                weight_sum += weight
            
            return float(weighted_sum / weight_sum) if weight_sum > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating weighted sentiment: {e}")
            return 0.0
    
    def _get_avg_sentiment(self,
                          country_code: str,
                          date: datetime,
                          days_back: int) -> float:
        """Get simple average sentiment"""
        try:
            cursor = self.db.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT AVG(sentiment_score) as avg_sentiment
            FROM news_articles
            WHERE country_code = %s
            AND published_at BETWEEN %s AND %s
            AND sentiment_confidence > 0.5
            """
            
            start_date = date - timedelta(days=days_back)
            cursor.execute(query, (country_code, start_date, date))
            result = cursor.fetchone()
            
            return float(result['avg_sentiment']) if result['avg_sentiment'] else 0.0
            
        except Exception as e:
            logger.error(f"Error getting avg sentiment: {e}")
            return 0.0
    
    def _get_sentiment_volatility(self,
                                  country_code: str,
                                  date: datetime,
                                  days_back: int) -> float:
        """Calculate sentiment standard deviation"""
        try:
            cursor = self.db.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT STDDEV(sentiment_score) as sentiment_std
            FROM news_articles
            WHERE country_code = %s
            AND published_at >= %s - INTERVAL '%s days'
            AND sentiment_confidence > 0.5
            """
            
            cursor.execute(query, (country_code, date, days_back))
            result = cursor.fetchone()
            
            return float(result['sentiment_std']) if result['sentiment_std'] else 0.0
            
        except Exception as e:
            logger.error(f"Error getting sentiment volatility: {e}")
            return 0.0
    
    def _get_positive_negative_ratio(self,
                                     country_code: str,
                                     date: datetime,
                                     days_back: int) -> float:
        """Calculate ratio of positive to negative sentiment"""
        try:
            cursor = self.db.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                AVG(sentiment_positive) as avg_pos,
                AVG(sentiment_negative) as avg_neg
            FROM news_articles
            WHERE country_code = %s
            AND published_at BETWEEN %s AND %s
            AND sentiment_confidence > 0.5
            """
            
            start_date = date - timedelta(days=days_back)
            cursor.execute(query, (country_code, start_date, date))
            result = cursor.fetchone()
            
            if result and result['avg_neg'] and result['avg_neg'] > 0:
                return float(result['avg_pos'] / result['avg_neg'])
            return 1.0  # Neutral ratio
            
        except Exception as e:
            logger.error(f"Error getting pos/neg ratio: {e}")
            return 1.0
    
    def _get_news_volume(self,
                        country_code: str,
                        date: datetime,
                        days_back: int) -> float:
        """Get normalized news volume"""
        try:
            cursor = self.db.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT COUNT(*) as news_count
            FROM news_articles
            WHERE country_code = %s
            AND published_at BETWEEN %s AND %s
            """
            
            start_date = date - timedelta(days=days_back)
            cursor.execute(query, (country_code, start_date, date))
            result = cursor.fetchone()
            
            # Normalize: log scale to handle varying volumes
            count = result['news_count'] if result else 0
            return float(np.log1p(count))  # log(1 + count)
            
        except Exception as e:
            logger.error(f"Error getting news volume: {e}")
            return 0.0
    
    def _calculate_sentiment_momentum(self,
                                     country_code: str,
                                     date: datetime) -> float:
        """Calculate rate of sentiment change"""
        try:
            # Get sentiment for last 3 days vs previous 3 days
            recent = self._get_avg_sentiment(country_code, date, 3)
            previous = self._get_avg_sentiment(
                country_code, date - timedelta(days=3), 3
            )
            
            # Momentum is the difference
            return recent - previous
            
        except Exception as e:
            logger.error(f"Error calculating sentiment momentum: {e}")
            return 0.0
    
    def get_trade_features(self,
                          country_code: str,
                          date: datetime) -> np.ndarray:
        """
        Get traditional trade features
        (You should replace this with your existing trade feature extraction)
        
        Args:
            country_code: Country identifier
            date: Reference date
            
        Returns:
            Trade feature array
        """
        try:
            cursor = self.db.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT 
                import_value,
                export_value,
                trade_balance,
                gdp,
                distance
            FROM trade_data
            WHERE country_code = %s
            AND date <= %s
            ORDER BY date DESC
            LIMIT 1
            """
            
            cursor.execute(query, (country_code, date))
            result = cursor.fetchone()
            
            if result:
                # Normalize features
                features = np.array([
                    np.log1p(result['import_value']),
                    np.log1p(result['export_value']),
                    result['trade_balance'] / 1e9,  # Normalize
                    np.log1p(result['gdp']),
                    result['distance'] / 10000  # Normalize distance
                ])
                return features
            else:
                return np.zeros(5, dtype=np.float32)
                
        except Exception as e:
            logger.error(f"Error getting trade features: {e}")
            return np.zeros(5, dtype=np.float32)
    
    def build_graph_features(self,
                            country_codes: List[str],
                            date: datetime) -> torch.Tensor:
        """
        Build feature matrix for all nodes in the graph
        
        Args:
            country_codes: List of country identifiers
            date: Reference date
            
        Returns:
            Feature matrix of shape (num_nodes, num_features)
        """
        features = []
        
        for country_code in country_codes:
            node_features = self.get_node_features_with_sentiment(
                country_code, date
            )
            features.append(node_features)
        
        return torch.stack(features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        return [
            # Sentiment features
            'sentiment_recent_7d',
            'sentiment_medium_23d',
            'sentiment_trend',
            'sentiment_volatility',
            'pos_neg_ratio',
            'news_volume_log',
            'sentiment_momentum',
            'is_extreme_sentiment',
            # Trade features
            'import_value_log',
            'export_value_log',
            'trade_balance_norm',
            'gdp_log',
            'distance_norm'
        ]


# Example usage
if __name__ == "__main__":
    import psycopg2
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    db = psycopg2.connect(
        dbname=os.getenv('DB_NAME', 'gnn_trade'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'password'),
        host=os.getenv('DB_HOST', 'localhost')
    )
    
    builder = SentimentFeatureBuilder(db)
    
    # Test feature extraction
    features = builder.get_node_features_with_sentiment(
        'US', 
        datetime.utcnow()
    )
    
    print(f"Feature shape: {features.shape}")
    print(f"Feature names: {builder.get_feature_names()}")
    print(f"Feature values: {features}")
    
    db.close()