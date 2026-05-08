"""
PostgreSQL database integration
Handles predictions storage, alerts, and metadata
"""
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from contextlib import contextmanager
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PostgresDB:
    def __init__(self):
        """Initialize PostgreSQL connection"""
        self.config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'trade_forecasting'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres')
        }
        
        try:
            # Test connection
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    logger.info(f"✓ PostgreSQL connected: {version[:50]}...")
            self.enabled = True
        except Exception as e:
            logger.warning(f"PostgreSQL not available: {e}. Running without DB.")
            self.enabled = False
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = psycopg2.connect(**self.config)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def save_predictions(self, predictions: List[Dict], model_version: str, sector: str):
        """Save predictions to database"""
        if not self.enabled:
            return
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get or create model version
                    cur.execute("""
                        SELECT id FROM model_versions 
                        WHERE version = %s
                    """, (model_version,))
                    
                    result = cur.fetchone()
                    if result:
                        model_id = result[0]
                    else:
                        # Create new model version
                        cur.execute("""
                            INSERT INTO model_versions (version, model_type, is_active, file_path)
                            VALUES (%s, 'GAT', TRUE, %s)
                            RETURNING id
                        """, (model_version, f"models/{model_version}.pt"))
                        model_id = cur.fetchone()[0]
                    
                    # Get country and product IDs
                    # (This assumes countries and products are already in DB)
                    
                    # Prepare prediction data
                    prediction_data = []
                    for pred in predictions:
                        prediction_data.append((
                            model_id,
                            1,  # India country_id (assuming)
                            pred.get('target_country_id'),
                            pred.get('product_id'),
                            datetime.now().date(),
                            pred.get('target_month'),
                            pred.get('target_year'),
                            pred.get('predicted_value_log'),
                            pred.get('predicted_value_usd'),
                            pred.get('confidence_score', 0.75),
                            pred.get('change_pct', 0)
                        ))
                    
                    # Bulk insert with conflict handling
                    execute_values(cur, """
                        INSERT INTO predictions (
                            model_version_id, source_country_id, target_country_id,
                            product_id, prediction_date, target_month, target_year,
                            predicted_value_log, predicted_value_usd,
                            confidence_score, prediction_change_pct
                        ) VALUES %s
                        ON CONFLICT (model_version_id, source_country_id, target_country_id, product_id, target_year, target_month)
                        DO UPDATE SET
                            predicted_value_log = EXCLUDED.predicted_value_log,
                            predicted_value_usd = EXCLUDED.predicted_value_usd,
                            confidence_score = EXCLUDED.confidence_score,
                            prediction_change_pct = EXCLUDED.prediction_change_pct,
                            created_at = CURRENT_TIMESTAMP
                    """, prediction_data)
                    
                    logger.info(f"Saved {len(predictions)} predictions to PostgreSQL")
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def save_alerts(self, alerts: List[Dict]):
        """Save alerts to database"""
        if not self.enabled:
            return
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    alert_data = []
                    for alert in alerts:
                        alert_data.append((
                            alert.get('source_country_id', 1),
                            alert.get('target_country_id'),
                            alert.get('product_id'),
                            alert.get('type'),
                            alert.get('severity'),
                            alert.get('title'),
                            alert.get('message'),
                            alert.get('prediction_change_pct', 0),
                            alert.get('sentiment_change', 0)
                        ))
                    
                    execute_values(cur, """
                        INSERT INTO alerts (
                            source_country_id, target_country_id, product_id,
                            alert_type, severity, title, description,
                            prediction_change_pct, sentiment_change
                        ) VALUES %s
                    """, alert_data)
                    
                    logger.info(f"Saved {len(alerts)} alerts to PostgreSQL")
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")
    
    def get_active_alerts(self, limit: int = 50) -> List[Dict]:
        """Get active alerts from database"""
        if not self.enabled:
            return []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT * FROM active_alerts_detailed
                        LIMIT %s
                    """, (limit,))
                    
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def get_india_predictions(self, sector: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get latest predictions for India"""
        if not self.enabled:
            return []
        
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = "SELECT * FROM india_latest_predictions"
                    if sector:
                        query += f" WHERE sector = '{sector}'"
                    query += f" LIMIT {limit}"
                    
                    cur.execute(query)
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return []


# Global database instance
db = PostgresDB()