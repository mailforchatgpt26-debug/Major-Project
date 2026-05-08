"""
Weekly Automation Pipeline
Fetches news → Analyzes sentiment → Retrains model

Save as: scripts/weekly_update.py
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger
from src.pipelines.gdelt_article_scheduler import GDELTArticleFetcher
from src.pipelines.sentiment_analyzer import CompleteSentimentPipeline
from src.data.preprocessing import DataPreprocessor
from src.utils.config import get_settings

settings = get_settings()
logger = setup_logger(
    name="weekly_update",
    log_level="INFO",
    log_file="weekly_update.log",
    console=True
)


class WeeklyUpdatePipeline:
    """
    Complete weekly update pipeline:
    1. Fetch latest news articles
    2. Analyze sentiment
    3. Preprocess data
    4. Retrain model (optional)
    """
    
    def __init__(self, retrain_model: bool = True):
        self.retrain_model = retrain_model
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            'timestamp': self.timestamp,
            'steps_completed': [],
            'errors': []
        }
    
    def step_1_fetch_articles(self) -> bool:
        """Fetch latest news articles"""
        logger.info("="*60)
        logger.info("STEP 1: FETCHING NEWS ARTICLES")
        logger.info("="*60)
        
        try:
            fetcher = GDELTArticleFetcher()
            df = fetcher.fetch_all_articles(max_per_pair=15)
            
            if len(df) > 0:
                fetcher.save_articles(df)
                logger.info(f"✓ Fetched {len(df)} articles")
                self.results['articles_fetched'] = len(df)
                self.results['steps_completed'].append('fetch_articles')
                return True
            else:
                logger.warning("⚠️  No articles fetched")
                self.results['errors'].append('No articles fetched')
                return False
                
        except Exception as e:
            logger.error(f"❌ Article fetch failed: {e}", exc_info=True)
            self.results['errors'].append(f'fetch_articles: {str(e)}')
            return False
    
    def step_2_analyze_sentiment(self) -> bool:
        """Analyze sentiment of fetched articles"""
        logger.info("="*60)
        logger.info("STEP 2: ANALYZING SENTIMENT")
        logger.info("="*60)
        
        try:
            pipeline = CompleteSentimentPipeline()
            articles, bilateral = pipeline.run_full_pipeline(
                extract_content=True,  # Set False for faster testing
                save_results=True
            )
            
            if len(bilateral) > 0:
                logger.info(f"✓ Analyzed {len(articles)} articles")
                logger.info(f"✓ Created {len(bilateral)} bilateral sentiment scores")
                self.results['articles_analyzed'] = len(articles)
                self.results['bilateral_pairs'] = len(bilateral)
                self.results['avg_sentiment'] = float(bilateral['sentiment_score'].mean())
                self.results['steps_completed'].append('analyze_sentiment')
                return True
            else:
                logger.warning("⚠️  No sentiment scores generated")
                self.results['errors'].append('No sentiment scores')
                return False
                
        except Exception as e:
            logger.error(f"❌ Sentiment analysis failed: {e}", exc_info=True)
            self.results['errors'].append(f'analyze_sentiment: {str(e)}')
            return False
    
    def step_3_preprocess_data(self) -> bool:
        """Run data preprocessing with new sentiment"""
        logger.info("="*60)
        logger.info("STEP 3: PREPROCESSING DATA")
        logger.info("="*60)
        
        try:
            preprocessor = DataPreprocessor()
            nodes, edges, metadata = preprocessor.run()
            
            logger.info(f"✓ Preprocessing complete")
            logger.info(f"  Nodes: {len(nodes)}")
            logger.info(f"  Edges: {len(edges)}")
            
            self.results['nodes_created'] = len(nodes)
            self.results['edges_created'] = len(edges)
            self.results['steps_completed'].append('preprocess_data')
            return True
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {e}", exc_info=True)
            self.results['errors'].append(f'preprocess_data: {str(e)}')
            return False
    
    def step_4_train_model(self) -> bool:
        """Retrain GNN model with new data"""
        logger.info("="*60)
        logger.info("STEP 4: RETRAINING MODEL")
        logger.info("="*60)
        
        if not self.retrain_model:
            logger.info("⏭️  Skipping model training (disabled)")
            return True
        
        try:
            # Run training script
            train_script = Path(__file__).parent / "train_model.py"
            
            result = subprocess.run(
                [sys.executable, str(train_script)],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("✓ Model training complete")
                self.results['steps_completed'].append('train_model')
                return True
            else:
                logger.error(f"❌ Training failed with code {result.returncode}")
                logger.error(result.stderr)
                self.results['errors'].append(f'train_model: exit code {result.returncode}')
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Training timed out after 1 hour")
            self.results['errors'].append('train_model: timeout')
            return False
        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            self.results['errors'].append(f'train_model: {str(e)}')
            return False
    
    def save_results(self):
        """Save pipeline results"""
        results_dir = Path("logs/weekly_updates")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"weekly_update_{self.timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved: {results_file}")
    
    def run(self) -> bool:
        """Run complete pipeline"""
        logger.info("\n" + "="*60)
        logger.info("🚀 WEEKLY UPDATE PIPELINE")
        logger.info(f"Started: {datetime.now()}")
        logger.info("="*60 + "\n")
        
        success = True
        
        # Step 1: Fetch articles
        if not self.step_1_fetch_articles():
            logger.warning("⚠️  Continuing despite fetch issues...")
        
        # Step 2: Analyze sentiment
        if not self.step_2_analyze_sentiment():
            logger.error("❌ Cannot continue without sentiment analysis")
            success = False
        
        # Step 3: Preprocess (only if sentiment worked)
        if success:
            if not self.step_3_preprocess_data():
                logger.error("❌ Preprocessing failed")
                success = False
        
        # Step 4: Train model (only if preprocessing worked)
        if success and self.retrain_model:
            if not self.step_4_train_model():
                logger.warning("⚠️  Model training failed, but data is updated")
        
        # Save results
        self.save_results()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("📊 PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Steps completed: {len(self.results['steps_completed'])}/4")
        for step in self.results['steps_completed']:
            logger.info(f"  ✓ {step}")
        
        if self.results['errors']:
            logger.info(f"\nErrors encountered: {len(self.results['errors'])}")
            for error in self.results['errors']:
                logger.error(f"  ✗ {error}")
        
        logger.info(f"\nCompleted: {datetime.now()}")
        logger.info("="*60 + "\n")
        
        return success


def main():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Weekly update pipeline')
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Skip model training (faster)'
    )
    
    args = parser.parse_args()
    
    pipeline = WeeklyUpdatePipeline(retrain_model=not args.no_train)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()