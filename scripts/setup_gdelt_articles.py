"""
One-click setup for GDELT articles.
Fetches initial articles and optionally sets up automated fetching.

Usage:
    python scripts/setup_gdelt_articles.py
"""

import sys
from pathlib import Path
import subprocess
import platform

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.gdelt_article_scheduler import GDELTArticleFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_articles():
    """Fetch initial articles."""
    print("\n" + "="*60)
    print("🌐 GDELT ARTICLE SETUP")
    print("="*60)
    print("Fetching initial articles (this may take 5-10 minutes)...")
    print("="*60 + "\n")
    
    fetcher = GDELTArticleFetcher()
    
    # Fetch articles
    df = fetcher.fetch_all_articles(max_per_pair=15)
    
    if len(df) > 0:
        fetcher.save_articles(df)
        
        print("\n" + "="*60)
        print("✅ INITIAL ARTICLES FETCHED!")
        print("="*60)
        print(f"Articles saved: {len(df):,}")
        print(f"Location: {fetcher.articles_file}")
        print("="*60)
        
        return True
    else:
        print("\n❌ No articles fetched!")
        return False


def setup_automated_fetching():
    """Setup automated article fetching."""
    print("\n" + "="*60)
    print("🤖 AUTOMATED FETCHING SETUP")
    print("="*60)
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("\n📝 macOS Setup (using launchd):")
        print("\n1. Create file: ~/Library/LaunchAgents/com.trade-forecasting.gdelt.plist")
        print("\n2. Paste this content:")
        print("""
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.trade-forecasting.gdelt</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/YOUR_USERNAME/Documents/GNN-Based-Trade-Forecasting/venv/bin/python</string>
        <string>/Users/YOUR_USERNAME/Documents/GNN-Based-Trade-Forecasting/src/pipelines/gdelt_article_scheduler.py</string>
        <string>--once</string>
    </array>
    <key>StartInterval</key>
    <integer>3600</integer>
    <key>StandardOutPath</key>
    <string>/tmp/gdelt-fetcher.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/gdelt-fetcher.error.log</string>
</dict>
</plist>
        """)
        print("\n3. Load it:")
        print("   launchctl load ~/Library/LaunchAgents/com.trade-forecasting.gdelt.plist")
        
    elif system == "Windows":
        print("\n📝 Windows Setup (using Task Scheduler):")
        print("\n1. Open Task Scheduler")
        print("2. Create Basic Task: 'GDELT Article Fetcher'")
        print("3. Trigger: Daily, repeat every 1 hour")
        print("4. Action: Start a program")
        print(f"5. Program: {Path.cwd() / 'venv' / 'Scripts' / 'python.exe'}")
        print(f"6. Arguments: src/pipelines/gdelt_article_scheduler.py --once")
        print(f"7. Start in: {Path.cwd()}")
        
    else:  # Linux
        print("\n📝 Linux Setup (using cron):")
        print("\n1. Edit crontab:")
        print("   crontab -e")
        print("\n2. Add this line:")
        print(f"0 * * * * cd {Path.cwd()} && venv/bin/python src/pipelines/gdelt_article_scheduler.py --once >> /tmp/gdelt-fetcher.log 2>&1")
    
    print("\n" + "="*60)
    print("\n💡 OR run manually as daemon:")
    print("   python src/pipelines/gdelt_article_scheduler.py --daemon")
    print("\n(This will keep running and fetch articles every hour)")
    print("="*60 + "\n")


def main():
    """Main setup."""
    
    # Step 1: Fetch initial articles
    success = setup_articles()
    
    if not success:
        print("\n⚠️  Initial fetch failed, but you can try again later")
    
    # Step 2: Show automated setup instructions
    print("\n" + "="*60)
    response = input("Setup automated fetching? (y/n): ").strip().lower()
    
    if response == 'y':
        setup_automated_fetching()
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\n📊 What you have now:")
    print("  ✅ Initial articles fetched")
    print("  ✅ Articles saved to data/raw/sentiment/articles.csv")
    print("\n🎯 Next steps:")
    print("  1. Run preprocessing: python scripts/preprocess_data.py")
    print("  2. (Optional) Setup automated fetching (see instructions above)")
    print("\n💡 To manually fetch more articles later:")
    print("   python src/pipelines/gdelt_article_scheduler.py --once")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()