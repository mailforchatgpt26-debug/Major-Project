"""
Background scheduler service
Runs weekly updates automatically
"""

import schedule
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_weekly_update():
    """Run the weekly update pipeline"""
    print(f"\n{'='*60}")
    print(f"🕐 Running scheduled update: {datetime.now()}")
    print(f"{'='*60}\n")
    
    script_path = Path(__file__).parent / "weekly_update.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Update completed successfully")
        else:
            print(f"❌ Update failed: {result.stderr}")
    
    except Exception as e:
        print(f"❌ Error running update: {e}")

# Schedule weekly on Sunday at 2 AM
schedule.every().sunday.at("02:00").do(run_weekly_update)

print("🚀 Scheduler service started")
print("📅 Schedule: Every Sunday at 2:00 AM")
print("Press Ctrl+C to stop\n")

# Keep running
while True:
    schedule.run_pending()
    time.sleep(60)  # Check every minute