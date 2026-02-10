"""
Database Initialization Script
Run this to create database tables.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.database.database import init_db

if __name__ == "__main__":
    print("🔧 Initializing MetroWatch Database...")
    print("=" * 50)
    
    try:
        init_db()
        print("=" * 50)
        print("✅ Database setup complete!")
        print("\nTables created:")
        print("  - vehicles (vehicle detections)")
        print("  - vehicle_logs (tracking events)")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)
