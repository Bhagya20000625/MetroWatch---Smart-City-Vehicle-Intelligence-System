"""
Test script to verify API and database integration
"""

import requests
import time

# Wait for server to start
time.sleep(3)

print("🧪 Testing MetroWatch API with Database Integration")
print("=" * 60)

# Test 1: Health Check
print("\n1️⃣ Testing Health Endpoint...")
try:
    response = requests.get("http://localhost:8000/api/v1/health")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Status: {data['status']}")
        print(f"   📝 Message: {data['message']}")
    else:
        print(f"   ❌ Failed with status code: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Check Analytics (Should be empty initially)
print("\n2️⃣ Testing Analytics Endpoint (Initial State)...")
try:
    response = requests.get("http://localhost:8000/api/v1/analytics/summary")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Total Vehicles in DB: {data['total_vehicles']}")
        print(f"   📊 By Type: {data['by_type']}")
        print(f"   🗺️  By Province: {data['by_province']}")
    else:
        print(f"   ❌ Failed with status code: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Upload an image (if available)
print("\n3️⃣ Would test image upload (needs test image)...")
print("   ⚠️  Skipping for now - you can test manually via Swagger at:")
print("   🌐 http://localhost:8000/docs")

print("\n" + "=" * 60)
print("✅ Database integration test complete!")
print("\n📌 Next Steps:")
print("   1. Open Swagger UI: http://localhost:8000/docs")
print("   2. Test /api/v1/detect endpoint with an image")
print("   3. Check analytics to see database persistence")
print("   4. Verify data in PostgreSQL with psql")
print("=" * 60)
