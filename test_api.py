#!/usr/bin/env python3
"""
Diagnostic script to test the Multi-Modal Search Engine API and Database
Run this to identify issues with your search engine setup.
"""

import requests
import chromadb
import json
import sys

def test_database():
    """Test ChromaDB database status"""
    print("=== Testing Database ===")
    try:
        from config import CHROMA_DB_PATH
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = client.list_collections()
        print(f"Available collections: {[c.name for c in collections]}")
        
        if collections:
            try:
                from config import COLLECTION_NAME
                collection = client.get_collection(COLLECTION_NAME)
                count = collection.count()
                print(f"✅ Collection '{COLLECTION_NAME}' found with {count} documents")
                
                # Test a simple query
                if count > 0:
                    test_results = collection.query(
                        query_embeddings=[[0.0] * 512],  # Dummy embedding
                        n_results=1,
                        include=['metadatas']
                    )
                    print(f"✅ Database query test successful")
                    return True, count
                else:
                    print("❌ Collection is empty - need to run ingest_data.py")
                    return False, 0
            except Exception as e:
                print(f"❌ Error accessing collection: {e}")
                return False, 0
        else:
            print("❌ No collections found - need to run ingest_data.py")
            return False, 0
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False, 0

def test_api():
    """Test API server status"""
    print("\n=== Testing API Server ===")
    try:
        # Test basic connectivity
        response = requests.get('http://127.0.0.1:8000/search?query=test', timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ API returned error status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ API server is not running!")
        print("   Start it with: uvicorn api:app --reload --host 127.0.0.1 --port 8000")
        return False
    except requests.exceptions.Timeout:
        print("❌ API request timed out - server might be overloaded")
        return False
    except Exception as e:
        print(f"❌ API error: {e}")
        return False

def test_frontend_connection():
    """Test if frontend can connect to API"""
    print("\n=== Testing Frontend Connection ===")
    try:
        # Test the exact URL the frontend uses
        response = requests.get('http://127.0.0.1:8000/search?query=a', timeout=10)
        if response.status_code == 200:
            print("✅ Frontend should be able to connect to API")
            return True
        else:
            print(f"❌ Frontend connection failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend connection error: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🔍 Multi-Modal Search Engine Diagnostic Tool")
    print("=" * 50)
    
    # Test database
    db_ok, doc_count = test_database()
    
    # Test API
    api_ok = test_api()
    
    # Test frontend connection
    frontend_ok = test_frontend_connection()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    if not db_ok:
        print("❌ DATABASE ISSUE")
        print("   → Run: python ingest_data.py")
        print("   → Wait for ingestion to complete")
    elif not api_ok:
        print("❌ API ISSUE") 
        print("   → Run: uvicorn api:app --reload --host 127.0.0.1 --port 8000")
        print("   → Check for error messages in the API output")
    elif not frontend_ok:
        print("❌ FRONTEND CONNECTION ISSUE")
        print("   → Check if API server is running on correct port")
        print("   → Try opening: http://127.0.0.1:8000/search?query=test")
    else:
        print("✅ EVERYTHING LOOKS GOOD!")
        print(f"   → Database has {doc_count} documents")
        print("   → API is responding correctly")
        print("   → Frontend should work properly")
        print("\n🎉 Try opening frontend/index.html in your browser!")

if __name__ == "__main__":
    main()
