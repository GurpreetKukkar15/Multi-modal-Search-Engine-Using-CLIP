#!/usr/bin/env python3
"""
Quick setup and run script for the Multi-Modal Search Engine
This script helps you get everything running with proper error checking.
"""

import subprocess
import sys
import time
import requests
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print(" Checking dependencies...")
    required_packages = ['torch', 'transformers', 'chromadb', 'fastapi', 'uvicorn', 'pillow']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("   Install them with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("✅ All dependencies are installed")
        return True

def check_database():
    """Check if database exists and has data"""
    print("\n🔍 Checking database...")
    try:
        import chromadb
        client = chromadb.Client()
        collections = client.list_collections()
        
        if not collections:
            print("❌ No database found. Need to run data ingestion.")
            return False
        
        collection = client.get_collection('image_search_val')
        count = collection.count()
        
        if count == 0:
            print("❌ Database is empty. Need to run data ingestion.")
            return False
        else:
            print(f"✅ Database found with {count} documents")
            return True
            
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def start_api_server():
    """Start the API server in the background"""
    print("\n🚀 Starting API server...")
    try:
        # Start uvicorn in the background
        process = subprocess.Popen([
            sys.executable, '-m', 'uvicorn', 'api:app', 
            '--reload', '--host', '127.0.0.1', '--port', '8000'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Test if server is responding
        try:
            response = requests.get('http://127.0.0.1:8000/health', timeout=5)
            if response.status_code == 200:
                print("✅ API server is running successfully")
                return process
            else:
                print("❌ API server started but not responding properly")
                return None
        except:
            print("❌ API server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def main():
    """Main setup and run function"""
    print("🎯 Multi-Modal Search Engine Setup & Run")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Check database
    if not check_database():
        print("\n📊 Database not ready. Starting data ingestion...")
        if not run_command("python ingest_data.py", "Data ingestion"):
            print("\n❌ Data ingestion failed. Please check the error messages above.")
            return
        print("\n⏳ Data ingestion completed. Please wait a moment...")
        time.sleep(2)
    
    # Start API server
    api_process = start_api_server()
    if not api_process:
        print("\n❌ Failed to start API server")
        return
    
    print("\n" + "=" * 50)
    print("🎉 SUCCESS! Your search engine is ready!")
    print("=" * 50)
    print("📱 Web Interface: Open frontend/index.html in your browser")
    print("🔗 API Health Check: http://127.0.0.1:8000/health")
    print("🔍 API Search Test: http://127.0.0.1:8000/search?query=dog")
    print("\n💡 To stop the server: Press Ctrl+C")
    
    try:
        # Keep the script running
        api_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
        api_process.terminate()
        print("✅ Server stopped")

if __name__ == "__main__":
    main()

