#!/usr/bin/env python3
"""
Simple test script to verify the ML Image Search Server is working
"""

import requests
import time

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        response = requests.get('http://localhost:5001/health', timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check successful!")
            print(f"   Model: {data.get('model')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed with error: {e}")
        return False

def test_preload_model():
    """Test the preload model endpoint"""
    try:
        response = requests.get('http://localhost:5001/preload-model', timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                print("✅ Model preload successful!")
                return True
            else:
                print(f"❌ Model preload failed: {data.get('message')}")
                return False
        else:
            print(f"❌ Model preload failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model preload failed with error: {e}")
        return False

if __name__ == '__main__':
    print("Testing ML Image Search Server...")
    print("=" * 40)
    
    # Test health endpoint
    if test_health_endpoint():
        print()
        # Test model preload
        test_preload_model()
    
    print("=" * 40)
    print("Test completed!")