#!/usr/bin/env python3
import os
import asyncio
import json
import aiohttp

async def test_fireworks():
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set")
        return
    
    # First, list available models
    async with aiohttp.ClientSession() as session:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Try to list models
        print("Listing models...")
        async with session.get("https://api.fireworks.ai/v1/models", headers=headers) as response:
            if response.status == 200:
                models = await response.json()
                print("Available models:")
                for model in models.get('data', []):
                    if "qwen" in model.get('id', '').lower():
                        print(f"  - {model.get('id')}")
            else:
                print(f"Error listing models: {response.status}")
                print(await response.text())
        
        # Try to use the model directly
        model_id = "accounts/fireworks/models/qwen3-235b-a22b"
        print(f"\nTesting model: {model_id}")
        
        # Try both endpoints
        endpoints = [
            "https://api.fireworks.ai/v1/chat/completions",
            "https://api.fireworks.ai/inference/v1/chat/completions"
        ]
        
        for endpoint in endpoints:
            print(f"\nTrying endpoint: {endpoint}")
            request_data = {
                "model": model_id,
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            }
            
            try:
                async with session.post(endpoint, headers=headers, json=request_data) as response:
                    print(f"Status: {response.status}")
                    response_text = await response.text()
                    print(f"Response: {response_text[:200]}...")
            except Exception as e:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_fireworks()) 