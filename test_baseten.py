#!/usr/bin/env python3
import os
import asyncio
from clients import PROVIDERS

async def test_baseten():
    # Make sure you have the BASETEN_API_KEY environment variable set
    if not os.environ.get('BASETEN_API_KEY'):
        print("Please set the BASETEN_API_KEY environment variable")
        return
    
    # Create a Baseten client
    client_class = PROVIDERS['baseten']
    client = client_class(model="Qwen/Qwen3-235B-A22B")
    
    # Initialize the client
    await client.initialize()
    
    # Test a simple prompt
    prompt = "Explain quantum computing in simple terms."
    print(f"Sending prompt to Baseten: {prompt}")
    
    # Get the completion
    response_text = ""
    async for chunk in await client.get_completion_stream(prompt):
        content = client.extract_content_from_chunk(chunk)
        if content:
            print(content, end="", flush=True)
            response_text += content
    
    print("\n\nFull response:")
    print(response_text)

if __name__ == "__main__":
    asyncio.run(test_baseten()) 