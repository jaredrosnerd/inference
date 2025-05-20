import os
from openai import AsyncOpenAI
from .base import LLMClient

class BasetenClient(LLMClient):
    """Baseten AI client implementation using OpenAI compatibility layer"""
    
    def __init__(self, model=None):
        self.model = model or os.environ.get("BASETEN_DEFAULT_MODEL_ID")
        self.client = None
    
    async def initialize(self):
        """Initialize the Baseten client with OpenAI compatibility"""
        api_key = os.environ.get("BASETEN_API_KEY")
        if not api_key:
            raise ValueError("BASETEN_API_KEY environment variable not set")
        
        if not self.model:
            raise ValueError("No model ID provided. Set BASETEN_DEFAULT_MODEL_ID or pass model ID to constructor")
        
        # Initialize OpenAI client with Baseten endpoint
        self.client = AsyncOpenAI(
            base_url="https://api.baseten.co/v1",  # Baseten OpenAI-compatible endpoint
            api_key=api_key
        )
        return self
    
    async def get_completion_stream(self, prompt):
        """Get a stream of completion chunks from Baseten API using OpenAI compatibility"""
        return self.client.chat.completions.acreate(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=True
        )
    
    def extract_content_from_chunk(self, chunk):
        """Extract content from a Fireworks API response chunk"""
        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
            content = chunk.choices[0].delta.content
            return content if content else ""
        return ""
    
    @property
    def name(self):
        """Return a human-readable name for this client"""
        model_name = self.model
        # Try to extract a more readable name if possible
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        return f"Baseten-{model_name}"