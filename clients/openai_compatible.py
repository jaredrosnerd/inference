import os
import time
import asyncio
from openai import AsyncOpenAI
from .base import LLMClient
import socket
from urllib.parse import urlparse

class OpenAICompatibleClient(LLMClient):
    """Unified client for all OpenAI-compatible APIs"""
    
    # Default configurations for different providers
    PROVIDER_CONFIGS = {
        'openai': {
            'base_url': None,  # Use default OpenAI URL
            'env_key': 'OPENAI_API_KEY',  # Name of the environment variable
            'default_model': 'gpt-3.5-turbo',
            'name_prefix': 'OpenAI',
            'ping_endpoint': '/v1/models'  # Endpoint to use for ping test
        },
        'fireworks': {
            'base_url': 'https://api.fireworks.ai/inference/v1',
            'env_key': 'FIREWORKS_API_KEY',
            'default_model': 'accounts/fireworks/models/llama4-maverick-instruct-basic',
            'name_prefix': 'Fireworks',
            'ping_endpoint': '/models'  # Endpoint to use for ping test
        },
        'baseten': {
            'base_url': 'https://api.baseten.co/v1',
            'env_key': 'BASETEN_API_KEY',
            'default_model': 'baseten/baseten-llama-3-8b-instruct',
            'name_prefix': 'Baseten',
            'ping_endpoint': '/models'  # Endpoint to use for ping test
        },
        'together': {
            'base_url': 'https://api.together.xyz/v1',
            'env_key': 'TOGETHER_API_KEY',
            'default_model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
            'name_prefix': 'Together',
            'ping_endpoint': '/models'  # Endpoint to use for ping test
        },
        'anthropic': {
            'base_url': 'https://api.anthropic.com/v1',
            'env_key': 'ANTHROPIC_API_KEY',
            'default_model': 'claude-3-opus-20240229',
            'name_prefix': 'Anthropic',
            'ping_endpoint': '/models'  # Endpoint to use for ping test
        }
    }
    
    def __init__(self, provider='openai', model=None, api_key=None, base_url=None):
        """Initialize the OpenAI-compatible client
        
        Args:
            provider: The provider name (openai, fireworks, baseten, etc.)
            model: The model to use (provider-specific)
            api_key: Optional API key (if not provided, will use environment variable)
            base_url: Optional base URL (if not provided, will use provider default)
        """
        if provider not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(self.PROVIDER_CONFIGS.keys())}")
        
        self.provider = provider
        self.config = self.PROVIDER_CONFIGS[provider]
        
        # Set model (use provided model, or default from config)
        self.model = model or self.config['default_model']
        if not self.model:
            raise ValueError(f"No model specified for {provider} and no default available")
        
        # Set API key and base URL
        self.api_key = api_key
        self.base_url = base_url or self.config['base_url']
        self.client = None
        
        # Network latency metrics
        self.network_latency = None  # Will be measured during initialization
    
    async def measure_network_latency(self):
        """Measure network latency using a simple ping implementation"""
        try:
            # Extract the hostname from the base URL
            parsed_url = urlparse(self.base_url or "https://api.openai.com")
            hostname = parsed_url.netloc
            
            # Ping the hostname
            latencies = []
            for _ in range(5):
                start_time = time.time()
                try:
                    # Simple socket connection to measure network latency
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2.0)
                    s.connect((hostname, 443))  # HTTPS port
                    s.close()
                    end_time = time.time()
                    latency = end_time - start_time
                    latencies.append(latency)
                except Exception as e:
                    print(f"Socket ping failed: {str(e)}")
                    continue
            
            if not latencies:
                print(f"Warning: Could not measure network latency for {self.provider}")
                self.network_latency = 0
                return 0
            
            # Calculate average latency (in seconds), excluding outliers
            latencies.sort()
            if len(latencies) > 2:
                # Remove highest and lowest values to reduce impact of outliers
                latencies = latencies[1:-1]
            
            self.network_latency = sum(latencies) / len(latencies)
            print(f"Network latency to {self.provider}: {self.network_latency*1000:.2f}ms")
            return self.network_latency
        except Exception as e:
            print(f"Warning: Could not measure network latency: {str(e)}")
            self.network_latency = 0
            return 0
    
    async def initialize(self):
        """Initialize the client with the appropriate configuration"""
        # Get API key from environment if not provided
        if not self.api_key:
            env_key_name = self.config['env_key']
            self.api_key = os.environ.get(env_key_name)
            if not self.api_key:
                raise ValueError(f"{env_key_name} environment variable not set")
        
        # Initialize OpenAI client with appropriate base URL
        client_kwargs = {'api_key': self.api_key}
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
        
        self.client = AsyncOpenAI(**client_kwargs)
        
        # Measure network latency
        await self.measure_network_latency()
        
        return self
    
    async def get_completion_stream(self, prompt):
        """Get a stream of completion chunks using OpenAI compatibility"""
        try:
            # Different providers might use different methods
            if hasattr(self.client.chat.completions, 'acreate'):
                # For Fireworks which uses acreate
                response = await self.client.chat.completions.acreate(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    stream=True
                )
                return response
            else:
                # For standard OpenAI API
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    stream=True
                )
                return response
        except Exception as e:
            raise ValueError(f"Error creating completion stream: {str(e)}")
    
    def extract_content_from_chunk(self, chunk):
        """Extract content from a response chunk"""
        try:
            # Handle different response formats
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    content = choice.delta.content
                    return content if content else ""
            return ""
        except Exception as e:
            print(f"Error extracting content from chunk: {str(e)}")
            return ""
    
    @property
    def name(self):
        """Return a human-readable name for this client"""
        model_name = self.model
        # Try to extract a more readable name if possible
        if '/' in model_name:
            model_name = model_name.split('/')[-1]
        return f"{self.config['name_prefix']}-{model_name}" 