# Import all client implementations here for easy access
from .base import LLMClient
from .openai_compatible import OpenAICompatibleClient
from .bedrock import BedrockClient

# Create provider-specific instances of the OpenAICompatibleClient
FireworksClient = lambda model=None: OpenAICompatibleClient(provider='fireworks', model=model)
OpenAIClient = lambda model=None: OpenAICompatibleClient(provider='openai', model=model)
BasetenClient = lambda model=None: OpenAICompatibleClient(provider='baseten', model=model)
TogetherClient = lambda model=None: OpenAICompatibleClient(provider='together', model=model)
AnthropicClient = lambda model=None: OpenAICompatibleClient(provider='anthropic', model=model)

# Map of provider names to client factory functions
PROVIDERS = {
    'fireworks': FireworksClient,
    'openai': OpenAIClient,
    'baseten': BasetenClient,
    'together': TogetherClient,
    'anthropic': AnthropicClient,
    'bedrock': lambda **kwargs: BedrockClient(**kwargs)  # Let main.py handle the region
} 