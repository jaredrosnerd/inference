"""
Centralized model configuration for all providers.
This file maps friendly model names to provider-specific model IDs.
"""

# Main model configuration dictionary
# Maps friendly model names to provider-specific model IDs or ARNs
MODELS = {
    # Large multimodal models
    "llama4-maverick": {
        "bedrock": "arn:aws:bedrock:us-west-2:932384979644:inference-profile/us.meta.llama4-maverick-17b-instruct-v1:0",
        "fireworks": "accounts/fireworks/models/llama4-maverick-instruct-basic",
    },
    "llama3.3-70b": {
        "bedrock": "arn:aws:bedrock:us-west-2:932384979644:inference-profile/us.meta.llama3-3-70b-instruct-v1:0",
        "fireworks": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    "deepseek-r1": {
        "bedrock": "arn:aws:bedrock:us-west-2:932384979644:inference-profile/us.deepseek.r1-v1:0",
        "fireworks": "accounts/fireworks/models/deepseek-r1",
    },
    "qwen3-235b-a22b": {
        "fireworks": "accounts/fireworks/models/qwen3-235b-a22b",
    },
}

DEFAULT_MODELS = {
    "bedrock": "llama4-maverick",
    "fireworks": "llama4-maverick",
}

def list_models_for_provider(provider: str):
    return [model for model in MODELS if provider in MODELS[model].keys()]

def get_model_id(friendly_name: str, provider: str):
    return MODELS[friendly_name][provider]

def list_all_friendly_models():
    return list(MODELS.keys())

def list_all_models_for_provider(provider: str):
    return list(MODELS[provider].keys())

def get_default_model(provider: str):
    return DEFAULT_MODELS[provider]
