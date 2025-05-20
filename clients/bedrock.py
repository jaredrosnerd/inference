import os
import json
import boto3
import asyncio
from .base import LLMClient
import time

class BedrockClient(LLMClient):
    """Client for AWS Bedrock API"""
    
    def __init__(self, model=None, api_key=None, base_url=None, region=None):
        """Initialize the Bedrock client
        
        Args:
            model: The model ID to use (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
            api_key: Not used (AWS credentials are handled by boto3)
            base_url: Not used (AWS endpoints are handled by boto3)
            region: AWS region (e.g., "us-west-2" for Oregon)
        """
        # Use the provided model ID or default to a standard model
        self.model = model or "mistral.mistral-large-2407-v1:0"
        self.client = None
        self.network_latency = 0
        
        # Set AWS region (default to us-west-2 if not specified)
        self.region = region or os.environ.get('AWS_REGION', 'us-west-2')
        
        # Determine the model provider
        self._set_provider()
    
    def _set_provider(self):
        """Determine the model provider based on the model ID"""
        # Handle special cases for different model ID formats
        if self.model.startswith("us.meta."):
            self.provider = "meta"
        elif self.model.startswith("arn:aws:bedrock:"):
            # Extract provider from ARN
            if "meta" in self.model:
                self.provider = "meta"
            elif "anthropic" in self.model:
                self.provider = "anthropic"
            elif "mistral" in self.model:
                self.provider = "mistral"
            elif "amazon" in self.model:
                self.provider = "amazon"
            elif "cohere" in self.model:
                self.provider = "cohere"
            elif "deepseek" in self.model:
                self.provider = "deepseek"
            else:
                # Default to meta for unknown ARNs
                self.provider = "meta"
        else:
            # Standard format: provider.model-name
            self.provider = self.model.split('.')[0] if '.' in self.model else "unknown"
    
    async def initialize(self):
        """Initialize the Bedrock client"""
        # Create Bedrock client using boto3 with specified region
        self.client = boto3.client('bedrock-runtime', region_name=self.region)
        
        # Measure network latency
        await self.measure_network_latency()
        
        # Update provider in case the model changed
        self._set_provider()
        
        print(f"Using model: {self.model}")
        
        return self
    
    async def measure_network_latency(self):
        """Measure network latency to AWS Bedrock using socket connection"""
        import time
        import socket
        
        try:
            # Get the endpoint for the region
            endpoint = f"bedrock-runtime.{self.region}.amazonaws.com"
            
            # Measure latency using socket connection
            latencies = []
            
            # Run multiple tests for more accurate measurement
            for _ in range(3):
                start_time = time.time()
                
                # Use a simple socket connection to measure latency
                # Run in an executor to avoid blocking the event loop
                await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: socket.create_connection((endpoint, 443), timeout=5).close()
                )
                
                end_time = time.time()
                latency = end_time - start_time
                latencies.append(latency)
            
            # Calculate average latency
            self.network_latency = sum(latencies) / len(latencies)
            print(f"Network latency to AWS Bedrock ({self.region}): {self.network_latency*1000:.2f}ms")
            
        except Exception as e:
            print(f"Warning: Could not measure network latency to AWS Bedrock: {str(e)}")
            
            # Use a default latency estimate based on region
            default_latencies = {
                'us-east-1': 0.050,  # 50ms for US East (N. Virginia)
                'us-east-2': 0.060,  # 60ms for US East (Ohio)
                'us-west-1': 0.080,  # 80ms for US West (N. California)
                'us-west-2': 0.070,  # 70ms for US West (Oregon)
                'eu-west-1': 0.100,  # 100ms for EU (Ireland)
                'eu-central-1': 0.110,  # 110ms for EU (Frankfurt)
                'ap-northeast-1': 0.150,  # 150ms for Asia Pacific (Tokyo)
                'ap-southeast-1': 0.180,  # 180ms for Asia Pacific (Singapore)
                'ap-southeast-2': 0.200,  # 200ms for Asia Pacific (Sydney)
            }
            
            # Get default latency for the region, or use 100ms if region not in the list
            self.network_latency = default_latencies.get(self.region, 0.100)
            print(f"Using default network latency estimate for {self.region}: {self.network_latency*1000:.2f}ms")
        
        return self.network_latency
    
    async def get_completion_stream(self, prompt):
        """Get a streaming completion from AWS Bedrock"""
        
        # Format the request based on the model provider
        if self.provider == "anthropic":
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5000,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": True
            }
        elif self.provider == "meta":
            # Check if this model requires provisioned throughput (ARN or us.meta.*)
            requires_provisioned = (
                self.model.startswith("arn:aws:bedrock:") or
                self.model.startswith("us.meta.")
            )
            
            if requires_provisioned:
                # For provisioned Meta models, don't include the stream parameter
                request_body = {
                    "prompt": f"<s>[INST] {prompt} [/INST]",
                    "max_gen_len": 5000
                    # No stream parameter for provisioned models
                }
            else:
                # For on-demand Meta models, include the stream parameter
                request_body = {
                    "prompt": f"<s>[INST] {prompt} [/INST]",
                    "max_gen_len": 5000,
                    "stream": True
                }
        elif self.provider == "mistral":
            request_body = {
                "prompt": prompt,
                "max_tokens": 5000,
                "stream": True
            }
        elif self.provider == "amazon":
            request_body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 5000,
                    "stopSequences": [],
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
        elif self.provider == "cohere":
            request_body = {
                "prompt": prompt,
                "max_tokens": 5000,
                "stream": True
            }
        elif self.provider == "deepseek":
            request_body = {
                "prompt": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "max_tokens": 5000,
                "stream": True
            }
        else:
            raise ValueError(f"Unsupported model provider: {self.provider}")
        
        # Create a generator that will yield chunks from the stream
        async def generate():
            try:
                # The model ID is already the correct ID (either standard ID or ARN)
                model_id = self.model
                
                # For ARN models, log that we're using an inference profile
                if model_id.startswith("arn:aws:bedrock:"):
                    print("Using inference profile: ", model_id)
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.invoke_model_with_response_stream(
                        modelId=model_id,
                        body=json.dumps(request_body)
                    )
                )
                
                # Process the streaming response
                stream = response.get('body')
                
                if stream:
                    for event in stream:
                        chunk = event.get('chunk')
                        if chunk:
                            chunk_data = json.loads(chunk.get('bytes').decode())
                            yield BedrockResponse(chunk_data)
            
            except Exception as e:
                if "ValidationException" in str(e) and "provisioned throughput" in str(e):
                    raise ValueError(f"Error: The model '{self.model}' requires provisioned throughput. Please use a model that supports on-demand throughput, such as amazon.titan-text-express-v1")
                elif "AccessDeniedException" in str(e):
                    raise ValueError(f"Error: You don't have access to the model '{self.model}'. Please request access in the AWS Bedrock console.")
                elif "ResourceNotFoundException" in str(e):
                    raise ValueError(f"Error: The model '{self.model}' was not found. Please check if the model ID is correct.")
                elif "Malformed input request" in str(e):
                    raise ValueError(f"Error: Malformed request for model '{self.model}'. Error details: {str(e)}")
                else:
                    raise ValueError(f"Error getting completion from AWS Bedrock: {str(e)}")
        
        return generate()
    
    def extract_content_from_chunk(self, chunk):
        """Extract content from a response chunk"""
        try:
            if not hasattr(chunk, 'data'):
                return ""
            
            data = chunk.data
            
            # For converse_stream responses
            if "text" in data:
                return data["text"]
            
            # Handle different response formats based on provider
            if self.provider == "anthropic":
                if "delta" in data and "text" in data["delta"]:
                    return data["delta"]["text"]
            elif self.provider == "meta":
                if "generation" in data:
                    return data["generation"]
            elif self.provider == "mistral":
                if "outputs" in data and len(data["outputs"]) > 0:
                    if "text" in data["outputs"][0]:
                        return data["outputs"][0]["text"]
            elif self.provider == "amazon":
                if "outputText" in data:
                    return data["outputText"]
            elif self.provider == "cohere":
                if "text" in data:
                    return data["text"]
            elif self.provider == "deepseek":
                if "generation" in data:
                    return data["generation"]
            
            return ""
        except Exception as e:
            print(f"Error extracting content from chunk: {str(e)}")
            return ""
    
    @property
    def name(self):
        """Return a human-readable name for this client"""
        # For ARN models, extract a more readable name
        if self.model.startswith("arn:aws:bedrock:"):
            # Try to extract a meaningful name from the ARN
            parts = self.model.split('/')
            if len(parts) > 1:
                return f"Bedrock-{parts[-1]}"
            return f"Bedrock-{self.provider}-provisioned"
        
        return f"Bedrock-{self.model}"

# Helper class to store response data
class BedrockResponse:
    """Wrapper for Bedrock response to make it compatible with other clients"""
    
    def __init__(self, data):
        self.data = data 