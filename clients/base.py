from abc import ABC, abstractmethod

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def initialize(self):
        """Initialize the client"""
        pass
    
    @abstractmethod
    async def get_completion_stream(self, prompt):
        """Get a stream of completion chunks for the given prompt
        
        Returns:
            An async generator that yields completion chunks
        """
        pass
    
    @abstractmethod
    def extract_content_from_chunk(self, chunk):
        """Extract text content from a chunk of the completion stream
        
        Returns:
            str: The extracted text content, or empty string if none
        """
        pass
    
    @property
    @abstractmethod
    def name(self):
        """Return the name of the client"""
        pass
