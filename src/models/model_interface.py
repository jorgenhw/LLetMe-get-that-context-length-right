from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ModelInterface(ABC):
    """Base interface that all LLM implementations must follow."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup()
    
    @abstractmethod
    def setup(self) -> None:
        """Initialize any necessary clients or configurations."""
        pass
    
    @abstractmethod
    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: The input prompt to send to the model
            max_tokens: Optional maximum number of tokens to generate
            
        Returns:
            str: The model's response
        """
        pass
    
    @abstractmethod
    def get_context_window(self) -> int:
        """Return the model's maximum context window size in tokens."""
        pass 