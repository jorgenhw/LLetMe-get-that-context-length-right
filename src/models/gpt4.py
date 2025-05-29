from typing import Dict, Any, Optional
import openai
from .model_interface import ModelInterface

class GPT4Model(ModelInterface):
    def setup(self) -> None:
        """Initialize the OpenAI client."""
        openai.api_key = self.config["api_key"]
        self.model_name = self.config["model_name"]
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 8192)
    
    def generate_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate a response using GPT-4."""
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens
        )
        return response.choices[0].message.content
    
    def get_context_window(self) -> int:
        """Return GPT-4's context window size."""
        return 8192  # Current GPT-4 context window 