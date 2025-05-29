from abc import ABC, abstractmethod
from typing import Dict, Any
import openai
import logging
from .data_structures import Document

logger = logging.getLogger(__name__)

class ModelInference(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, context: str) -> str:
        pass

    @abstractmethod
    def query_model(self, document: Document, query: str) -> str:
        pass

class OpenAIInference(ModelInference):
    def __init__(self, api_key: str, model_name: str = "gpt-4o", mock: bool = False):
        logger.debug(f"Initializing OpenAIInference with model: {model_name}")
        self.mock = mock
        if not mock:
            self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"OpenAIInference initialized successfully for model: {model_name}")

    def generate_response(self, prompt: str, context: str) -> str:
        logger.debug(f"Generating response for prompt: {prompt[:100]}...")
        if self.mock:
            logger.debug("Running in mock mode - returning mock response")
            return "This is a mock response for testing purposes."
            
        try:
            logger.debug("Making API call to OpenAI")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
                ],
                temperature=0.0
            )
            result = response.choices[0].message.content
            logger.debug(f"Received response from OpenAI: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return ""

    def query_model(self, document: Document, query: str) -> str:
        """Query the model with a document and query"""
        logger.debug(f"Querying model with query: {query[:100]}...")
        return self.generate_response(query, document.content)
