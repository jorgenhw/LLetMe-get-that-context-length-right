from typing import Dict, Type
import yaml
import os
from .model_interface import ModelInterface
from .gpt4 import GPT4Model

class ModelRegistry:
    """Registry for managing available LLM implementations."""
    
    # Map model names to their implementations
    MODEL_IMPLEMENTATIONS = {
        "gpt4": GPT4Model,
        # Other models will be added here as they're implemented
        # "deepseek": DeepSeekModel,
        # "claude": ClaudeModel,
        # "llama70b": Llama70BModel,
        # "llama8b": Llama8BModel,
    }
    
    @classmethod
    def load_config(cls) -> Dict:
        """Load the configuration file."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    
    @classmethod
    def get_enabled_models(cls) -> Dict[str, ModelInterface]:
        """Get all enabled models from the configuration."""
        config = cls.load_config()
        enabled_models = {}
        
        for model_name, model_config in config["models"].items():
            if model_config.get("enabled", False):
                if model_name in cls.MODEL_IMPLEMENTATIONS:
                    # Replace environment variables in api_key
                    if "api_key" in model_config:
                        model_config["api_key"] = os.path.expandvars(model_config["api_key"])
                    
                    model_class = cls.MODEL_IMPLEMENTATIONS[model_name]
                    enabled_models[model_name] = model_class(model_config)
                else:
                    print(f"Warning: Model {model_name} is enabled but implementation not found")
        
        return enabled_models 