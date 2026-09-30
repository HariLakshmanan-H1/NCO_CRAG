import requests
import logging
import os
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GemmaLLM:
    def __init__(self, 
                 model_name: str = "gemma:2b",  # Use 2B model for faster responses
                 base_url: str = None,
                 timeout: int = 120,  # Increased timeout
                 temperature: float = 0.7,
                 max_retries: int = 3):
        
        self.model_name = model_name
        # Use env variable if provided, else fallback to localhost
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Check if Ollama is available
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if self.model_name not in model_names:
                    logger.warning(f"Model {self.model_name} not found. Available: {model_names}")
                    # Try to find similar model
                    for m in model_names:
                        if 'gemma' in m.lower():
                            self.model_name = m
                            logger.info(f"Using alternative model: {self.model_name}")
                            return True
                    return False
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(2),  # Reduced retries
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 500,  # Reduced tokens
                 temperature: Optional[float] = None) -> str:
        """
        Generate text using the Gemma model via Ollama
        """
        try:
            # Truncate prompt if too long
            if len(prompt) > 2000:
                prompt = prompt[:2000] + "... [truncated]"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature or self.temperature,
                    "top_k": 20,  # Reduced for faster generation
                    "top_p": 0.8,  # Reduced for faster generation
                    "repeat_penalty": 1.1,
                    "stop": ["\n\n\n", "Human:", "Assistant:"]  # Add stop sequences
                }
            }
            
            logger.info(f"Sending request to Ollama (timeout: {self.timeout}s)")
            
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            if "eval_count" in result:
                logger.info(f"Generated {result['eval_count']} tokens in {result.get('eval_duration', 0)/1e9:.2f}s")
            
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return "⚠️ The request is taking too long. Please try:\n" \
                   "• Using a simpler query\n" \
                   "• Checking if Ollama is overloaded\n" \
                   "• Trying again with a more specific description"
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            return f"⚠️ Cannot connect to Ollama. Please ensure it's running at {self.base_url}"
            
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return f"⚠️ Error generating response. Using retrieved occupations as fallback."
    
    def generate_fast(self, prompt: str, max_tokens: int = 300) -> str:
        """Fast generation with minimal settings"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.5,  # Lower temperature for faster, more deterministic output
                    "top_k": 10,
                    "top_p": 0.5,
                    "repeat_penalty": 1.0,
                    "num_ctx": 1024  # Smaller context window
                }
            }
            
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=30  # Shorter timeout for fast generation
            )
            
            response.raise_for_status()
            return response.json().get("response", "")
            
        except:
            return ""