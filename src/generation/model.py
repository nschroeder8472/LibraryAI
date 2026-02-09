"""Language model utilities with multi-provider support.

Supports local HuggingFace models, OpenAI API, Anthropic API, and Ollama.
The backend is selected via the GENERATION_BACKEND environment variable.
"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseLanguageModel(ABC):
    """Abstract base class for language model providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...


class LocalLanguageModel(BaseLanguageModel):
    """Local HuggingFace model loaded via transformers."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B",
                 device: str = "cpu", max_new_tokens: int = 512,
                 temperature: float = 0.1, top_p: float = 0.9,
                 do_sample: bool = False,
                 use_8bit: bool = False, use_4bit: bool = False):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        logger.info(f"Loading local model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"low_cpu_mem_usage": True}
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            logger.info("Using 4-bit quantization")
        elif use_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logger.info("Using 8-bit quantization")
        elif device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )
        if not (use_8bit or use_4bit):
            self.model.to(device)
        self.model.eval()
        logger.info(f"Model loaded on {device}")

    def generate(self, prompt: str) -> str:
        import torch

        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        inputs = self.tokenizer(
            input_text, return_tensors="pt",
            truncation=True, max_length=8192
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated_text.strip()


class OpenAILanguageModel(BaseLanguageModel):
    """OpenAI API-based language model."""

    def __init__(self, model_name: str = "gpt-4o-mini",
                 max_new_tokens: int = 512,
                 temperature: float = 0.1, api_key: str = None):
        import os
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI backend. "
                "Install it with: pip install openai"
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var "
                "or pass api_key parameter."
            )
        self.client = OpenAI(api_key=key)
        logger.info(f"OpenAI client initialized with model: {model_name}")

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()


class AnthropicLanguageModel(BaseLanguageModel):
    """Anthropic API-based language model."""

    def __init__(self, model_name: str = "claude-sonnet-4-20250514",
                 max_new_tokens: int = 512,
                 temperature: float = 0.1, api_key: str = None):
        import os
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic backend. "
                "Install it with: pip install anthropic"
            )

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                "or pass api_key parameter."
            )
        self.client = anthropic.Anthropic(api_key=key)
        logger.info(f"Anthropic client initialized with model: {model_name}")

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


class OllamaLanguageModel(BaseLanguageModel):
    """Ollama API-based language model (local server)."""

    def __init__(self, model_name: str = "llama3.2",
                 max_new_tokens: int = 512,
                 temperature: float = 0.1,
                 base_url: str = None):
        import os
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        logger.info(f"Ollama client initialized: {model_name} at {self.base_url}")

    def generate(self, prompt: str) -> str:
        import urllib.request
        import json

        url = f"{self.base_url}/api/generate"
        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_new_tokens,
                "temperature": self.temperature,
            },
        }).encode()

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        return data.get("response", "").strip()


# Keep backward-compatible alias
LanguageModel = LocalLanguageModel


def create_language_model(backend: str = "local", **kwargs) -> BaseLanguageModel:
    """Factory function to create the appropriate language model.

    Args:
        backend: One of 'local', 'openai', 'anthropic', 'ollama'
        **kwargs: Arguments passed to the specific model class

    Returns:
        A BaseLanguageModel instance
    """
    backends = {
        "local": LocalLanguageModel,
        "openai": OpenAILanguageModel,
        "anthropic": AnthropicLanguageModel,
        "ollama": OllamaLanguageModel,
    }

    cls = backends.get(backend)
    if cls is None:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Choose from: {', '.join(backends.keys())}"
        )

    # Filter kwargs to only those accepted by the target class
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}

    return cls(**filtered)
