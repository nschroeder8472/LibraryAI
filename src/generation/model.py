"""Language model utilities with local-only provider support.

Supports local HuggingFace models and Ollama (local server).
The backend is selected via the GENERATION_BACKEND environment variable.
All inference runs locally — no data leaves your machine.
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


class OllamaLanguageModel(BaseLanguageModel):
    """Ollama-based language model (local server, no data leaves your machine).

    Ollama runs models locally and exposes them via a REST API.
    Install: https://ollama.com
    Pull a model: ollama pull llama3.2
    """

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
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running (ollama serve). Error: {e}"
            ) from e

        return data.get("response", "").strip()


# Keep backward-compatible alias
LanguageModel = LocalLanguageModel


def create_language_model(backend: str = "ollama", **kwargs) -> BaseLanguageModel:
    """Factory function to create the appropriate language model.

    Both backends run entirely locally — no data leaves your machine.

    Args:
        backend: 'ollama' (recommended) or 'local' (HuggingFace transformers)
        **kwargs: Arguments passed to the specific model class

    Returns:
        A BaseLanguageModel instance
    """
    backends = {
        "local": LocalLanguageModel,
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
