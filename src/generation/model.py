"""Language model utilities."""
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging

logger = logging.getLogger(__name__)

class LanguageModel:
    """Llama language model for generation."""

    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B",
                 device: str = "cpu", max_new_tokens: int = 512,
                 temperature: float = 0.1, top_p: float = 0.9,
                 do_sample: bool = False,
                 use_8bit: bool = False, use_4bit: bool = False):
        """
        Initialize language model.

        Args:
            model_name: Hugging Face model name
            device: Device to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_8bit: Load model in 8-bit quantization
            use_4bit: Load model in 4-bit quantization
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

        logger.info(f"Loading model: {model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine dtype and quantization config
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

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_kwargs
        )

        # Only move to device manually when not using quantization
        # (bitsandbytes handles device placement automatically)
        if not (use_8bit or use_4bit):
            self.model.to(device)

        self.model.eval()

        logger.info(f"Model loaded on {device}")

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        # Use chat template if the tokenizer supports it (instruction-tuned models)
        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        ).to(self.device)

        # Generate
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
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()
