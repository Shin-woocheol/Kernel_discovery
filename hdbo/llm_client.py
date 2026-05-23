"""
Thin LLM client abstraction for kernel discovery.

Supports two backends:
  - "openai": OpenAI API (GPT models)
  - "vllm":   vLLM OpenAI-compatible server (local open-source models)

Both backends expose the same `generate(prompt) -> str` interface.
vLLM backend supports `seed` for reproducible generation.
"""
import os
import re
import time
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load API keys from .env file if it exists
load_dotenv()

# --- Truncation logging ---
_truncation_log_path: Optional[str] = None


def set_truncation_log(path: str) -> None:
    """Configure a file path to log max_token truncation warnings."""
    global _truncation_log_path
    _truncation_log_path = path


def _warn_truncated(model_name: str, max_tokens: int, prompt_snippet: str) -> None:
    msg = (
        f"[TRUNCATION WARNING] {model_name} hit max_tokens={max_tokens}. "
        f"Response was cut off. Prompt snippet: {prompt_snippet!r}"
    )
    print(f"    {msg}")
    if _truncation_log_path:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(_truncation_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
        except OSError as e:
            print(f"    [llm_client] truncation log write failed: {e}")


class LLMClient:
    """Base class for LLM clients used in kernel evolution."""

    def __init__(self, model_name: str, seed: Optional[int] = None):
        self.model_name = model_name
        self.seed = seed

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client (supports both chat.completions and responses API)."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(model_name, seed)
        from openai import OpenAI

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._client = OpenAI(api_key=self._api_key)
        self._use_responses_api = model_name.startswith("gpt-5")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        start = time.perf_counter()

        if self._use_responses_api:
            response = self._client.responses.create(
                model=self.model_name,
                reasoning={"effort": "low"},
                input=[{"role": "user", "content": prompt}],
            )
            text = response.output_text
        else:
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if self.seed is not None:
                kwargs["seed"] = self.seed
            response = self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            text = choice.message.content
            if choice.finish_reason == "length":
                _warn_truncated(self.model_name, max_tokens, prompt[-200:])

        elapsed = time.perf_counter() - start
        print(f"    [LLM] {self.model_name} responded in {elapsed:.2f}s")
        return text


_THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


class VLLMClient(LLMClient):
    """vLLM OpenAI-compatible server client. Supports seed for reproducibility.

    For reasoning-capable models (Qwen3, etc.), thinking mode is disabled by
    default via chat_template_kwargs so the response stays in our expected
    code/formula format. Any leftover <think>...</think> blocks are stripped
    defensively.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        seed: Optional[int] = None,
        enable_thinking: bool = False,
    ):
        super().__init__(model_name, seed)
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._enable_thinking = enable_thinking

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        start = time.perf_counter()

        extra_body: dict = {}
        if self.seed is not None:
            extra_body["seed"] = self.seed
        if not self._enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        if choice.finish_reason == "length":
            _warn_truncated(self.model_name, max_tokens, prompt[-200:])

        # --- Strip residual <think>...</think> blocks (defensive) ---
        if "<think" in text.lower():
            text = _THINK_TAG_RE.sub("", text).lstrip()

        elapsed = time.perf_counter() - start
        print(f"    [LLM] vLLM/{self.model_name} responded in {elapsed:.2f}s")
        return text


# --- Logging wrapper ---

class LoggingLLMClient(LLMClient):
    """Wraps any LLMClient to log prompts and responses to a file."""

    def __init__(self, inner: LLMClient, log_path: Optional[str] = None):
        super().__init__(inner.model_name, inner.seed)
        self._inner = inner
        self._log_path = log_path
        self._query_count = 0

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        text = self._inner.generate(prompt, temperature, max_tokens)
        self._log_query(prompt, text)
        return text

    def _log_query(self, prompt: str, response: str) -> None:
        if not self._log_path:
            return
        self._query_count += 1
        bar = "=" * 76
        chunk = (
            f"\n{bar}\n"
            f"[Kernel LLM query #{self._query_count}] model={self.model_name}\n"
            f"{bar}\n"
            f"--- prompt ---\n{prompt[:100000]}\n\n"
            f"--- response ---\n{response[:100000] if response else ''}\n"
            f"{bar}\n"
        )
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(chunk)
                f.flush()
        except OSError as e:
            print(f"[llm_client] log failed: {e}")


# --- Factory ---

def create_llm_client(
    backend: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    seed: Optional[int] = None,
    log_path: Optional[str] = None,
) -> LLMClient:
    """
    Create an LLM client.

    Args:
        backend: "openai" or "vllm"
        model_name: Model identifier (e.g. "gpt-4o-mini", "Qwen/Qwen3-8B")
        api_key: API key (for openai backend; defaults to OPENAI_API_KEY env var)
        base_url: vLLM server URL (default: http://localhost:8000/v1)
        seed: Random seed for reproducible generation
        log_path: If set, wrap client with logging to this file
    """
    if backend == "openai":
        client = OpenAIClient(model_name=model_name, api_key=api_key, seed=seed)
    elif backend == "vllm":
        url = base_url or "http://localhost:8000/v1"
        client = VLLMClient(model_name=model_name, base_url=url, seed=seed)
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}. Use 'openai' or 'vllm'.")

    if log_path:
        client = LoggingLLMClient(client, log_path=log_path)

    return client
