from __future__ import annotations

import requests
from dataclasses import dataclass

from src import config


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    error: str = ""


class BaseLLMClient:
    provider = "base"
    model = "base"

    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Free fake LLM for testing code, logs and metrics without API calls."""
    provider = "mock"
    model = "mock-model"

    def generate(self, prompt: str) -> LLMResponse:
        # Simple deterministic answer extraction for pipeline testing only.
        text = (
            "Thought: I will use the provided context and return a concise answer.\n"
            "Action: ReadContext\n"
            "Observation: Context inspected.\n"
            "Final Answer: mock answer"
        )
        return LLMResponse(text=text, provider=self.provider, model=self.model)


class OllamaLLMClient(BaseLLMClient):
    provider = "ollama"

    def __init__(self) -> None:
        self.base_url = config.OLLAMA_BASE_URL.rstrip("/")
        self.model = config.OLLAMA_MODEL

    def generate(self, prompt: str) -> LLMResponse:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_OUTPUT_TOKENS,
            },
        }
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(text=data.get("response", ""), provider=self.provider, model=self.model)


class OpenAILLMClient(BaseLLMClient):
    provider = "openai"

    def __init__(self) -> None:
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing. Add it to .env or use LLM_PROVIDER=mock.")
        from openai import OpenAI

        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL

    def generate(self, prompt: str) -> LLMResponse:
        is_gpt5 = self.model.startswith("gpt-5")
        token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"
        temperature = 1.0 if is_gpt5 else config.TEMPERATURE
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            token_param: config.MAX_OUTPUT_TOKENS,
        }
        if is_gpt5:
            request["reasoning_effort"] = config.OPENAI_REASONING_EFFORT

        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            if "reasoning_effort" not in request or "reasoning_effort" not in str(exc).lower():
                raise
            request.pop("reasoning_effort")
            response = self.client.chat.completions.create(**request)

        if not response.choices:
            raise RuntimeError("OpenAI returned no completion choices.")

        choice = response.choices[0]
        text = choice.message.content or ""
        error = ""
        if not text.strip():
            usage = getattr(response, "usage", None)
            details = getattr(usage, "completion_tokens_details", None)
            diagnostics = [
                f"finish_reason={getattr(choice, 'finish_reason', None)}",
                f"completion_tokens={getattr(usage, 'completion_tokens', None)}",
                f"reasoning_tokens={getattr(details, 'reasoning_tokens', None)}",
            ]
            error = "WARNING: OpenAI returned empty visible text (" + ", ".join(diagnostics) + ")."

        return LLMResponse(text=text, provider=self.provider, model=self.model, error=error)


class GeminiLLMClient(BaseLLMClient):
    provider = "gemini"

    def __init__(self) -> None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing. Add it to .env or use LLM_PROVIDER=mock.")
        import google.generativeai as genai

        genai.configure(api_key=config.GEMINI_API_KEY)
        self.model = config.GEMINI_MODEL
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt: str) -> LLMResponse:
        response = self.client.generate_content(
            prompt,
            generation_config={
                "temperature": config.TEMPERATURE,
                "max_output_tokens": config.MAX_OUTPUT_TOKENS,
            },
        )
        return LLMResponse(text=response.text or "", provider=self.provider, model=self.model)


def get_llm_client() -> BaseLLMClient:
    provider = config.LLM_PROVIDER
    if provider == "mock":
        return MockLLMClient()
    if provider == "ollama":
        return OllamaLLMClient()
    if provider == "openai":
        return OpenAILLMClient()
    if provider == "gemini":
        return GeminiLLMClient()
    raise ValueError(f"Unsupported LLM_PROVIDER={provider}. Use mock, ollama, openai, or gemini.")
