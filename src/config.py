import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


def _env_or_dotenv(key: str, default: str = "") -> str:
    """Prefer a real environment variable, then the first non-empty .env value."""
    value = os.getenv(key, "")
    if value.strip():
        return value

    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, raw_value = line.split("=", 1)
            if name.strip() != key:
                continue
            candidate = raw_value.strip().strip('"').strip("'")
            if candidate:
                return candidate
    return default

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs"

LLM_PROVIDER = _env_or_dotenv("LLM_PROVIDER", "mock").lower().strip()
TEMPERATURE = float(_env_or_dotenv("TEMPERATURE", "0.2"))
MAX_OUTPUT_TOKENS = int(_env_or_dotenv("MAX_OUTPUT_TOKENS", "700"))

OPENAI_API_KEY = _env_or_dotenv("OPENAI_API_KEY", "")
OPENAI_MODEL = _env_or_dotenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_REASONING_EFFORT = _env_or_dotenv("OPENAI_REASONING_EFFORT", "low").lower().strip()

GEMINI_API_KEY = _env_or_dotenv("GEMINI_API_KEY", "")
GEMINI_MODEL = _env_or_dotenv("GEMINI_MODEL", "gemini-1.5-flash")

OLLAMA_BASE_URL = _env_or_dotenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _env_or_dotenv("OLLAMA_MODEL", "llama3.1:8b")
