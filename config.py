"""Конфігурація проєкту: шляхи, завантаження даних і промптів, фабрика LLM."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
CHECKPOINT_DB = ROOT / "agent_state.db"
OUTBOX_PATH = DATA_DIR / "outbox.json"


def load_candidates() -> dict[str, dict]:
    """Мокові резюме кандидатів."""
    return json.loads((DATA_DIR / "candidates.json").read_text(encoding="utf-8"))


def load_jobs() -> dict[str, dict]:
    """Мокові вимоги вакансій."""
    return json.loads((DATA_DIR / "jobs.json").read_text(encoding="utf-8"))


def load_prompt(name: str) -> str:
    """Системний промпт із prompts/<name>.md."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def make_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Фабрика LLM. Провайдер повністю визначається змінними .env —
    код однаково працює і з OpenAI, і з локальним LM Studio."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY відсутній. Скопіюйте .env.example у .env "
            "(cp .env.example .env) і підставте ключ — для справжнього OpenAI "
            "реальний ключ, для локального LM Studio будь-який непорожній рядок."
        )
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        api_key=api_key,
        temperature=temperature,
    )


def langfuse_enabled() -> bool:
    """Tracing вмикається лише за наявності обох ключів Langfuse — без них
    код працює як і раніше, без будь-яких додаткових залежностей чи мережі.

    Значення обрізається від пробілів: ключ із самих пробілів — це
    порожня конфігурація, а не валідний ключ."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


def make_langfuse_handler():
    """CallbackHandler Langfuse для LangGraph, або None, якщо трейсинг вимкнено.

    Імпорт langfuse навмисно всередині функції: якщо ключів немає (або
    бібліотека не повністю налаштована), config.py все одно імпортується
    без побічних ефектів і без мережі. У Langfuse 3.x+ шлях імпорту —
    langfuse.langchain (у 2.x був langfuse.callback — той шлях більше не
    працює)."""
    if not langfuse_enabled():
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()
