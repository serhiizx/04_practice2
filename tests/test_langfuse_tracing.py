"""Офлайн-тести підключення Langfuse: без мережі, без ключів, без LLM.

Живий трейс без ключів перевірити неможливо — перевіряється коректна
деградація: чиста логіка langfuse_enabled(), відсутність імпорту langfuse
на рівні модуля config.py (статичний капкан через ast — надійніший за
підміну sys.modules, бо не залежить від того, коли langfuse завантажився
в процес pytest), робота main.screen() без ключів і умовний flush().
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import config
import main
from conftest import StubTool


# --- 1. langfuse_enabled() -------------------------------------------------


@pytest.mark.parametrize(
    "public_key,secret_key,expected",
    [
        (None, None, False),
        ("pk-lf-test", None, False),
        (None, "sk-lf-test", False),
        ("", "sk-lf-test", False),
        # Ключ із самих пробілів — порожня конфігурація, а не валідний ключ.
        ("   ", "sk-lf-test", False),
        ("pk-lf-test", "sk-lf-test", True),
    ],
)
def test_langfuse_enabled(monkeypatch, public_key, secret_key, expected):
    for name, value in (("LANGFUSE_PUBLIC_KEY", public_key), ("LANGFUSE_SECRET_KEY", secret_key)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert config.langfuse_enabled() is expected


# --- 2. make_langfuse_handler() без ключів не імпортує langfuse -----------


def test_make_langfuse_handler_none_bez_kliuchiv(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    assert config.make_langfuse_handler() is None


def test_config_ne_importuye_langfuse_na_rivni_modulya():
    """Статичний капкан: серед import-інструкцій РІВНЯ МОДУЛЯ config.py
    (tree.body, а не вкладених у FunctionDef) немає жодної langfuse."""
    tree = ast.parse(Path(config.__file__).read_text(encoding="utf-8"))

    top_level_module_names = []
    for node in tree.body:  # лише вузли рівня модуля
        if isinstance(node, ast.Import):
            top_level_module_names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_module_names.append(node.module)

    assert not any(name.startswith("langfuse") for name in top_level_module_names)


# --- 3. Граф з main.screen() працює однаково без ключів Langfuse ----------


class _StubScreeningGraph:
    """Двійник скомпільованого графа: без interrupt, без checkpointer,
    без LLM — лише фіксує config, з яким його реально викликали."""

    def __init__(self, captured_configs: list[dict]):
        self._captured = captured_configs

    async def ainvoke(self, state, config):
        self._captured.append(config)
        return {**state, "report": "Звіт: strong_match (без Langfuse-ключів)"}


async def _run_screen_stub(monkeypatch, tmp_path, thread_id: str, captured=None):
    """Обв'язка: build_graph і load_mcp_tools — стаби (LLM і MCP-сервер тут
    не тестуються, це покрито test_human_in_the_loop.py і
    test_langgraph_wiring.py), сам виклик config-словника — справжній код."""
    monkeypatch.setattr(main, "CHECKPOINT_DB", tmp_path / "state.db")

    async def _stub_load_mcp_tools():
        return None, [StubTool("fetch_resume")]

    async def _stub_build_graph(tools, checkpointer=None, llm=None):
        return _StubScreeningGraph(captured if captured is not None else [])

    monkeypatch.setattr(main, "load_mcp_tools", _stub_load_mcp_tools)
    monkeypatch.setattr(main, "build_graph", _stub_build_graph)

    return await main.screen("CAND-001", "JOB-BACKEND", thread_id)


async def test_screen_pratsyuye_povnistyu_bez_langfuse_kliuchiv(tmp_path, monkeypatch):
    """Ключова перевірка брифу: трейсинг — не обов'язкова залежність
    робочого шляху."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    captured_configs: list[dict] = []
    state = await _run_screen_stub(monkeypatch, tmp_path, "thread-bez-langfuse", captured_configs)

    assert state["report"] == "Звіт: strong_match (без Langfuse-ключів)"
    assert len(captured_configs) == 1
    used_config = captured_configs[0]
    # Без ключів handler — None, тож callbacks порожній, а не [None].
    assert used_config["callbacks"] == []
    assert used_config["metadata"]["langfuse_tags"] == ["impl:langgraph", "candidate:CAND-001"]
    assert used_config["metadata"]["langfuse_session_id"] == "thread-bez-langfuse"


# --- 4. Умовний flush(): лише коли трейсинг увімкнено ----------------------


class _FakeLangfuseClient:
    """Двійник клієнта get_client() — рахує виклики flush(), без мережі."""

    def __init__(self):
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1


@pytest.mark.parametrize("keys_present,expected_flushes", [(False, 0), (True, 1)])
async def test_flush_lyshe_koly_treisynh_uvimknenyi(
    tmp_path, monkeypatch, keys_present, expected_flushes
):
    """Мутація: прибирання `if langfuse_enabled():` навколо flush() у main.py
    ламає варіант keys_present=False (flush_calls стає 1)."""
    if keys_present:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        # Стаб handler'а: справжній CallbackHandler тут не перевіряється.
        monkeypatch.setattr(main, "make_langfuse_handler", lambda: object())
    else:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    fake_client = _FakeLangfuseClient()
    monkeypatch.setattr("langfuse.get_client", lambda: fake_client)

    await _run_screen_stub(monkeypatch, tmp_path, f"thread-flush-{keys_present}")

    assert fake_client.flush_calls == expected_flushes


# --- 5. Модулі імпортуються в чистому середовищі без жодного ключа --------


def test_config_main_langgraph_mas_importuyutsya_bez_kliuchiv():
    """У чистому підпроцесі (без LANGFUSE_*/OPENAI_API_KEY): наявність коду
    трейсингу не ламає імпорт жодного з трьох модулів."""
    import os

    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LANGFUSE_") and key != "OPENAI_API_KEY"
    }

    result = subprocess.run(
        [sys.executable, "-c", "import config, main, langgraph_mas"],
        cwd=str(config.ROOT),
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
