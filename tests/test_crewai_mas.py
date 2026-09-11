"""Офлайн-тести CrewAI-оркестрації: без мережі, без ключів API, без
підняття справжнього MCP-підпроцесу (MCPServerAdapter не використовується
в жодному тесті — build_crew приймає готовий список інструментів напряму).

1. _tools_for — той самий allowlist для всіх трьох ролей.
2. run_crew пропускає підсумок crew.kickoff() через redact_pii.
3. Інструментація Langfuse вмикається лише за наявності ключів і не
   імпортує langfuse/openinference на рівні модуля.
4. crewai_mas імпортується в чистому підпроцесі без жодного ключа.
5. build_crew читає рівно ті самі файли prompts/*.md, що й LangGraph —
   жодного вшитого тексту замість load_prompt().
"""

import ast
import subprocess
import sys
from pathlib import Path

import crewai_mas
from conftest import StubTool
from crewai.tools import BaseTool


class _StubCrewTool(BaseTool):
    """Мінімальний BaseTool-сумісний двійник: crewai.Agent валідує tools
    через pydantic і відхиляє об'єкт, що не є інстансом BaseTool (на
    відміну від LangChain-адаптера, тут голого .name недостатньо)."""

    name: str
    description: str = "стаб-інструмент для тестів"

    def _run(self, *_args, **_kwargs) -> str:
        return "стаб"


def _crew_tools() -> list[_StubCrewTool]:
    return [
        _StubCrewTool(name=n)
        for n in ("fetch_resume", "fetch_job_requirements", "score_candidate", "send_candidate_email")
    ]


# --- 1. Фільтрація інструментів за allowlist --------------------------------


def test_tools_for_povertaye_lyshe_dozvoleni_instrumenty():
    tools = [
        StubTool("fetch_resume"),
        StubTool("fetch_job_requirements"),
        StubTool("score_candidate"),
        StubTool("send_candidate_email"),
    ]

    resume_parser_tools = {t.name for t in crewai_mas._tools_for("resume_parser", tools)}
    matcher_tools = {t.name for t in crewai_mas._tools_for("requirements_matcher", tools)}
    communicator_tools = {t.name for t in crewai_mas._tools_for("communicator", tools)}

    assert resume_parser_tools == {"fetch_resume"}
    assert matcher_tools == {"fetch_job_requirements", "score_candidate"}
    assert communicator_tools == {"send_candidate_email"}


def test_build_crew_kozhen_agent_otrymuye_lyshe_svoyi_instrumenty():
    """Наскрізна перевірка: не лише _tools_for як чиста функція, а й те, що
    Agent у зібраній Crew реально отримав відфільтрований список."""
    crew = crewai_mas.build_crew(_crew_tools(), "CAND-001", "JOB-BACKEND")

    by_role = {agent.role: {t.name for t in agent.tools} for agent in crew.agents}

    assert by_role["ResumeParser"] == {"fetch_resume"}
    assert by_role["RequirementsMatcher"] == {"fetch_job_requirements", "score_candidate"}
    # ВАЖЛИВО для порівняння з LangGraph: тут Communicator отримує ризиковий
    # send_candidate_email напряму як власний інструмент. У LangGraph
    # (langgraph_mas.py, _communicator_node) цей інструмент НЕ передається
    # жодному агенту взагалі — його викликає граф лише після interrupt на
    # human_approval. Тут єдиний захист — human_input=True на verdict_task
    # (консольне підтвердження ПІСЛЯ виконання задачі) плюс текст промпту;
    # агент технічно спроможний викликати інструмент до того, як людина
    # побачить запит на підтвердження (докладніше — README, розділ 7).
    assert by_role["Communicator"] == {"send_candidate_email"}


def test_build_crew_agenty_otrymuyut_tu_samu_model_shcho_i_langgraph(monkeypatch):
    """Без явного llm= CrewAI читає MODEL/MODEL_NAME/OPENAI_MODEL_NAME (не
    LLM_MODEL, яку задає .env.example) — і з LM Studio запит пішов би на
    неіснуючу модель. build_crew має в'язати кожного агента саме до
    LLM_MODEL/OPENAI_BASE_URL/OPENAI_API_KEY, як і config.make_llm."""
    monkeypatch.setenv("LLM_MODEL", "google/gemma-4-e4b")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio-any-string")

    crew = crewai_mas.build_crew(_crew_tools(), "CAND-001", "JOB-BACKEND")

    for agent in crew.agents:
        assert agent.llm.model == "google/gemma-4-e4b"
        assert agent.llm.base_url == "http://127.0.0.1:1234/v1"


def test_build_crew_bez_ovveridiv_env_bere_default_model(monkeypatch):
    """Без LLM_MODEL у .env — той самий дефолт, що й у config.make_llm."""
    monkeypatch.delenv("LLM_MODEL", raising=False)

    crew = crewai_mas.build_crew(_crew_tools(), "CAND-001", "JOB-BACKEND")

    assert all(agent.llm.model == "gpt-4o-mini" for agent in crew.agents)


# --- 2. run_crew маскує PII у фінальному звіті ------------------------------


class _StubAdapterCM:
    """Двійник контекстного менеджера MCPServerAdapter: не піднімає
    підпроцес, просто віддає готовий список інструментів."""

    def __init__(self, tools):
        self._tools = tools

    def __enter__(self):
        return self._tools

    def __exit__(self, *_exc):
        return False


class _StubCrew:
    """Двійник Crew: kickoff() повертає наперед заданий текст без жодного
    виклику LLM чи MCP-сервера."""

    def __init__(self, text: str):
        self._text = text

    def kickoff(self):
        return self._text


def test_run_crew_maskuye_pii_kandydata_cand_004(monkeypatch):
    # Персональні дані CAND-004 (data/candidates.json), як їх міг би
    # повернути crew.kickoff(), якби модель процитувала резюме у звіті.
    text_z_pii = (
        "Кандидат CAND-004 Наталія Бондаренко.\n"
        "Score: 93 -> strong_match.\n"
        "Контакт: n.bondarenko@example.com, телефон +380671234567.\n"
        "Дата народження: 12.04.1990. ІПН: 3214567890.\n"
        "Адреса: вул. Хрещатик, 22, Київ."
    )

    monkeypatch.setattr(crewai_mas, "MCPServerAdapter", lambda _params: _StubAdapterCM([]))
    monkeypatch.setattr(crewai_mas, "build_crew", lambda *_a, **_kw: _StubCrew(text_z_pii))

    report = crewai_mas.run_crew("CAND-004", "JOB-BACKEND")

    # Сирі персональні дані не мають пройти в підсумок.
    assert "n.bondarenko@example.com" not in report
    assert "+380671234567" not in report
    assert "12.04.1990" not in report
    assert "3214567890" not in report
    assert "вул. Хрещатик, 22, Київ" not in report

    # А маски — мають.
    assert "[PII:EMAIL]" in report
    assert "[PII:PHONE]" in report
    assert "[PII:DOB]" in report
    assert "[PII:TAXID]" in report
    assert "[PII:ADDRESS]" in report
    assert "Замасковано PII" in report

    # Незасекречена частина звіту (score/вердикт) лишається читаною.
    assert "CAND-004" in report
    assert "strong_match" in report


def test_run_crew_bez_pii_ne_dodaye_pomitku(monkeypatch):
    text_bez_pii = "Кандидат CAND-001. Score: 62 -> maybe. Без персональних даних."

    monkeypatch.setattr(crewai_mas, "MCPServerAdapter", lambda _params: _StubAdapterCM([]))
    monkeypatch.setattr(crewai_mas, "build_crew", lambda *_a, **_kw: _StubCrew(text_bez_pii))

    report = crewai_mas.run_crew("CAND-001", "JOB-BACKEND")

    assert "Замасковано PII" not in report
    assert "maybe" in report


# --- 3. Інструментація трейсингу лише за наявності ключів ------------------


def test_config_ne_importuye_langfuse_ani_openinference_na_rivni_modulya():
    """Статичний капкан (той самий підхід, що й test_langfuse_tracing.py
    для config.py): жодного import langfuse/openinference серед
    інструкцій РІВНЯ МОДУЛЯ crewai_mas.py."""
    source = Path(crewai_mas.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    top_level_module_names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_module_names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_module_names.append(node.module)

    assert not any(
        name.startswith("langfuse") or name.startswith("openinference")
        for name in top_level_module_names
    )


def test_instrument_crewai_nichoho_ne_robyt_bez_kliuchiv(monkeypatch):
    """Без ключів Langfuse _instrument_crewai() не сміє навіть спробувати
    імпортувати важкі модулі трейсингу — перевіряємо це через "детектор
    падіння": якщо get_client чи CrewAIInstrumentor раптом викликані,
    стаб кидає AssertionError і тест провалюється."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    def _boom(*_a, **_kw):
        raise AssertionError("get_client не мав викликатись без ключів Langfuse")

    monkeypatch.setattr("langfuse.get_client", _boom)
    monkeypatch.setattr(
        "openinference.instrumentation.crewai.CrewAIInstrumentor.instrument",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            AssertionError("CrewAIInstrumentor.instrument не мав викликатись без ключів")
        ),
    )

    crewai_mas._instrument_crewai()  # не мало кинути жодного AssertionError


def test_instrument_crewai_klykaye_get_client_i_instrumentor_koly_kliuchi_ye(monkeypatch):
    """З ключами — обидва кроки інструментації відпрацьовують рівно раз."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    calls = {"get_client": 0, "instrument": 0}
    monkeypatch.setattr("langfuse.get_client", lambda: calls.__setitem__("get_client", calls["get_client"] + 1))
    monkeypatch.setattr(
        "openinference.instrumentation.crewai.CrewAIInstrumentor.instrument",
        lambda self, **_kw: calls.__setitem__("instrument", calls["instrument"] + 1),
    )

    crewai_mas._instrument_crewai()

    assert calls == {"get_client": 1, "instrument": 1}


# --- 4. Модуль імпортується в чистому підпроцесі без жодного ключа ---------


def test_crewai_mas_importuyetsya_bez_kliuchiv():
    clean_env = {
        key: value
        for key, value in __import__("os").environ.items()
        if not key.startswith("LANGFUSE_") and key != "OPENAI_API_KEY"
    }

    result = subprocess.run(
        [sys.executable, "-c", "import crewai_mas"],
        cwd=str(crewai_mas.ROOT),
        env=clean_env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


# --- 5. build_crew читає рівно ті самі промпти, що й LangGraph --------------


def test_build_crew_chytaye_ti_sami_faily_promptiv(monkeypatch):
    """Захист від тихого розходження двох реалізацій: якщо хтось замінить
    load_prompt("communicator") на вшитий рядок прямо в crewai_mas.py, цей
    тест провалиться — і на кількості викликів load_prompt, і на вмісті
    backstory агентів."""
    calls: list[str] = []
    original_load_prompt = crewai_mas.load_prompt

    def _spy(name: str) -> str:
        calls.append(name)
        return original_load_prompt(name)

    monkeypatch.setattr(crewai_mas, "load_prompt", _spy)

    crew = crewai_mas.build_crew(_crew_tools(), "CAND-001", "JOB-BACKEND")

    assert calls == ["resume_parser", "requirements_matcher", "communicator"]

    backstories = {agent.role: agent.backstory for agent in crew.agents}
    assert backstories["ResumeParser"] == original_load_prompt("resume_parser")
    assert backstories["RequirementsMatcher"] == original_load_prompt("requirements_matcher")
    assert backstories["Communicator"] == original_load_prompt("communicator")
