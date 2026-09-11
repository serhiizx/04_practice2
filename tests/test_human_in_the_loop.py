"""Офлайн-тести human-in-the-loop: граф зупиняється ПЕРЕД надсиланням листа
через interrupt(), а approve/reject/edit — операції над станом у
checkpointer (AsyncSqliteSaver у тимчасовій базі), без мережі й без LLM.

Ризиковий інструмент send_candidate_email виконується по-справжньому
(домен mcp_server.send_candidate_email, обгорнутий у формат MCP-адаптера) —
це доводить, що лист і справді (не) з'являється в outbox.json, а не лише
що граф "викликав щось з правильним іменем". OUTBOX_PATH підмінено на
тимчасовий файл, тому робочий data/outbox.json не чіпається. supervisor і
requirements_matcher до моделі не звертаються: verdict уже в стані, тож
supervisor одразу віддає "done" (langgraph_mas._supervisor_node), а
create_react_agent підмінено стабом, щоб конструювання графа не вимагало
справжнього llm.bind_tools()."""

import json

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

import langgraph_mas
import mcp_server
from conftest import StubLLM, StubTool, stub_react_agent_with_text


class RealSendEmailTool:
    """Справжній mcp_server.send_candidate_email, обгорнутий так, як його
    віддає langchain_mcp_adapters (список content-блоків), щоб _send_email_node
    міг розпарсити результат через _tool_result_text, як і в реальному графі."""

    name = "send_candidate_email"

    def __init__(self):
        self.calls: list[dict] = []

    async def ainvoke(self, args: dict):
        self.calls.append(args)
        result = mcp_server.send_candidate_email(**args)
        return [{"type": "text", "text": json.dumps(result)}]


CAND_ID = "CAND-004"
JOB_ID = "JOB-BACKEND"
ORIGINAL_BODY = "Вітаємо, ви пройшли скринінг на позицію Backend Engineer!"
EDITED_BODY = "Дякуємо за співбесіду, ми ще розглядаємо ваше рішення."


def _initial_state() -> dict:
    """Стан, з яким запускається граф: verdict і email_draft вже готові —
    саме так виглядає стан на момент, коли supervisor бачить готовий вердикт
    (langgraph_mas._supervisor_node: `if state.get("verdict"): next_agent="done"`)
    і router (LLM) навіть не викликається."""
    return {
        "candidate_id": CAND_ID,
        "job_id": JOB_ID,
        "messages": [],
        "verdict": {
            "candidate_id": CAND_ID,
            "job_id": JOB_ID,
            "score": 93,
            "decision": "strong_match",
            "rationale": "Сильний кандидат, усі must-have закриті.",
            "gaps": [],
            "injection_detected": False,
        },
        "email_draft": {
            "candidate_id": CAND_ID,
            "decision": "strong_match",
            "subject": "Вітаємо з успішним скринінгом!",
            "body": ORIGINAL_BODY,
        },
    }


def _tools(send_tool: RealSendEmailTool) -> list:
    return [
        StubTool("fetch_resume"),
        StubTool("fetch_job_requirements"),
        StubTool("score_candidate"),
        send_tool,
    ]


@pytest.fixture(autouse=True)
def _stub_react_agent(monkeypatch):
    """requirements_matcher до моделі в цих тестах не доходить, але вузол
    конструюється завжди — create_react_agent(StubLLM, ...) з реальним
    llm.bind_tools() без мережі впав би вже на етапі побудови графа."""
    monkeypatch.setattr(
        langgraph_mas, "create_react_agent",
        lambda *_a, **_kw: stub_react_agent_with_text("stub"),
    )


async def _build(tmp_path, send_tool: RealSendEmailTool, checkpointer):
    return await langgraph_mas.build_graph(
        _tools(send_tool), checkpointer=checkpointer, llm=StubLLM({})
    )


# --- 1. Граф зупиняється перед надсиланням ----------------------------------


async def test_hraf_zupynyayetsya_pered_nadsylannyam(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "OUTBOX_PATH", tmp_path / "outbox.json")
    send_tool = RealSendEmailTool()
    config = {"configurable": {"thread_id": "interrupt"}}

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as cp:
        graph = await _build(tmp_path, send_tool, cp)
        state = await graph.ainvoke(_initial_state(), config)

    assert "__interrupt__" in state
    assert send_tool.calls == []
    assert not (tmp_path / "outbox.json").exists()

    payload = state["__interrupt__"][0].value
    # Людина повинна бачити адресата (candidate_id), рішення, тему й ПОВНИЙ
    # текст листа — саме те, що вона схвалює.
    assert payload["candidate_id"] == CAND_ID
    assert payload["decision"] == "strong_match"
    assert payload["subject"] == "Вітаємо з успішним скринінгом!"
    assert payload["body"] == ORIGINAL_BODY


# --- 2. approve ---------------------------------------------------------


async def test_approve_klykaye_nadsylannya_rivno_odyn_raz_z_tym_zhe_tekstom(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "OUTBOX_PATH", tmp_path / "outbox.json")
    send_tool = RealSendEmailTool()
    config = {"configurable": {"thread_id": "approve"}}

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as cp:
        graph = await _build(tmp_path, send_tool, cp)
        interrupted = await graph.ainvoke(_initial_state(), config)
        assert "__interrupt__" in interrupted

        final_state = await graph.ainvoke(Command(resume={"action": "approve"}), config)

    assert len(send_tool.calls) == 1
    assert send_tool.calls[0]["body"] == ORIGINAL_BODY
    assert send_tool.calls[0]["subject"] == "Вітаємо з успішним скринінгом!"

    outbox = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    assert len(outbox) == 1
    assert outbox[0]["body"] == ORIGINAL_BODY
    assert outbox[0]["to"]  # доставлено на адресу кандидата з бази
    assert final_state.get("report")


# --- 3. reject -----------------------------------------------------------


async def test_reject_ne_klykaye_nadsylannya_ale_dohodyt_do_zvitu(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "OUTBOX_PATH", tmp_path / "outbox.json")
    send_tool = RealSendEmailTool()
    config = {"configurable": {"thread_id": "reject"}}

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as cp:
        graph = await _build(tmp_path, send_tool, cp)
        interrupted = await graph.ainvoke(_initial_state(), config)
        assert "__interrupt__" in interrupted

        final_state = await graph.ainvoke(Command(resume={"action": "reject"}), config)

    assert send_tool.calls == []
    assert not (tmp_path / "outbox.json").exists()
    # Причина відмови (яке саме рішення прийняла людина) лишається у стані.
    assert final_state["next_agent"] == "reject"
    assert final_state.get("report")


# --- 4. edit ---------------------------------------------------------------


async def test_edit_nadsylaye_vypravlenyi_tekst_a_ne_pervynnu_chernetku(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "OUTBOX_PATH", tmp_path / "outbox.json")
    send_tool = RealSendEmailTool()
    config = {"configurable": {"thread_id": "edit"}}

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as cp:
        graph = await _build(tmp_path, send_tool, cp)
        interrupted = await graph.ainvoke(_initial_state(), config)
        assert "__interrupt__" in interrupted

        final_state = await graph.ainvoke(
            Command(resume={"action": "edit", "body": EDITED_BODY}), config
        )

    assert len(send_tool.calls) == 1
    assert send_tool.calls[0]["body"] == EDITED_BODY
    assert final_state["email_draft"]["body"] == EDITED_BODY

    outbox = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    assert outbox[0]["body"] == EDITED_BODY
    assert EDITED_BODY != ORIGINAL_BODY  # переконуємось, що тест не тавтологічний


# --- 5. Звіт проходить output guardrail -------------------------------------


def test_zvit_maskuye_pii_kandydata_cand_004():
    """_build_report — чиста функція, не потребує графа/checkpointer. Тест
    навмисно кладе справжні PII-значення CAND-004 (data/candidates.json) у
    вердикт і чернетку листа, щоб перевірити, що output guardrail (redact_pii)
    їх маскує в фінальному звіті."""
    tax_id = "3214567890"
    phone = "+380671234567"
    dob = "12.04.1990"
    email = "n.bondarenko@example.com"

    state = {
        "verdict": {
            "candidate_id": CAND_ID,
            "job_id": JOB_ID,
            "score": 93,
            "decision": "strong_match",
            "rationale": (
                f"Кандидатка (ІПН: {tax_id}, дата народження: {dob}) підходить."
            ),
            "gaps": [],
            "injection_detected": False,
        },
        "email_draft": {
            "candidate_id": CAND_ID,
            "decision": "strong_match",
            "subject": "Вітаємо!",
            "body": f"Зв'яжемось за телефоном {phone} або на {email}.",
        },
    }

    report = langgraph_mas._build_report(state)

    assert tax_id not in report
    assert phone not in report
    assert dob not in report
    assert email not in report
    assert "[PII:TAXID]" in report
    assert "[PII:PHONE]" in report
    assert "[PII:DOB]" in report
    assert "[PII:EMAIL]" in report


# --- 6. Відновлення стану: checkpointer, а не просто "заблокований" граф ---


async def test_stan_pislya_zupynky_chytayetsya_z_checkpointer(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_server, "OUTBOX_PATH", tmp_path / "outbox.json")
    send_tool = RealSendEmailTool()
    config = {"configurable": {"thread_id": "resumable"}}

    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as cp:
        graph = await _build(tmp_path, send_tool, cp)
        interrupted = await graph.ainvoke(_initial_state(), config)
        assert "__interrupt__" in interrupted

        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("human_approval",)
        assert snapshot.values["candidate_id"] == CAND_ID
        assert snapshot.values["verdict"]["decision"] == "strong_match"
        assert snapshot.values["email_draft"]["subject"] == "Вітаємо з успішним скринінгом!"

        # Доводимо, що це справді відновлюваний прогін, а не заблокований:
        # відновлення тим самим checkpointer'ом і thread_id доводить його до кінця.
        final_state = await graph.ainvoke(Command(resume={"action": "approve"}), config)

    assert "__interrupt__" not in final_state
    assert final_state.get("report")
    assert len(send_tool.calls) == 1


# --- 7. Відновлення після справжнього закриття й переоткриття checkpointer -

# Тест №6 вище доводить лише, що aget_state читає з checkpointer, а не з
# in-memory стану графа — але відновлює прогін у тому самому контекстному
# менеджері, на тому самому об'єкті AsyncSqliteSaver. Це не доводить
# відновлення "після перезапуску процесу": весь процес pytest тут не
# перезапускається, і такого способу з тестів немає. Що можна й треба
# довести — це те, що saver не тримає стан у пам'яті об'єкта: закриваємо
# checkpointer (async with виходить), відкриваємо його ЗНОВУ на тому самому
# файлі бази (нове з'єднання, новий об'єкт AsyncSqliteSaver, новий граф) — і
# відновлюємо той самий thread_id. Якби стан жив у пам'яті об'єкта, а не в
# самій sqlite-базі, друге відкриття його б не побачило.


async def test_stan_vidnovlyuyetsya_pislya_spravzhnyoho_perevidkryttya_checkpointera(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(mcp_server, "OUTBOX_PATH", tmp_path / "outbox.json")
    db_path = str(tmp_path / "state.db")
    config = {"configurable": {"thread_id": "restart"}}

    # Перше "з'єднання": граф зупиняється на interrupt, checkpointer
    # закривається (async with виходить) разом із графом і send_tool.
    async with AsyncSqliteSaver.from_conn_string(db_path) as cp:
        graph = await _build(tmp_path, RealSendEmailTool(), cp)
        interrupted = await graph.ainvoke(_initial_state(), config)
        assert "__interrupt__" in interrupted

    # Друге "з'єднання": новий AsyncSqliteSaver на тому самому файлі, новий
    # граф, новий send_tool — жодного об'єкта з першого блоку тут не видно.
    new_send_tool = RealSendEmailTool()
    async with AsyncSqliteSaver.from_conn_string(db_path) as cp2:
        graph2 = await _build(tmp_path, new_send_tool, cp2)

        snapshot = await graph2.aget_state(config)
        assert snapshot.next == ("human_approval",)
        assert snapshot.values["email_draft"]["subject"] == "Вітаємо з успішним скринінгом!"

        final_state = await graph2.ainvoke(Command(resume={"action": "approve"}), config)

    assert "__interrupt__" not in final_state
    assert len(new_send_tool.calls) == 1
    outbox = json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
    assert outbox[0]["body"] == ORIGINAL_BODY
