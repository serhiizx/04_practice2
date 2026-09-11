"""Наскрізні прогони зібраного графа LangGraph через ВСІХ трьох агентів,
на стаб-моделі й стаб-інструментах, без мережі.

Чому цей файл потрібен окремо від test_langgraph_wiring.py і
test_human_in_the_loop.py: усі наявні виклики graph.ainvoke там стартують
зі стану, де verdict і email_draft уже покладені руками в тестовому стані —
тому _supervisor_node одразу бачить готовий verdict і віддає "done" ще ДО
першого справжнього кроку, і виконуються лише вузли human_approval/
send_email/report. Жоден наявний тест не проганяв граф з ПОРОЖНЬОГО стану
крізь supervisor -> resume_parser -> requirements_matcher -> communicator.
Саме тому два дефекти пройшли повний цикл рев'ю непоміченими:

- дефект 1: supervisor може повернути next_agent="done" ще до того, як
  communicator сформував email_draft (RouteDecision дозволяє "done" у
  будь-який момент) — _human_approval_node індексував state["email_draft"]
  напряму й падав KeyError;
- дефект 2: якщо fetch_resume постійно повертає помилку, resume_facts
  лишається None назавжди, а лічильник parser_retries раніше інкрементувався
  лише коли resume_facts вже істинний — тож MAX_PARSER_RETRIES ніколи не
  спрацьовував, і граф крутився між supervisor і resume_parser до
  GraphRecursionError.

Обидва тести нижче відтворюють поведінку ДО фіксу (падали б на попередньому
коді) і залишаються регресійним запобіжником на майбутнє.
"""

import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

import langgraph_mas
from conftest import StubLLM, StubReactAgent, StubTool, stub_react_agent_with_text
from schemas import EmailDraft, ResumeFacts, RouteDecision, ScreeningVerdict

CANDIDATE_ID = "CAND-001"
JOB_ID = "JOB-BACKEND"


def _tool_message(name: str, payload: dict) -> ToolMessage:
    return ToolMessage(
        content=[{"type": "text", "text": json.dumps(payload)}],
        name=name,
        tool_call_id="1",
    )


def _base_tools(fetch_resume_result: dict) -> list[StubTool]:
    return [
        StubTool("fetch_resume", result=fetch_resume_result),
        StubTool("fetch_job_requirements", result={"status": "ok", "data": {}}),
        StubTool("score_candidate"),
        StubTool("send_candidate_email", result={"status": "ok", "data": {"to": "cand@example.com"}}),
    ]


# --- 1. Повний happy path: усі три агенти реально виконуються --------------


async def test_povnyi_shlyah_vid_porozhnyoho_stanu_do_zvitu(tmp_path, monkeypatch):
    """supervisor -> resume_parser -> requirements_matcher -> communicator ->
    human_approval -> send_email -> report. Router-стаб віддає рішення по
    порядку, requirements_matcher — реальний виклик ReAct-агента (тут
    підмінений стабом), решта вузлів — справжня логіка langgraph_mas.py."""
    tools = _base_tools({
        "status": "ok",
        "data": {"candidate_id": CANDIDATE_ID, "full_name": "Т. Т.", "resume_text": "Досвід: 5 років Python."},
    })

    score_messages = [
        _tool_message("score_candidate", {
            "status": "ok",
            "data": {
                "score": 80, "decision": "strong_match",
                "matched_must_have": ["Python"], "missing_must_have": [],
                "matched_nice_to_have": [], "meets_min_years": True,
            },
        }),
        AIMessage(content="Готово."),
    ]
    react_agent = StubReactAgent(score_messages)

    stub_llm = StubLLM({
        RouteDecision: [
            RouteDecision(next_agent="resume_parser", reason="фактів ще немає"),
            RouteDecision(next_agent="requirements_matcher", reason="факти є, скорингу немає"),
            RouteDecision(next_agent="communicator", reason="скоринг є, вердикту немає"),
        ],
        ResumeFacts: ResumeFacts(skills=["Python"], years_experience=5, education="", location="Kyiv"),
        ScreeningVerdict: ScreeningVerdict(
            candidate_id=CANDIDATE_ID, job_id=JOB_ID, score=80,
            decision="strong_match", rationale="Підходить.", gaps=[],
        ),
        EmailDraft: EmailDraft(
            candidate_id=CANDIDATE_ID, decision="strong_match",
            subject="Вітаємо!", body="Вітаємо з успішним скринінгом.",
        ),
    })

    def _fake_create_react_agent(llm, node_tools, **_kw):
        assert llm is stub_llm  # requirements_matcher отримав ту саму LLM
        return react_agent

    monkeypatch.setattr(langgraph_mas, "create_react_agent", _fake_create_react_agent)

    config = {"configurable": {"thread_id": "e2e-happy"}, "recursion_limit": 25}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as cp:
        graph = await langgraph_mas.build_graph(tools, checkpointer=cp, llm=stub_llm)
        state = await graph.ainvoke(
            {"candidate_id": CANDIDATE_ID, "job_id": JOB_ID, "messages": []}, config,
        )
        assert "__interrupt__" in state  # дійшли до HITL — усі три агенти відпрацювали

        final_state = await graph.ainvoke(Command(resume={"action": "approve"}), config)

    assert "__interrupt__" not in final_state
    assert final_state["resume_facts"]["skills"] == ["Python"]
    assert final_state["score"]["score"] == 80
    assert final_state["verdict"]["score"] == 80
    assert final_state["verdict"]["decision"] == "strong_match"
    assert final_state.get("report")
    assert CANDIDATE_ID in final_state["report"]


# --- 2. Дефект 1: done без чернетки листа -----------------------------------


async def test_defekt1_done_bez_chernetky_zavershuye_kerovanym_zvitom_a_ne_padaye(tmp_path, monkeypatch):
    """RouteDecision.next_agent="done" дозволений у будь-який момент — навіть
    коли жоден вузол (resume_parser/requirements_matcher/communicator) ще не
    виконався й email_draft у стані немає. Граф має дійти до звіту, який
    чесно каже, що вердикту й чернетки немає, а не впасти на
    state["email_draft"]."""
    monkeypatch.setattr(
        langgraph_mas, "create_react_agent",
        lambda *_a, **_kw: stub_react_agent_with_text("stub"),
    )
    tools = _base_tools({"status": "ok", "data": {}})
    stub_llm = StubLLM({RouteDecision: RouteDecision(
        next_agent="done", reason="маленька модель помилилась і одразу завершила"
    )})

    config = {"configurable": {"thread_id": "e2e-defekt1"}, "recursion_limit": 25}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as cp:
        graph = await langgraph_mas.build_graph(tools, checkpointer=cp, llm=stub_llm)
        final_state = await graph.ainvoke(
            {"candidate_id": CANDIDATE_ID, "job_id": JOB_ID, "messages": []}, config,
        )  # не мало кинути KeyError('email_draft')

    assert "__interrupt__" not in final_state
    assert final_state.get("email_draft") is None
    assert final_state.get("verdict") is None
    assert final_state.get("report")
    assert "чернетк" in final_state["report"].lower()


# --- 3. Дефект 2: fetch_resume завжди помиляється (невідомий кандидат) -----


async def test_defekt2_postiyna_pomylka_fetch_resume_zavershuyetsya_bez_recursion_error(tmp_path, monkeypatch):
    """fetch_resume постійно повертає помилку (невідомий candidate_id) —
    resume_facts лишається None назавжди. Router-стаб навмисно "впертий":
    завжди відповідає resume_parser, як помилилась би маленька локальна
    модель. Граф має завершитись за скінченну кількість кроків (захист
    MAX_PARSER_RETRIES спрацьовує і веде на done -> report), а не впертись
    у recursion_limit."""
    monkeypatch.setattr(
        langgraph_mas, "create_react_agent",
        lambda *_a, **_kw: stub_react_agent_with_text("stub"),
    )
    tools = _base_tools({"status": "error", "error": "кандидата CAND-999 немає в базі"})
    stub_llm = StubLLM({RouteDecision: RouteDecision(
        next_agent="resume_parser", reason="спробуємо ще раз"
    )})

    config = {"configurable": {"thread_id": "e2e-defekt2"}, "recursion_limit": 25}
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "state.db")) as cp:
        graph = await langgraph_mas.build_graph(tools, checkpointer=cp, llm=stub_llm)
        final_state = await graph.ainvoke(
            {"candidate_id": "CAND-999", "job_id": JOB_ID, "messages": []}, config,
        )  # не мало кинути GraphRecursionError

    assert "__interrupt__" not in final_state
    assert final_state.get("resume_facts") is None
    assert final_state.get("parser_retries", 0) >= langgraph_mas.MAX_PARSER_RETRIES
    assert final_state.get("report")
