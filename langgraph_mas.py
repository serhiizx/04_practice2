"""MAS HR-скринінгу в LangGraph: supervisor + три агенти.

Supervisor — LLM-маршрутизатор зі structured output, агенти — ReAct-підграфи
з власними наборами інструментів згідно з allowlist. Домен живе в MCP-сервері,
цей модуль відповідає лише за оркестрацію.
"""

import json
import sys
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt

from config import ROOT, load_prompt, make_llm
from guardrails import AGENT_TOOL_ALLOWLIST, ToolDenied, check_tool_call, detect_injection, redact_pii
from schemas import EmailDraft, ResumeFacts, RouteDecision, ScreeningVerdict

# MCP-сервер піднімається як підпроцес того самого інтерпретатора.
MCP_SERVER_CONFIG = {
    "hr": {
        "command": sys.executable,
        "args": [str(ROOT / "mcp_server.py")],
        "transport": "stdio",
    }
}

MAX_PARSER_RETRIES = 2  # захист від зациклення supervisor → parser → supervisor


class ScreeningState(TypedDict, total=False):
    """Стан скринінгу, спільний для всіх вузлів графа."""

    candidate_id: str
    job_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    resume_facts: dict | None
    score: dict | None
    injection_detected: bool
    injection_patterns: list[str]
    parser_retries: int
    verdict: dict | None
    email_draft: dict | None
    next_agent: str
    report: str


async def load_mcp_tools() -> tuple[MultiServerMCPClient, list[BaseTool]]:
    """Підключається до власного MCP-сервера і віддає його інструменти."""
    client = MultiServerMCPClient(MCP_SERVER_CONFIG)
    tools = await client.get_tools()
    return client, tools


def tools_for(agent: str, tools: list[BaseTool]) -> list[BaseTool]:
    """Інструменти, дозволені агенту. Allowlist на рівні розводки графа;
    check_tool_call лишається другим рубежем у рантаймі."""
    allowed = AGENT_TOOL_ALLOWLIST[agent]
    return [t for t in tools if t.name in allowed]


def _tool_result_text(content) -> str:
    """Дістає текст із результату MCP-інструмента.

    langchain_mcp_adapters повертає не рядок, а список content-блоків
    [{"type": "text", "text": "..."}] — і в прямому ainvoke, і в ToolMessage."""
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def _score_from_tool_messages(messages) -> dict | None:
    """Дістає результат score_candidate з ToolMessage історії агента, а НЕ з
    вільного тексту фінальної відповіді — той може бути спотворений.

    Бере ОСТАННЄ успішне повідомлення: агент міг схибити (невалідні
    аргументи) і повторити виклик уже успішно."""
    result = None
    for msg in messages:
        if getattr(msg, "name", None) != "score_candidate":
            continue
        try:
            payload = json.loads(_tool_result_text(msg.content))
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("status") == "ok":
            result = payload.get("data")
    return result


def _supervisor_node(llm):
    """Вузол маршрутизації. Рішення ухвалює LLM через structured output."""
    router = llm.with_structured_output(RouteDecision)
    system = load_prompt("supervisor")

    async def node(state: ScreeningState) -> dict:
        if state.get("verdict"):
            return {"next_agent": "done"}

        # Захист від зациклення: після ліміту йдемо далі з тим, що є. Якщо
        # resume_facts так і немає (fetch_resume постійно падав) — ще один
        # цикл нічого не додасть, а matcher без жодного факту безглуздий;
        # завершуємо контрольовано, а не крутимось до GraphRecursionError.
        if state.get("parser_retries", 0) >= MAX_PARSER_RETRIES and not state.get("score"):
            if state.get("resume_facts") is not None:
                return {"next_agent": "requirements_matcher"}
            return {"next_agent": "done"}

        summary = {
            "candidate_id": state["candidate_id"],
            "job_id": state["job_id"],
            "resume_facts": state.get("resume_facts"),
            "score": state.get("score"),
            "parser_retries": state.get("parser_retries", 0),
        }
        decision = await router.ainvoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Поточний стан:\n{json.dumps(summary, ensure_ascii=False, indent=2)}"},
            ]
        )
        update: dict = {"next_agent": decision.next_agent}
        # Інкремент при БУДЬ-ЯКОМУ поверненні на resume_parser: інакше, якщо
        # fetch_resume постійно падає (resume_facts назавжди None), лічильник
        # не росте й захист вище ніколи не вмикається.
        if decision.next_agent == "resume_parser":
            update["parser_retries"] = state.get("parser_retries", 0) + 1
        return update

    return node


def _resume_parser_node(llm, tools):
    """Дістає резюме напряму інструментом, без ReAct-циклу — навмисно.

    У ReAct-циклі модель першою бачить сирий текст fetch_resume: ін'єкція
    встигає подіяти ДО detect_injection, а той перевіряє вже переказ моделі,
    а не оригінал. Тут модель бачить лише verdict.safe_text. Витяг
    структурованих фактів лишається за LLM (with_structured_output)."""
    fetch_resume = next(t for t in tools_for("resume_parser", tools) if t.name == "fetch_resume")
    extractor = llm.with_structured_output(ResumeFacts)

    async def node(state: ScreeningState) -> dict:
        try:
            args = check_tool_call(
                "resume_parser", "fetch_resume", {"candidate_id": state["candidate_id"]}
            )
        except ToolDenied as exc:
            return {
                "resume_facts": None,
                "messages": [HumanMessage(
                    content=f"ResumeParser: виклик fetch_resume відхилено guardrail'ом: {exc}"
                )],
            }

        raw_result = await fetch_resume.ainvoke(args)
        payload = json.loads(_tool_result_text(raw_result))
        if payload.get("status") != "ok":
            return {
                "resume_facts": None,
                "messages": [HumanMessage(
                    content=f"ResumeParser: fetch_resume повернув помилку: {payload.get('error')}"
                )],
            }

        verdict = detect_injection(payload["data"]["resume_text"])
        facts = await extractor.ainvoke(
            [
                {"role": "system", "content": load_prompt("resume_parser")},
                {"role": "user", "content": f"Витягни факти з резюме:\n{verdict.safe_text}"},
            ]
        )
        return {
            "resume_facts": facts.model_dump(),
            "injection_detected": verdict.detected,
            "injection_patterns": verdict.patterns,
            "messages": [HumanMessage(content=f"ResumeParser: {facts.model_dump_json()}")],
        }

    return node


def _requirements_matcher_node(llm, tools):
    """ReAct-агент, що тягне вимоги вакансії й рахує score інструментом."""
    agent = create_react_agent(
        llm,
        tools_for("requirements_matcher", tools),
        prompt=load_prompt("requirements_matcher"),
    )

    async def node(state: ScreeningState) -> dict:
        facts = state.get("resume_facts") or {}
        if not facts.get("skills"):
            return {
                "messages": [HumanMessage(
                    content="RequirementsMatcher: у фактах немає навичок, потрібен повторний парсинг."
                )]
            }

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=(
                f"Вакансія: {state['job_id']}. "
                f"Навички кандидата: {facts['skills']}. "
                f"Роки досвіду: {facts['years_experience']}. "
                f"Виклич fetch_job_requirements, потім score_candidate."
            ))]}
        )
        # Score береться з ToolMessage самого виклику score_candidate, а НЕ з
        # вільного тексту фінальної відповіді агента — той лише переказ, який
        # модель могла спотворити (regex по тексту довіряв би цьому переказу).
        score = _score_from_tool_messages(result["messages"])
        final_text = result["messages"][-1].content
        return {
            "score": score,
            "messages": [HumanMessage(content=f"RequirementsMatcher: {final_text}")],
        }

    return node


def _communicator_node(llm):
    """Формує вердикт і чернетку листа. Інструмент надсилання НЕ викликає —
    це робить граф після схвалення людиною."""
    verdict_llm = llm.with_structured_output(ScreeningVerdict)
    email_llm = llm.with_structured_output(EmailDraft)

    async def node(state: ScreeningState) -> dict:
        score = state.get("score") or {}
        context = {
            "candidate_id": state["candidate_id"],
            "job_id": state["job_id"],
            "score": score,
            "resume_facts": state.get("resume_facts"),
            "injection_detected": state.get("injection_detected", False),
        }
        system = load_prompt("communicator")
        payload = json.dumps(context, ensure_ascii=False, indent=2)

        verdict = await verdict_llm.ainvoke(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"Сформуй вердикт:\n{payload}"}]
        )
        # Score береться з інструмента, а не з моделі — переписати його
        # ін'єкцією неможливо. Якщо авторитетного значення немає, це має бути
        # ВИДНО: позначаємо вердикт неповним і відкочуємось до "reject",
        # а не тихо беремо число, яке вигадала модель.
        has_authoritative_score = score.get("score") is not None and score.get("decision") is not None
        if has_authoritative_score:
            verdict.score = score["score"]
            verdict.decision = score["decision"]
        else:
            verdict.score = 0
            verdict.decision = "reject"
            verdict.rationale = (
                "Вердикт неповний: авторитетний score від інструмента score_candidate "
                f"відсутній у стані. {verdict.rationale}"
            )
        verdict.injection_detected = state.get("injection_detected", False)

        draft = await email_llm.ainvoke(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"Напиши лист кандидату за вердиктом:\n{verdict.model_dump_json()}"}]
        )
        # Output guardrail: навіть якщо промпт не вберіг лист від PII,
        # останній рубіж — маскування перед тим, як лист побачить людина.
        draft.body, _ = redact_pii(draft.body)
        return {"verdict": verdict.model_dump(), "email_draft": draft.model_dump()}

    return node


def _route(state: ScreeningState) -> str:
    return state.get("next_agent", "done")


def _human_approval_node(state: ScreeningState) -> dict:
    """HITL: граф зупиняється і чекає рішення людини щодо листа.

    Ризиковий send_candidate_email викликає граф ПІСЛЯ цієї зупинки, а не
    агент за власним рішенням — точка зупинки детермінована незалежно від
    того, чи "захоче" модель викликати інструмент. Людина бачить адресата,
    рішення, тему й повний текст. Відповідь:
    {"action": "approve" | "reject" | "edit", "body": "..."?}.

    supervisor може повернути "done" ДО того, як communicator сформував
    чернетку (маршрут done -> human_approval безумовний) — тоді питати
    людину нема про що, йдемо прямо до звіту."""
    draft = state.get("email_draft")
    if draft is None:
        return {"next_agent": "reject"}

    response = interrupt(
        {
            "type": "email_approval",
            "candidate_id": draft["candidate_id"],
            "decision": draft["decision"],
            "subject": draft["subject"],
            "body": draft["body"],
        }
    )

    action = response.get("action", "reject")
    if action == "edit":
        draft = {**draft, "body": response["body"]}
    return {"email_draft": draft, "next_agent": action}


def _send_email_node(tools: list[BaseTool]):
    """Викликає ризиковий MCP-інструмент — лише після схвалення людиною
    у _human_approval_node. Communicator сам цей інструмент не викликає."""
    send_tool = None

    async def node(state: ScreeningState) -> dict:
        nonlocal send_tool
        draft = state.get("email_draft")
        if draft is None:
            # За штатної розводки сюди без чернетки не дійти, але вузол не
            # повинен падати KeyError при прямому виклику (напр. з тесту).
            return {"messages": [HumanMessage(
                content="send_candidate_email: пропущено — у стані немає email_draft."
            )]}

        if send_tool is None:
            send_tool = next(t for t in tools if t.name == "send_candidate_email")

        raw = await send_tool.ainvoke(
            {
                "candidate_id": draft["candidate_id"],
                "decision": draft["decision"],
                "subject": draft["subject"],
                "body": draft["body"],
            }
        )
        return {"messages": [HumanMessage(content=f"send_candidate_email: {_tool_result_text(raw)}")]}

    return node


def _verdict_lines(verdict: dict) -> list[str]:
    return [
        f"Кандидат: {verdict.get('candidate_id')}",
        f"Вакансія: {verdict.get('job_id')}",
        f"Score: {verdict.get('score')}  →  {verdict.get('decision')}",
    ]


def _build_report(state: ScreeningState) -> str:
    """Фінальний звіт користувачу. Проходить output guardrail (redact_pii) —
    це останній рубіж перед тим, як звіт побачить людина."""
    verdict = state.get("verdict") or {}
    draft = state.get("email_draft")

    if draft is None:
        # supervisor завершив прогін ("done") до того, як communicator устиг
        # сформувати вердикт і чернетку — типово помилка маршрутизації
        # невеликої моделі або виснажені MAX_PARSER_RETRIES. Кажемо про це
        # прямо, а не показуємо звіт із порожніми полями як нормальний.
        lines = [
            "Скринінг НЕ завершено вердиктом: чернетку листа не сформовано "
            "(communicator не встиг відпрацювати до того, як маршрутизатор "
            "передав керування на завершення).",
        ]
        if verdict:
            lines += _verdict_lines(verdict)[:3]
        else:
            lines.append(f"Кандидат: {state.get('candidate_id')}")
    else:
        lines = _verdict_lines(verdict) + [
            f"Обґрунтування: {verdict.get('rationale')}",
            f"Gaps: {', '.join(verdict.get('gaps') or []) or 'немає'}",
            f"Виявлено спробу маніпуляції в резюме: "
            f"{'так' if verdict.get('injection_detected') else 'ні'}",
            "",
            f"Лист ({draft.get('subject')}):",
            draft.get("body", ""),
        ]

    report, found_pii = redact_pii("\n".join(lines))
    if found_pii:
        report += f"\n\n[Замасковано PII: {', '.join(found_pii)}]"
    return report


def _report_node(state: ScreeningState) -> dict:
    return {"report": _build_report(state)}


async def build_graph(tools: list[BaseTool], checkpointer=None, llm=None):
    """Збирає граф: supervisor маршрутизує, агенти працюють, супервізор знову.

    Після того як супервізор вирішить "done", граф іде НЕ в END, а на
    human_approval — і лише після схвалення людиною (approve/edit) переходить
    до send_email; reject веде прямо до report, минаючи надсилання.

    `llm` можна передати готовим (тести зі стаб-моделлю, без мережі й без
    ключів API) — інакше створюється справжня модель через make_llm().
    """
    if llm is None:
        llm = make_llm()

    builder = StateGraph(ScreeningState)
    builder.add_node("supervisor", _supervisor_node(llm))
    builder.add_node("resume_parser", _resume_parser_node(llm, tools))
    builder.add_node("requirements_matcher", _requirements_matcher_node(llm, tools))
    builder.add_node("communicator", _communicator_node(llm))
    builder.add_node("human_approval", _human_approval_node)
    builder.add_node("send_email", _send_email_node(tools))
    builder.add_node("report", _report_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _route,
        {
            "resume_parser": "resume_parser",
            "requirements_matcher": "requirements_matcher",
            "communicator": "communicator",
            "done": "human_approval",
        },
    )
    builder.add_edge("resume_parser", "supervisor")
    builder.add_edge("requirements_matcher", "supervisor")
    builder.add_edge("communicator", "supervisor")
    builder.add_conditional_edges(
        "human_approval",
        lambda state: "send_email" if state["next_agent"] in ("approve", "edit") else "report",
        {"send_email": "send_email", "report": "report"},
    )
    builder.add_edge("send_email", "report")
    builder.add_edge("report", END)

    return builder.compile(checkpointer=checkpointer)
