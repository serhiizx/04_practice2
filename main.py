"""Точка входу: скринінг кандидата з термінала.

    uv run python main.py CAND-001
    uv run python main.py CAND-003 --job JOB-BACKEND
    uv run python main.py CAND-002 --impl crewai
"""

import argparse
import asyncio
import json

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from config import CHECKPOINT_DB, langfuse_enabled, make_langfuse_handler
from langgraph_mas import build_graph, load_mcp_tools


def _ask_human(payload: dict) -> dict:
    """Запитує рішення людини щодо листа просто в терміналі.

    Винесена окремою функцією (а не input() у тілі циклу screen()), щоб
    тести могли підмінити її монкіпатчем без інтерактивного вводу."""
    print("\n" + "=" * 70)
    print("ПОТРІБНЕ ПІДТВЕРДЖЕННЯ ЛЮДИНИ")
    print("=" * 70)
    print(f"Кандидат: {payload['candidate_id']}")
    print(f"Рішення:  {payload['decision']}")
    print(f"Тема:     {payload['subject']}")
    print(f"\n{payload['body']}\n")
    print("=" * 70)

    while True:
        answer = input("[a]pprove / [r]eject / [e]dit: ").strip().lower()
        if answer.startswith("a"):
            return {"action": "approve"}
        if answer.startswith("r"):
            return {"action": "reject"}
        if answer.startswith("e"):
            return {"action": "edit", "body": input("Новий текст листа: ").strip()}
        print("Не зрозумів. Введіть a, r або e.")


async def screen(
    candidate_id: str, job_id: str, thread_id: str, decision_fn=_ask_human
) -> dict:
    """Повний прогін скринінгу з human-in-the-loop.

    Зупиняється на interrupt стільки разів, скільки граф його підніме
    (наразі рівно один — human_approval), і щоразу питає рішення через
    decision_fn — за замовчуванням _ask_human (термінал), його підміняють
    у тестах, щоб не чіпати input().

    decision_fn параметризований (а не хардкод input() у тілі) саме для
    того, щоб порівняльний прогін (compare.py) міг передати наперед задане
    рішення (наприклад, завжди "approve") й не зависати на терміналі та не
    робити прогони LangGraph і CrewAI незіставними через відповіді людини."""
    client, tools = await load_mcp_tools()
    handler = make_langfuse_handler()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 25,
        # Порожній список callbacks, якщо ключів Langfuse немає — трейсинг
        # не є обов'язковою залежністю робочого шляху.
        "callbacks": [handler] if handler else [],
        "metadata": {
            "langfuse_tags": ["impl:langgraph", f"candidate:{candidate_id}"],
            "langfuse_session_id": thread_id,
        },
    }

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer:
        graph = await build_graph(tools, checkpointer=checkpointer)

        state = await graph.ainvoke(
            {"candidate_id": candidate_id, "job_id": job_id, "messages": []}, config
        )

        # Граф зупинився на interrupt — питаємо людину й відновлюємо прогін.
        while "__interrupt__" in state:
            payload = state["__interrupt__"][0].value
            state = await graph.ainvoke(Command(resume=decision_fn(payload)), config)

    # Скидаємо буфер подій Langfuse лише якщо трейсинг увімкнено — інакше
    # get_client() створив би клієнта й ліз у мережу без потреби.
    if langfuse_enabled():
        from langfuse import get_client

        get_client().flush()

    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="HR-скринінг кандидата")
    parser.add_argument("candidate_id", help="напр. CAND-001")
    parser.add_argument("--job", default="JOB-BACKEND")
    parser.add_argument("--impl", default="langgraph", choices=["langgraph", "crewai"])
    parser.add_argument("--thread", default="cli", help="thread_id для checkpointer")
    args = parser.parse_args()

    if args.impl == "crewai":
        from crewai_mas import run_crew

        print(run_crew(args.candidate_id, args.job))
        return

    try:
        state = asyncio.run(screen(args.candidate_id, args.job, args.thread))
    except GraphRecursionError:
        print(
            "\nСкринінг зупинено: граф перевищив ліміт кроків (recursion_limit). "
            "Це означає, що supervisor не зійшовся на рішенні "
            f"(див. MAX_PARSER_RETRIES у langgraph_mas.py) — спробуйте ще раз "
            f"або перевірте, що candidate_id='{args.candidate_id}' і "
            f"job_id='{args.job}' існують у data/."
        )
        return

    print("\n" + state.get("report", json.dumps(state.get("verdict"), ensure_ascii=False)))


if __name__ == "__main__":
    main()
