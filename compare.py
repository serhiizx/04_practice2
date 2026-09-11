"""Порівняння двох реалізацій MAS на однаковому наборі прогонів.

Умова чесності: ті самі кандидати, та сама модель, ті самі запити. Без цього
цифри токенів і вартості порівнювати не можна.

Практичне зауваження: `screen()` у LangGraph-реалізації зупиняється
на interrupt і питає рішення людини в терміналі (main._ask_human, input()).
Для порівняльного прогону це неприйнятне — воно б зависало без інтерактивного
терміналу, а різні відповіді людини для різних кандидатів зробили б прогони
незіставними. Тому main.screen() отримав параметр decision_fn (за
замовчуванням _ask_human), і тут ми передаємо _auto_approve — наперед задане
рішення "approve" для кожного interrupt, так само як run_crew уже має
auto_approve. Ніякого фальшивого input() усередині screen() не додано.

Без ключів (OPENAI_API_KEY, LANGFUSE_*) живі прогони обох реалізацій
неможливі: немає LLM. Модуль тоді чесно заповнює лише те, що вимірюється
офлайн (рядки коду), а решту позначає NOT_MEASURED — жодних вигаданих чисел.
"""

import asyncio
import json
import os
import time
from pathlib import Path

from config import ROOT, langfuse_enabled

DEMO_CANDIDATES = ["CAND-001", "CAND-002", "CAND-003", "CAND-004"]
COMPARISON_PATH = ROOT / "comparison.json"

# Чесна позначка невиміряного значення — не null, не 0, а явний текст,
# який неможливо сплутати зі справжнім виміром.
NOT_MEASURED = "потребує прогону з ключами API"


def count_loc(path: Path) -> int:
    """Рядки коду без порожніх і без рядків-коментарів.

    Докстрінги рахуються як код — це рядкові літерали, а не коментарі."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def fetch_cost_by_trace_name() -> dict:
    """Витягує вартість і токени з Langfuse, згруповані за назвою трейсу.

    Без ключів Langfuse — явна позначка недоступності, а не порожній {} і не
    вигадане число. Мережевий виклик відбувається лише якщо ключі є."""
    if not langfuse_enabled():
        return {
            "status": "unavailable",
            "reason": (
                "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY відсутні — "
                f"{NOT_MEASURED}"
            ),
        }
    from langfuse import get_client

    client = get_client()
    client.flush()
    query = json.dumps(
        {
            "view": "observations",
            "metrics": [
                {"measure": "totalCost", "aggregation": "sum"},
                {"measure": "totalTokens", "aggregation": "sum"},
            ],
            "dimensions": [{"field": "traceName"}],
            "filters": [],
            "fromTimestamp": "2026-09-01T00:00:00Z",
            "toTimestamp": "2026-12-31T00:00:00Z",
        }
    )
    try:
        return {"status": "measured", "metrics": client.api.metrics.get(query=query)}
    except Exception as exc:  # API Langfuse може віддати іншу форму відповіді
        return {"status": "error", "reason": f"метрики недоступні через API: {exc}; взяти з UI Langfuse"}


def _auto_approve(payload: dict) -> dict:
    """Наперед задане рішення для порівняльного прогону: завжди 'approve'.

    Підмінює _ask_human у main.screen() через decision_fn, щоб не питати
    людину в терміналі й лишити прогони кандидатів зіставними між собою."""
    return {"action": "approve"}


def _has_llm_key() -> bool:
    """Чи є мінімум для живого прогону — ключ OpenAI (або сумісного API)."""
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


async def _run_langgraph(candidate_ids: list[str]) -> dict:
    """Прогін LangGraph-реалізації з автосхваленням, або чесні позначки,
    якщо ключів немає чи прогін впав."""
    result: dict = {"loc": count_loc(ROOT / "langgraph_mas.py")}
    if not _has_llm_key():
        result["verdicts"] = {cid: NOT_MEASURED for cid in candidate_ids}
        result["wall_seconds"] = NOT_MEASURED
        return result

    from main import screen

    verdicts: dict = {}
    started = time.perf_counter()
    try:
        for candidate_id in candidate_ids:
            state = await screen(
                candidate_id,
                "JOB-BACKEND",
                thread_id=f"cmp-lg-{candidate_id}",
                decision_fn=_auto_approve,
            )
            verdicts[candidate_id] = (state.get("verdict") or {}).get("decision")
    except Exception as exc:  # мережа/ключі можуть відпасти вже під час прогону
        result["verdicts"] = {cid: NOT_MEASURED for cid in candidate_ids}
        result["wall_seconds"] = NOT_MEASURED
        result["error"] = f"прогін не вдався: {exc}"
        return result

    result["verdicts"] = verdicts
    result["wall_seconds"] = round(time.perf_counter() - started, 2)
    return result


def _run_crewai(candidate_ids: list[str]) -> dict:
    """Прогін CrewAI-реалізації з auto_approve=True, або чесні позначки."""
    result: dict = {"loc": count_loc(ROOT / "crewai_mas.py")}
    if not _has_llm_key():
        result["reports"] = {cid: NOT_MEASURED for cid in candidate_ids}
        result["wall_seconds"] = NOT_MEASURED
        return result

    from crewai_mas import run_crew

    reports: dict = {}
    started = time.perf_counter()
    try:
        for candidate_id in candidate_ids:
            reports[candidate_id] = run_crew(candidate_id, "JOB-BACKEND", auto_approve=True)[:200]
    except Exception as exc:
        result["reports"] = {cid: NOT_MEASURED for cid in candidate_ids}
        result["wall_seconds"] = NOT_MEASURED
        result["error"] = f"прогін не вдався: {exc}"
        return result

    result["reports"] = reports
    result["wall_seconds"] = round(time.perf_counter() - started, 2)
    return result


async def run_comparison(candidate_ids: list[str] | None = None) -> dict:
    """Проганяє обидві реалізації на однакових кандидатах і збирає метрики.

    Виміряне (рядки коду завжди; verdict/час/вартість — лише за наявності
    ключів) і невиміряне (позначено NOT_MEASURED) явно розрізнені в
    результаті, а не змішані в одні й ті самі поля."""
    candidate_ids = candidate_ids or DEMO_CANDIDATES
    results: dict = {
        "candidates": candidate_ids,
        "langgraph": await _run_langgraph(candidate_ids),
        "crewai": _run_crewai(candidate_ids),
        "cost": fetch_cost_by_trace_name(),
    }
    COMPARISON_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run_comparison()), ensure_ascii=False, indent=2, default=str))
