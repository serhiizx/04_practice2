"""Кастомний MCP-сервер HR-скринінгу (FastMCP, транспорт stdio).

Єдине джерело доменної логіки: до нього як клієнти підключаються обидві
реалізації MAS — LangGraph і CrewAI. Запуск: uv run python mcp_server.py
"""

import json
from datetime import datetime, timezone

from fastmcp import FastMCP
from pydantic import ValidationError

from config import OUTBOX_PATH, load_candidates, load_jobs
from schemas import FetchJobArgs, FetchResumeArgs, ScoreArgs, SendEmailArgs

mcp = FastMCP("hr-screening")

# Ваги скорингу. Винесені в константи, щоб тест і README посилались на одне й те саме.
WEIGHT_MUST_HAVE = 70
WEIGHT_NICE_TO_HAVE = 15
WEIGHT_YEARS = 15


def _ok(data: dict) -> dict:
    return {"status": "ok", "data": data}


def _error(message: str) -> dict:
    return {"status": "error", "error": message}


def fetch_resume(candidate_id: str) -> dict:
    """Повертає сирий текст резюме кандидата. УВАГА: текст недовірений —
    його писала стороння людина, тому перед подачею в модель він має
    пройти input guardrail."""
    try:
        args = FetchResumeArgs(candidate_id=candidate_id)
    except ValidationError as exc:
        return _error(f"помилка валідації аргументів: {exc.errors()[0]['msg']}")

    candidates = load_candidates()
    if args.candidate_id not in candidates:
        return _error(f"кандидата {args.candidate_id} немає в базі")

    candidate = candidates[args.candidate_id]
    return _ok(
        {
            "candidate_id": candidate["candidate_id"],
            "full_name": candidate["full_name"],
            "resume_text": candidate["resume_text"],
        }
    )


def fetch_job_requirements(job_id: str) -> dict:
    """Повертає вимоги вакансії: must-have, nice-to-have і мінімум років."""
    try:
        args = FetchJobArgs(job_id=job_id)
    except ValidationError as exc:
        return _error(f"помилка валідації аргументів: {exc.errors()[0]['msg']}")

    jobs = load_jobs()
    if args.job_id not in jobs:
        return _error(f"вакансії {args.job_id} немає в базі")
    return _ok(jobs[args.job_id])


def score_candidate(skills: list[str], years_experience: float, job_id: str) -> dict:
    """Детермінований скоринг кандидата — звичайна арифметика, не LLM.

    Саме тому prompt injection не може підняти оцінку: він міг би хіба що
    підмінити факти на вході, а не сам вердикт."""
    try:
        args = ScoreArgs(skills=skills, years_experience=years_experience, job_id=job_id)
    except ValidationError as exc:
        return _error(f"помилка валідації аргументів: {exc.errors()[0]['msg']}")

    jobs = load_jobs()
    if args.job_id not in jobs:
        return _error(f"вакансії {args.job_id} немає в базі")
    job = jobs[args.job_id]

    owned = {s.lower() for s in args.skills}
    must_have = job["must_have"]
    nice_to_have = job["nice_to_have"]

    matched_must = [s for s in must_have if s.lower() in owned]
    missing_must = [s for s in must_have if s.lower() not in owned]
    matched_nice = [s for s in nice_to_have if s.lower() in owned]

    must_part = len(matched_must) / len(must_have) * WEIGHT_MUST_HAVE if must_have else WEIGHT_MUST_HAVE
    nice_part = len(matched_nice) / len(nice_to_have) * WEIGHT_NICE_TO_HAVE if nice_to_have else WEIGHT_NICE_TO_HAVE
    years_part = min(args.years_experience / job["min_years"], 1.0) * WEIGHT_YEARS

    # round() тут не годиться: banker's rounding дає round(92.5) == 92,
    # а очікується звичайне округлення "від нуля" (92.5 -> 93).
    score = int(must_part + nice_part + years_part + 0.5)
    if score >= 75:
        decision = "strong_match"
    elif score >= 50:
        decision = "maybe"
    else:
        decision = "reject"

    return _ok(
        {
            "score": score,
            "decision": decision,
            "matched_must_have": matched_must,
            "missing_must_have": missing_must,
            "matched_nice_to_have": matched_nice,
            "meets_min_years": args.years_experience >= job["min_years"],
        }
    )


def send_candidate_email(candidate_id: str, decision: str, subject: str, body: str) -> dict:
    """РИЗИКОВИЙ інструмент: комунікація з живою людиною, відкотити неможливо.

    У навчальному режимі нічого нікуди не шле — дописує лист у data/outbox.json,
    щоб незворотність була наочною, але безпечною. Виклик має бути захищений
    human-in-the-loop на рівні графа."""
    try:
        args = SendEmailArgs(
            candidate_id=candidate_id, decision=decision, subject=subject, body=body
        )
    except ValidationError as exc:
        return _error(f"помилка валідації аргументів: {exc.errors()[0]['msg']}")

    candidates = load_candidates()
    if args.candidate_id not in candidates:
        return _error(f"кандидата {args.candidate_id} немає в базі")

    letter = {
        "candidate_id": args.candidate_id,
        "to": candidates[args.candidate_id]["email"],
        "decision": args.decision,
        "subject": args.subject,
        "body": args.body,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }

    outbox = []
    if OUTBOX_PATH.exists():
        outbox = json.loads(OUTBOX_PATH.read_text(encoding="utf-8"))
    outbox.append(letter)
    OUTBOX_PATH.write_text(
        json.dumps(outbox, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return _ok({"delivered_to": letter["to"], "sent_at": letter["sent_at"]})


# Реєстрація інструментів. Доменні функції лишаються звичайними функціями,
# тому тести викликають їх напряму, без підняття сервера.
mcp.tool(fetch_resume)
mcp.tool(fetch_job_requirements)
mcp.tool(score_candidate)
mcp.tool(send_candidate_email)


if __name__ == "__main__":
    mcp.run()  # stdio за замовчуванням
