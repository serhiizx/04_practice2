"""Тести MCP-інструментів. Доменні функції викликаються напряму,
а окремий тест через in-memory Client перевіряє сам сервер."""

import json

from fastmcp import Client

import mcp_server
from mcp_server import fetch_job_requirements, fetch_resume, mcp, score_candidate


def test_skoryng_silnoho_kandydata():
    """CAND-001: усі must-have, один nice-to-have, досвіду з запасом."""
    result = score_candidate(
        skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
        years_experience=6,
        job_id="JOB-BACKEND",
    )
    assert result["status"] == "ok"
    assert result["data"]["score"] == 93
    assert result["data"]["decision"] == "strong_match"
    assert result["data"]["missing_must_have"] == []


def test_skoryng_slabkoho_kandydata():
    """CAND-002: лише Python, досвіду бракує."""
    result = score_candidate(skills=["Python"], years_experience=1, job_id="JOB-BACKEND")
    assert result["data"]["score"] == 28
    assert result["data"]["decision"] == "reject"
    assert result["data"]["meets_min_years"] is False
    assert set(result["data"]["missing_must_have"]) == {"PostgreSQL", "Docker"}


def test_fetch_resume_nevidomyi_kandydat_povertaye_pomylku():
    result = fetch_resume(candidate_id="CAND-999")
    assert result["status"] == "error"
    assert "CAND-999" in result["error"]


def test_nevalidni_arhumenty_vidkydayutsya_shemoyu():
    """Невалідний job_id має відхилятись схемою, а не падати всередині."""
    result = score_candidate(skills=["Python"], years_experience=3, job_id="lowercase")
    assert result["status"] == "error"
    assert "валідац" in result["error"].lower()


def test_fetch_job_requirements_viddaye_vymohy():
    result = fetch_job_requirements(job_id="JOB-BACKEND")
    assert result["status"] == "ok"
    assert result["data"]["min_years"] == 3


async def test_server_viddaye_chotyry_instrumenty():
    """Перевірка самого MCP-сервера, а не лише доменних функцій."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools} == {
        "fetch_resume",
        "fetch_job_requirements",
        "score_candidate",
        "send_candidate_email",
    }


async def test_vyklyk_cherez_mcp_protokol_povertaye_json(tmp_path, monkeypatch):
    """Ризиковий інструмент пише в outbox, а не шле реальний лист."""
    monkeypatch.setattr(mcp_server, "OUTBOX_PATH", tmp_path / "outbox.json")
    async with Client(mcp) as client:
        result = await client.call_tool(
            "send_candidate_email",
            {
                "candidate_id": "CAND-001",
                "decision": "strong_match",
                "subject": "Запрошення на співбесіду",
                "body": "Вітаємо! Запрошуємо вас на співбесіду.",
            },
        )
    payload = json.loads(result.content[0].text)
    assert payload["status"] == "ok"
    assert json.loads((tmp_path / "outbox.json").read_text(encoding="utf-8"))
