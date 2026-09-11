"""Тести доменних даних — без них решта тестів перевіряла б порожнечу."""

import re

from config import load_candidates, load_jobs


def test_zavantazheno_chotyry_kandydaty():
    assert set(load_candidates()) == {"CAND-001", "CAND-002", "CAND-003", "CAND-004"}


def test_cand_003_mistyt_injection_payload():
    text = load_candidates()["CAND-003"]["resume_text"].lower()
    assert "ignore" in text
    assert "instructions" in text


def test_cand_004_mistyt_pii():
    text = load_candidates()["CAND-004"]["resume_text"]
    assert "@" in text, "потрібен email"
    assert re.search(r"\b\d{10}\b", text), "потрібен 10-значний ІПН"
    assert re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", text), "потрібна дата народження"


def test_vakansiya_backend_maye_vymohy():
    job = load_jobs()["JOB-BACKEND"]
    assert job["min_years"] == 3
    assert [s.lower() for s in job["must_have"]] == ["python", "postgresql", "docker"]
