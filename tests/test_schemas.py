"""Тести Pydantic-схем: коректні й некоректні входи."""

import pytest
from pydantic import ValidationError

from schemas import FetchResumeArgs, RouteDecision, ScoreArgs, SendEmailArgs


def test_fetch_resume_pryimaye_korektnyi_id():
    assert FetchResumeArgs(candidate_id="CAND-001").candidate_id == "CAND-001"


@pytest.mark.parametrize("bad_id", ["CAND-1", "cand-001", "XXXX-001", "", "CAND-0001"])
def test_fetch_resume_vidkydaye_nekorektnyi_id(bad_id):
    with pytest.raises(ValidationError):
        FetchResumeArgs(candidate_id=bad_id)


def test_score_args_vidkydaye_vidyemni_roky():
    with pytest.raises(ValidationError):
        ScoreArgs(skills=["Python"], years_experience=-1, job_id="JOB-BACKEND")


def test_score_args_vidkydaye_porozhnii_spysok_navychok():
    with pytest.raises(ValidationError):
        ScoreArgs(skills=[], years_experience=3, job_id="JOB-BACKEND")


def test_send_email_vidkydaye_porozhnie_tilo():
    with pytest.raises(ValidationError):
        SendEmailArgs(
            candidate_id="CAND-001", decision="reject", subject="Тема", body="   "
        )


def test_route_decision_vidkydaye_nevidomoho_ahenta():
    with pytest.raises(ValidationError):
        RouteDecision(next_agent="hacker", reason="спроба обійти маршрутизацію")
