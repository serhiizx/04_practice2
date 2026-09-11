"""Pydantic v2 схеми: аргументи інструментів і доменні моделі."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

CANDIDATE_ID_PATTERN = r"^CAND-\d{3}$"
JOB_ID_PATTERN = r"^JOB-[A-Z]+$"

Decision = Literal["strong_match", "maybe", "reject"]
AgentName = Literal["resume_parser", "requirements_matcher", "communicator", "done"]


# --- Аргументи MCP-інструментів -------------------------------------------


class FetchResumeArgs(BaseModel):
    """Аргументи fetch_resume."""

    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN, description="Ідентифікатор кандидата, напр. CAND-001")


class FetchJobArgs(BaseModel):
    """Аргументи fetch_job_requirements."""

    job_id: str = Field(pattern=JOB_ID_PATTERN, description="Ідентифікатор вакансії, напр. JOB-BACKEND")


class ScoreArgs(BaseModel):
    """Аргументи score_candidate."""

    skills: list[str] = Field(min_length=1, description="Навички, витягнуті з резюме")
    years_experience: float = Field(ge=0, le=60, description="Роки релевантного досвіду")
    job_id: str = Field(pattern=JOB_ID_PATTERN)

    @field_validator("skills")
    @classmethod
    def navychky_ne_porozhni(cls, value: list[str]) -> list[str]:
        cleaned = [s.strip() for s in value if s and s.strip()]
        if not cleaned:
            raise ValueError("список навичок не може складатися з порожніх рядків")
        return cleaned


class SendEmailArgs(BaseModel):
    """Аргументи ризикового інструмента send_candidate_email."""

    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN)
    decision: Decision
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=4000)

    @field_validator("subject", "body")
    @classmethod
    def tekst_ne_porozhnii(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("текст не може складатися лише з пробілів")
        return value.strip()


# --- Доменні моделі --------------------------------------------------------


class ResumeFacts(BaseModel):
    """Структуровані факти, витягнуті ResumeParser'ом із сирого тексту."""

    skills: list[str] = Field(default_factory=list)
    years_experience: float = Field(default=0, ge=0, le=60)
    education: str = ""
    location: str = ""


class JobRequirements(BaseModel):
    """Вимоги вакансії."""

    job_id: str
    title: str
    must_have: list[str]
    nice_to_have: list[str]
    min_years: float


class ScoreResult(BaseModel):
    """Результат детермінованого скорингу."""

    score: int = Field(ge=0, le=100)
    decision: Decision
    matched_must_have: list[str]
    missing_must_have: list[str]
    matched_nice_to_have: list[str]
    meets_min_years: bool


class ScreeningVerdict(BaseModel):
    """Фінальний вердикт скринінгу."""

    candidate_id: str
    job_id: str
    score: int = Field(ge=0, le=100)
    decision: Decision
    rationale: str
    gaps: list[str] = Field(default_factory=list)
    injection_detected: bool = False


class RouteDecision(BaseModel):
    """Рішення supervisor'а, кому передати роботу далі."""

    next_agent: AgentName
    reason: str = Field(min_length=1, description="Чому саме цей агент")


class EmailDraft(BaseModel):
    """Чернетка листа кандидату — те, що людина бачить під час HITL."""

    candidate_id: str = Field(pattern=CANDIDATE_ID_PATTERN)
    decision: Decision
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
