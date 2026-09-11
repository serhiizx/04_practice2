"""Збирає Task_002_Жданюк_83.ipynb із демонстраційних сценаріїв.

Notebook мусить виконуватись ПОВНІСТЮ навіть без ключів API: клітинки, що
працюють офлайн (MCP-інструменти, guardrails, вміст comparison.json і
redteam_results.json), виконуються насправді; клітинки, що потребують LLM
(повний прогін LangGraph/CrewAI), самі перевіряють наявність OPENAI_API_KEY
і друкують зрозуміле повідомлення замість traceback, якщо ключа немає.
"""

import nbformat as nbf

from config import ROOT

NOTEBOOK_PATH = ROOT / "Task_002_Жданюк_83.ipynb"

NO_KEY_LANGGRAPH = (
    "import os\n\n"
    "if not os.environ.get('OPENAI_API_KEY', '').strip():\n"
    "    print('Цей крок потребує ключа OPENAI_API_KEY (справжній OpenAI або '\n"
    "          'будь-який непорожній рядок для локального LM Studio) — пропущено. '\n"
    "          'Заповніть .env за прикладом .env.example (cp .env.example .env) '\n"
    "          'і перезапустіть цю комірку.')\n"
    "else:\n"
    "    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver\n"
    "    from langgraph.types import Command\n"
    "    from config import CHECKPOINT_DB\n"
    "    from langgraph_mas import build_graph, load_mcp_tools\n\n"
    "    async def demo(candidate_id, action):\n"
    "        client, tools = await load_mcp_tools()\n"
    "        cfg = {'configurable': {'thread_id': f'nb-{candidate_id}-{action}'},\n"
    "               'recursion_limit': 25}\n"
    "        async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as cp:\n"
    "            graph = await build_graph(tools, checkpointer=cp)\n"
    "            state = await graph.ainvoke({'candidate_id': candidate_id,\n"
    "                                         'job_id': 'JOB-BACKEND', 'messages': []}, cfg)\n"
    "            while '__interrupt__' in state:\n"
    "                print('HITL запит:', state['__interrupt__'][0].value['subject'])\n"
    "                state = await graph.ainvoke(Command(resume={'action': action}), cfg)\n"
    "        return state\n\n"
    "    state = await demo('CAND-003', 'reject')\n"
    "    print(state['report'])"
)

NO_KEY_CREWAI = (
    "import os\n\n"
    "if not os.environ.get('OPENAI_API_KEY', '').strip():\n"
    "    print('Цей крок потребує ключа OPENAI_API_KEY — пропущено. '\n"
    "          'Заповніть .env за прикладом .env.example і перезапустіть цю комірку.')\n"
    "else:\n"
    "    from crewai_mas import run_crew\n\n"
    "    print(run_crew('CAND-001', auto_approve=True))"
)

CELLS = [
    ("md", "# Практичне завдання №2 — MAS HR-скринінгу з MCP\n\n"
           "Мультиагентна система скринінгу кандидатів: "
           "LangGraph і CrewAI поверх спільного MCP-сервера, з guardrails, "
           "HITL і tracing.\n\n"
           "**Дані мокові, це навчальний проєкт.** Ключів API (OPENAI_API_KEY, "
           "LANGFUSE_*) під час підготовки цього notebook не було — тому кожна "
           "клітинка, що потребує LLM, сама перевіряє наявність ключа і чесно "
           "друкує повідомлення про пропуск замість traceback. Клітинки, що не "
           "потребують моделі (інструменти MCP-сервера, guardrails, вміст "
           "збережених JSON-звітів), виконуються насправді."),
    ("md", "## 1. MCP-сервер: чотири інструменти\n\n"
           "`send_candidate_email` тут не викликається — щоб не дописувати в "
           "робочий `data/outbox.json` побічний рядок при кожному запуску "
           "notebook. Його викликає граф LangGraph після HITL-схвалення (розділ 5) "
           "і тест `tests/test_human_in_the_loop.py`."),
    ("code", "from mcp_server import fetch_job_requirements, fetch_resume, score_candidate\n\n"
             "print(fetch_resume('CAND-001'))\n"
             "print(fetch_job_requirements('JOB-BACKEND'))\n"
             "print(score_candidate(skills=['Python', 'PostgreSQL', 'Docker', 'Kubernetes'],\n"
             "                       years_experience=6, job_id='JOB-BACKEND'))"),
    ("md", "## 2. Input guardrail: injection у резюме CAND-003\n\n"
           "`detect_injection` — евристичний префільтр, не межа безпеки (докладніше "
           "в README, розділ Guardrails). Резюме CAND-003 містить вшите "
           "\"IGNORE ALL PREVIOUS INSTRUCTIONS...\"."),
    ("code", "from config import load_candidates\n"
             "from guardrails import detect_injection\n\n"
             "verdict = detect_injection(load_candidates()['CAND-003']['resume_text'])\n"
             "print('виявлено:', verdict.detected)\n"
             "print('патерни:', verdict.patterns)\n"
             "print()\n"
             "print(verdict.safe_text)"),
    ("md", "## 3. Tool guardrail: парсер не має права надіслати лист\n\n"
           "Allowlist — єдиний структурний бар'єр серед трьох рівнів guardrails: "
           "`resume_parser` має право лише на `fetch_resume`, і жодна переконлива "
           "інструкція з резюме це не змінить."),
    ("code", "from guardrails import ToolDenied, check_tool_call\n\n"
             "try:\n"
             "    check_tool_call('resume_parser', 'send_candidate_email',\n"
             "                    {'candidate_id': 'CAND-003', 'decision': 'strong_match',\n"
             "                     'subject': 'Офер', 'body': 'Вас прийнято.'})\n"
             "except ToolDenied as exc:\n"
             "    print('заблоковано:', exc)"),
    ("md", "## 4. Output guardrail: маскування PII у резюме CAND-004\n\n"
           "Резюме CAND-004 містить ІПН, дату народження, телефон, email і адресу. "
           "`redact_pii` — пом'якшення витоку в лог/UI, а не приватність за "
           "побудовою: дані вже пройшли крізь модель до цього маскування."),
    ("code", "from guardrails import redact_pii\n\n"
             "redacted, found = redact_pii(load_candidates()['CAND-004']['resume_text'])\n"
             "print(redacted)\n"
             "print('знайдено:', found)"),
    ("md", "## 5. Повний прогін LangGraph із HITL\n\n"
           "Граф зупиняється перед надсиланням листа (`interrupt()` у "
           "`_human_approval_node`). У notebook підтвердження передається "
           "програмно через `Command(resume=...)` — тут узято `action='reject'`, "
           "щоб демонстрація не писала в `data/outbox.json`.\n\n"
           "**Потребує ключ OPENAI_API_KEY** — без нього комірка друкує "
           "повідомлення про пропуск і завершується без помилки."),
    ("code", NO_KEY_LANGGRAPH),
    ("md", "## 6. Порівняння з CrewAI\n\n"
           "Той самий кейс (CAND-001), інша оркестрація. **Потребує ключ "
           "OPENAI_API_KEY.**"),
    ("code", NO_KEY_CREWAI),
    ("md", "## 7. Порівняльна таблиця й вартість (`comparison.json`)\n\n"
           "Рядки коду виміряні завжди (офлайн). Вердикти, час і вартість "
           "позначені `\"потребує прогону з ключами API\"`, якщо прогін "
           "`compare.py` виконувався без ключів — це чесна позначка "
           "невиміряного значення, а не вигадане число."),
    ("code", "import json\n"
             "from config import ROOT\n\n"
             "print(json.dumps(json.loads((ROOT / 'comparison.json').read_text()),\n"
             "                 ensure_ascii=False, indent=2))"),
    ("md", "## 8. Red-teaming (`redteam_results.json`)\n\n"
           "Детермінована частина (11 спроб обходу allowlist, 10 payload-ів "
           "детектора injection, маскування PII на CAND-004 і задокументована "
           "межа) виконана без моделі. Секція `deepteam` потребує "
           "OPENAI_API_KEY і не виконувалась — `executed: false` із причиною."),
    ("code", "print(json.dumps(json.loads((ROOT / 'redteam_results.json').read_text()),\n"
             "                 ensure_ascii=False, indent=2))"),
]


def build() -> None:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        nbf.v4.new_markdown_cell(content) if kind == "md" else nbf.v4.new_code_cell(content)
        for kind, content in CELLS
    ]
    nbf.write(notebook, str(NOTEBOOK_PATH))
    print(f"створено {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build()
