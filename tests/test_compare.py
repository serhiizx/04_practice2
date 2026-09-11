"""Офлайн-тести compare.py: без мережі, без ключів API, без LLM.

Перевіряють те, що реально можна перевірити без ключів: підрахунок рядків
коду, чесну позначку недоступності вартості/токенів (а не вигадане число чи
падіння), та що run_comparison() без ключів створює comparison.json, де
рядки коду виміряні, а решта явно позначена як невиміряна.
"""

import asyncio
import json

import compare


# --- 1. count_loc ігнорує порожні рядки й коментарі --------------------


def test_count_loc_ignoruye_porozhni_ryadky_i_komentari(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "# коментар на весь рядок",
                "",
                "x = 1  # коментар не рахується, бо рядок не ПОЧИНАЄТЬСЯ з #",
                "   ",
                "def f():",
                '    """Докстрінг — це рядковий літерал, а не коментар."""',
                "    return x",
                "",
                "# ще один коментар",
            ]
        ),
        encoding="utf-8",
    )
    # Код: "x = 1  # ...", "def f():", докстрінг, "return x" = 4 рядки.
    # Порожні рядки й рядки, що ПОЧИНАЮТЬСЯ з "#", не рахуються.
    assert compare.count_loc(sample) == 4


# --- 2. Вартість без ключів Langfuse -----------------------------------


def test_fetch_cost_bez_klyuchiv_povertaye_poznachku_a_ne_padaye(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    result = compare.fetch_cost_by_trace_name()

    assert result["status"] == "unavailable"
    assert compare.NOT_MEASURED in result["reason"]
    # Жодних числових полів вартості/токенів не вигадано.
    assert "totalCost" not in result
    assert "totalTokens" not in result
    assert "metrics" not in result


# --- 3. run_comparison без ключів: LOC виміряно, решта чесно позначена ---


async def _run_without_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    comparison_path = tmp_path / "comparison.json"
    monkeypatch.setattr(compare, "COMPARISON_PATH", comparison_path)
    result = await compare.run_comparison(["CAND-001"])
    return result, comparison_path


async def test_run_comparison_bez_klyuchiv_ne_padaye_i_zapovnyuye_loc(monkeypatch, tmp_path):
    result, comparison_path = await _run_without_keys(monkeypatch, tmp_path)

    # Рядки коду виміряні офлайн — справжні числа, не позначки NOT_MEASURED.
    # Точне значення навмисно не фіксується: воно змінюється з кожною
    # правкою коду, і тест перетворився б на гальмо для рефакторингу.
    for impl in ("langgraph", "crewai"):
        assert isinstance(result[impl]["loc"], int)
        assert result[impl]["loc"] > 0

    # Усе, що вимагає живого прогону з LLM, чесно позначено, а не вигадане.
    assert result["langgraph"]["wall_seconds"] == compare.NOT_MEASURED
    assert result["crewai"]["wall_seconds"] == compare.NOT_MEASURED
    assert all(v == compare.NOT_MEASURED for v in result["langgraph"]["verdicts"].values())
    assert all(v == compare.NOT_MEASURED for v in result["crewai"]["reports"].values())
    assert result["cost"]["status"] == "unavailable"

    assert comparison_path.exists()
    on_disk = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert on_disk == result


def test_run_comparison_bez_klyuchiv_nemaye_vygadanyh_chysel(monkeypatch, tmp_path):
    # Тест синхронний (не async def), тому власноруч запускає event loop
    # через asyncio.run, а не покладається на asyncio_mode = auto.
    result, comparison_path = asyncio.run(_run_without_keys(monkeypatch, tmp_path))

    raw = comparison_path.read_text(encoding="utf-8")
    # У полях, що вимагають живого прогону, немає жодного числа — лише
    # позначка NOT_MEASURED. wall_seconds — рядок, а не float/int.
    assert isinstance(result["langgraph"]["wall_seconds"], str)
    assert isinstance(result["crewai"]["wall_seconds"], str)
    assert "totalCost" not in raw
    assert "totalTokens" not in raw


# --- 4. compare імпортується без ключів API ----------------------------


def test_compare_importuyetsya_bez_klyuchiv(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    import importlib

    importlib.reload(compare)
    assert hasattr(compare, "run_comparison")
    assert hasattr(compare, "count_loc")
