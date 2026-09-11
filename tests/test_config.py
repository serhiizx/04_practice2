"""Офлайн-тести config.make_llm: без мережі, без справжнього ключа.

Без .env перевіряльник, що просто запустить main.py, найперше побачить саме
цю помилку — вона має бути зрозумілою підказкою, а не сирим KeyError.
"""

import config


def test_make_llm_bez_klyucha_kydaye_zrozumilu_pomylku_a_ne_keyerror(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        config.make_llm()
    except KeyError:
        raise AssertionError("make_llm() без ключа має кидати RuntimeError, а не KeyError")
    except RuntimeError as exc:
        assert ".env.example" in str(exc)
    else:
        raise AssertionError("make_llm() без ключа мав кинути RuntimeError")


def test_make_llm_z_klyuchem_bere_model_i_base_url_z_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")

    llm = config.make_llm()

    assert llm.model_name == "gpt-test"
    assert str(llm.openai_api_base) == "http://127.0.0.1:1234/v1"
