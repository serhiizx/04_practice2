"""Спільні стаби для офлайн-тестів LangGraph-графа: без мережі й без LLM.

StubLLM підтримує лише той інтерфейс, який реально використовує
langgraph_mas.py: with_structured_output(Schema).ainvoke(...). StubTool
відтворює реальний контракт MCP-інструмента (langchain_mcp_adapters повертає
список content-блоків, а не рядок напряму — див. _tool_result_text у
langgraph_mas.py). StubReactAgent підмінює create_react_agent для вузла
requirements_matcher, який лишається ReAct-агентом.
"""

import json

from langchain_core.messages import AIMessage


class StubTool:
    """Мінімальний двійник BaseTool: .name для tools_for/allowlist,
    ainvoke(args) — асинхронний виклик, що повертає результат у форматі
    реального MCP-адаптера: [{"type": "text", "text": "<json-рядок>"}]."""

    def __init__(self, name: str, result: dict | None = None):
        self.name = name
        self._result = result

    async def ainvoke(self, _args):
        return [{"type": "text", "text": json.dumps(self._result)}]


class _StubStructured:
    """Повертає з наперед заданої черги обʼєкт, що відповідає schema."""

    def __init__(self, parent: "StubLLM", schema):
        self._parent = parent
        self._schema = schema

    async def ainvoke(self, messages):
        self._parent.calls.append((self._schema, messages))
        queue = self._parent.responses[self._schema]
        # Останню відповідь у черзі повторюємо, якщо викликів більше, ніж заготовок.
        return queue.pop(0) if len(queue) > 1 else queue[0]


class StubLLM:
    """Стаб LLM: with_structured_output(Schema).ainvoke(...) -> наперед заданий обʼєкт.

    responses: {Schema: обʼєкт | список обʼєктів (черга по порядку викликів)}.
    """

    def __init__(self, responses: dict[type, object]):
        self.responses = {
            schema: (value if isinstance(value, list) else [value])
            for schema, value in responses.items()
        }
        self.calls: list[tuple[type, list]] = []

    def with_structured_output(self, schema):
        return _StubStructured(self, schema)


class StubReactAgent:
    """Двійник об'єкта, що повертає create_react_agent: лише ainvoke, що
    віддає наперед задану історію повідомлень (включно з ToolMessage, якщо
    тест перевіряє, що вузол читає результат інструмента з історії, а не з
    фінального тексту)."""

    def __init__(self, messages: list):
        self.messages = messages

    async def ainvoke(self, _input):
        return {"messages": self.messages}


def stub_react_agent_with_text(final_text: str) -> StubReactAgent:
    """Зручний конструктор для випадків, коли важливий лише фінальний текст."""
    return StubReactAgent([AIMessage(content=final_text)])
