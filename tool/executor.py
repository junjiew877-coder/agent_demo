import json
from typing import Any, Callable, Dict, List

ToolFunc = Callable[[Dict[str, Any]], str]


class ToolExecutor:
    """
    管理工具注册、OpenAI tools 列表生成，以及按 JSON 参数执行工具。
    """

    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: ToolFunc,
    ) -> None:
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {
            "description": description,
            "parameters": parameters,
            "func": func,
        }
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> ToolFunc | None:
        entry = self.tools.get(name)
        return entry["func"] if entry else None

    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """供 chat.completions.create(..., tools=...) 使用。"""
        out: List[Dict[str, Any]] = []
        for name, info in self.tools.items():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": info["description"],
                        "parameters": info["parameters"],
                    },
                }
            )
        return out

    def invoke(self, name: str, arguments_json: str) -> str:
        entry = self.tools.get(name)
        if not entry:
            return f"错误：未找到名为 '{name}' 的工具。"
        try:
            args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as e:
            return f"错误：工具参数不是合法 JSON: {e}"
        if not isinstance(args, dict):
            return "错误：工具参数必须是 JSON 对象。"
        try:
            return entry["func"](args)
        except Exception as e:
            return f"工具执行出错: {e}"
