import os
import unicodedata
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple

# 加载 .env 文件中的环境变量
load_dotenv()

from react_log import trace_line


def _env_model_chat() -> str:
    return (
        os.getenv("LLM_MODEL_CHAT")
        or os.getenv("LLM_MODEL_ID")
        or "deepseek-chat"
    )


def _env_model_reasoner() -> str:
    return (
        os.getenv("LLM_MODEL_REASONER")
        or os.getenv("LLM_MODEL_ID_THOUGHT")
        or "deepseek-reasoner"
    )


def get_default_model_ids() -> Tuple[str, str]:
    """返回 (对话模型 id, 思考模型 id)，供 CLI 展示与配置。"""
    return (_env_model_chat(), _env_model_reasoner())


def _char_display_width(c: str) -> int:
    ea = unicodedata.east_asian_width(c)
    return 2 if ea in ("F", "W") else 1


def _str_display_width(s: str) -> int:
    return sum(_char_display_width(c) for c in s)


def _wrap_to_display_width(text: str, max_dw: int) -> List[str]:
    """按终端显示宽度折行（中日文等按双宽计）。"""
    if max_dw < 8:
        max_dw = 8
    if not text:
        return [""]
    lines: List[str] = []
    buf: List[str] = []
    w = 0
    for c in text:
        cw = _char_display_width(c)
        if w + cw > max_dw and buf:
            lines.append("".join(buf))
            buf = [c]
            w = cw
        else:
            buf.append(c)
            w += cw
    if buf:
        lines.append("".join(buf))
    return lines


def _pad_to_display_width(text: str, target_dw: int) -> str:
    out = list(text)
    w = _str_display_width(text)
    while w < target_dw:
        out.append(" ")
        w += 1
    return "".join(out)


def _print_reasoning_box(reasoning_text: str) -> None:
    """将链式思考完整包在同一宽度的框内，长行自动折行，避免内容超出竖线。"""
    try:
        term_cols = max(48, os.get_terminal_size().columns)
    except OSError:
        term_cols = 100
    # 与「┃  …  ┃」对齐：左右各 2 个半宽字符的显示宽度
    inner = max(32, min(120, term_cols - 4))
    bar_len = inner + 2

    title_top = "《Thought · 链式思考》模型内部推理"
    title_bot = "《Thought 结束》"

    body_rows: List[str] = []
    for para in reasoning_text.rstrip().split("\n"):
        if not para:
            body_rows.append("")
            continue
        body_rows.extend(_wrap_to_display_width(para, inner))

    trace_line(f"\n┏{'═' * bar_len}┓")
    for t in _wrap_to_display_width(title_top, inner):
        trace_line("┃ " + _pad_to_display_width(t, inner) + " ┃")
    trace_line(f"┣{'─' * bar_len}┫")
    for row in body_rows:
        for seg in _wrap_to_display_width(row, inner):
            trace_line("┃ " + _pad_to_display_width(seg, inner) + " ┃")
    trace_line(f"┣{'─' * bar_len}┫")
    for t in _wrap_to_display_width(title_bot, inner):
        trace_line("┃ " + _pad_to_display_width(t, inner) + " ┃")
    trace_line(f"┗{'═' * bar_len}┛\n")


class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    支持 DeepSeek：对话模型（如 deepseek-chat）与思考模型（deepseek-reasoner）。
    """
    def __init__(
        self,
        model: Optional[str] = None,
        apiKey: Optional[str] = None,
        baseUrl: Optional[str] = None,
        timeout: Optional[int] = None,
        *,
        use_reasoner: bool = False,
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        use_reasoner: True 时使用思考模型（默认读取 LLM_MODEL_REASONER / LLM_MODEL_ID_THOUGHT）。
        若显式传入 model，则直接使用该 id，且不再根据 use_reasoner 切换 id（但仍可按模型名处理推理流）。
        """
        self._model_chat = _env_model_chat()
        self._model_reasoner = _env_model_reasoner()

        if model:
            self.model = model
            self.use_reasoner = use_reasoner or (
                "reasoner" in model.lower() or "r1" in model.lower()
            )
        else:
            self.use_reasoner = bool(use_reasoner)
            self.model = self._model_reasoner if self.use_reasoner else self._model_chat

        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def get_model_profile(self) -> str:
        """当前配置档：'reasoner' 或 'chat'。"""
        return "reasoner" if self.use_reasoner else "chat"

    def set_model_profile(self, profile: str) -> None:
        """切换为思考模型或对话模型。profile: 'chat' | 'reasoner'（大小写不敏感）。"""
        p = (profile or "").strip().lower()
        if p in ("chat", "c", "0"):
            self.use_reasoner = False
            self.model = self._model_chat
        elif p in ("reasoner", "think", "thinking", "r1", "r", "1"):
            self.use_reasoner = True
            self.model = self._model_reasoner
        else:
            raise ValueError(
                f"未知的模型档: {profile!r}，请使用 chat 或 reasoner。"
            )

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> Optional[str]:
        """
        调用大语言模型进行思考，并返回其响应。
        思考模型（deepseek-reasoner）流式返回 reasoning_content 与 content；
        ReAct 解析仅使用 content（正式回复），链式思考可选打印。
        """
        trace_line(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            show_chain = os.getenv("LLM_SHOW_CHAIN_THOUGHT", "1").strip().lower() not in (
                "0", "false", "no", "off",
            )
            reasoning_parts: List[str] = []
            collected_content: List[str] = []

            for chunk in response:
                delta = chunk.choices[0].delta
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_parts.append(rc)
                content = delta.content or ""
                collected_content.append(content)

            reasoning_text = "".join(reasoning_parts)
            content_text = "".join(collected_content)

            trace_line("✅ 大语言模型响应成功")

            if self.use_reasoner and reasoning_text and show_chain:
                _print_reasoning_box(reasoning_text)

            # 思考模型若 content 为空，则回退到推理文本（避免完全无输出）
            out = content_text if content_text.strip() else reasoning_text
            return out if out else None

        except Exception as e:
            trace_line(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = HelloAgentsLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "今年是哪一年"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        # if responseText:
        #     print("\n\n--- 完整模型响应 ---")
        #     print(responseText)

    except ValueError as e:
        print(e)
