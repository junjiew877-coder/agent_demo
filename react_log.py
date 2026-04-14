"""
ReAct / LLM / 工具 的同步日志：始终 print，且在 Web 请求绑定了缓冲区时追加到列表。
避免 call_LLM ↔ agent_demo 循环导入。
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from typing import List, Optional

_TRACE_LINES: ContextVar[Optional[List[str]]] = ContextVar("TRACE_LINES", default=None)


def trace_line(msg: str) -> None:
    """写入终端；若当前在 capture 中，同时追加一行到缓冲区。"""
    print(msg, flush=True)
    buf = _TRACE_LINES.get()
    if buf is not None:
        buf.append(msg)


def begin_trace_capture(lines: List[str]) -> Token:
    """开始把 trace_line 复制到 lines。"""
    return _TRACE_LINES.set(lines)


def end_trace_capture(token: Token) -> None:
    _TRACE_LINES.reset(token)
