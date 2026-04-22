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
    """
    与 begin_trace_capture 成对使用。注意：在 FastAPI StreamingResponse(SSE) 中，
    生成器在 yield 之后可能处于与 set() 时不同的 Context，reset(token) 会抛
    ValueError: ... was created in a different Context。此时直接清空绑定即可；
    日志已写入传入的 list 对象，不受影响。
    """
    try:
        _TRACE_LINES.reset(token)
    except ValueError:
        _TRACE_LINES.set(None)
