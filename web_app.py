"""
工具调用智能体 Web 入口：与 RayClaw 一致的 UI（static/index.html）+ HTTP SSE 流式 /api/chat。

依赖: pip install -r requirements.txt（在 agent_demo 目录）

启动示例（仓库根目录 xr-rayclaw，便于 import agent_demo）:
  python -m uvicorn agent_demo.web_app:app --host 0.0.0.0 --port 8000

或在 agent_demo 目录且已配置 PYTHONPATH 指向父目录时:
  python -m uvicorn web_app:app --host 0.0.0.0 --port 8000

浏览器: http://127.0.0.1:8000/

环境变量:
  AGENT_USE_REASONER=1  使用思考模型；默认 0 为对话模型
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

load_dotenv()

from agent_demo import ReActAgent, build_agent

_agent: Optional[ReActAgent] = None

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequestBody(BaseModel):
    text: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    model: str = "deepseek-chat"
    dialog_id: str = "web_session"
    user_id: str = "web_user"


def _sse_data(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _chat_sse(body: ChatRequestBody) -> Iterator[bytes]:
    query = body.text.strip()
    if not query:
        yield _sse_data({"type": "error", "message": "text is required"}).encode("utf-8")
        yield _sse_data({"type": "done"}).encode("utf-8")
        return
    agent = _get_agent()
    try:
        for ev in agent.run_stream(query, web_model=body.model or None):
            yield _sse_data(ev).encode("utf-8")
    except Exception as e:
        yield _sse_data({"type": "error", "message": str(e)}).encode("utf-8")
        yield _sse_data({"type": "done"}).encode("utf-8")


app = FastAPI(title="Agent Demo Web", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    global _agent
    flag = os.getenv("AGENT_USE_REASONER", "0").strip().lower()
    use_reasoner = flag in ("1", "true", "yes", "on")
    _agent = build_agent(use_reasoner=use_reasoner)


def _get_agent() -> ReActAgent:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    return _agent


@app.get("/")
def index() -> FileResponse:
    path = STATIC_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="缺少 static/index.html，请从仓库中恢复该文件。",
        )
    return FileResponse(path, media_type="text/html; charset=utf-8")


@app.post("/api/chat")
async def chat(body: ChatRequestBody):
    """SSE：事件 type 为 delta | reasoning | tool | error | done（与 RayClaw Web 一致）。"""
    return StreamingResponse(
        _chat_sse(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/clear")
def api_clear() -> dict:
    _get_agent().clear_session()
    return {"ok": True}


@app.get("/api/health")
def api_health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def api_config() -> dict:
    ag = _get_agent()
    llm = ag.llm_client
    return {
        "model_profile": llm.get_model_profile(),
        "model_id": llm.model,
        "max_steps": ag.max_steps,
    }
