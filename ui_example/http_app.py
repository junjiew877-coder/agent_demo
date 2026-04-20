import json
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from claw.runtime import ChatRuntime
from claw.types import Context

chat_runtime: ChatRuntime | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global chat_runtime
    chat_runtime = ChatRuntime()
    try:
        yield
    finally:
        if chat_runtime is not None:
            await chat_runtime.close()
        chat_runtime = None


app = FastAPI(title="RayClaw Web", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequestBody(BaseModel):
    text: str
    messages: list[dict[str, str]] = Field(default_factory=list)
    model: str = "deepseek-chat"
    dialog_id: str = "web_session"
    user_id: str = "web_user"


def _sse_data(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _build_context(body: ChatRequestBody) -> Context:
    return Context(
        trace_id="",
        account_id=body.user_id,
        session_id=body.dialog_id,
        datetime=datetime.now(),
        location="Web",
        model=body.model,
    )


async def _chat_sse(body: ChatRequestBody) -> AsyncIterator[bytes]:
    assert chat_runtime is not None
    context = _build_context(body)
    history: list[dict[str, str]] = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in body.messages
        if m.get("content")
    ]
    query = body.text.strip()
    if not query:
        yield _sse_data({"type": "error", "message": "text is required"}).encode("utf-8")
        return

    async for event in chat_runtime.run(
        context=context, history=history, query=query
    ):
        if event.type == "response_stream_event":
            yield _sse_data({"type": "delta", "text": event.delta}).encode("utf-8")
        elif event.type == "reasoning_event":
            yield _sse_data({"type": "reasoning", "text": event.content}).encode(
                "utf-8"
            )
        elif event.type == "tool_finished":
            yield _sse_data(
                {
                    "type": "tool",
                    "tool_name": event.tool_name,
                    "preview": event.preview,
                }
            ).encode("utf-8")
        elif event.type == "done":
            yield _sse_data({"type": "done"}).encode("utf-8")


@app.post("/api/chat")
async def chat(body: ChatRequestBody):
    return StreamingResponse(
        _chat_sse(body),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


_public = Path(__file__).resolve().parent.parent.parent / "web" / "public"
if _public.is_dir():
    app.mount(
        "/ui",
        StaticFiles(directory=str(_public), html=True),
        name="ui",
    )
