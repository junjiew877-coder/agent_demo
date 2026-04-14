"""
ReAct 智能体 Web 入口（非流式）。

启动（在项目根目录 agent_demo 下）:
  pip install -r requirements.txt
  uvicorn web_app:app --host 0.0.0.0 --port 8000

浏览器访问: http://127.0.0.1:8000/

环境变量（与现有 .env 一致，见 call_LLM / tool）:
  AGENT_USE_REASONER=1  使用思考模型；默认 0 为对话模型
"""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()

from agent_demo import ReActAgent, build_agent

_agent: Optional[ReActAgent] = None

INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ReAct 智能体</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.25rem; }
    textarea { width: 100%; min-height: 6rem; padding: 0.5rem; box-sizing: border-box; }
    button { margin-top: 0.5rem; padding: 0.4rem 1rem; cursor: pointer; }
    #out { margin-top: 1rem; white-space: pre-wrap; border: 1px solid #ccc; padding: 1rem; border-radius: 6px; min-height: 4rem; background: #fafafa; }
    .err { color: #b00020; }
    .meta { font-size: 0.85rem; color: #555; margin-top: 1rem; }
  </style>
</head>
<body>
  <h1>ReAct 智能体</h1>
  <p>输入问题后点击发送；多轮记忆在服务端保留，直至点击「清空会话」。</p>
  <textarea id="q" placeholder="例如：苹果当前的最新款手机是什么？"></textarea>
  <div>
    <button type="button" id="send">发送</button>
    <button type="button" id="clear">清空会话</button>
  </div>
  <div id="out"></div>
  <p class="meta">API: <code>POST /api/chat</code> JSON <code>{"message":"..."}</code> · <code>POST /api/clear</code></p>
  <script>
    const out = document.getElementById('out');
    const q = document.getElementById('q');
    document.getElementById('send').onclick = async () => {
      const message = q.value.trim();
      if (!message) { out.textContent = '请输入问题。'; out.className = 'err'; return; }
      out.textContent = '处理中…';
      out.className = '';
      try {
        const r = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message })
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || r.statusText);
        out.textContent = data.answer || '(空)';
        if (data.ok === false) out.className = 'err';
      } catch (e) {
        out.textContent = '请求失败: ' + e.message;
        out.className = 'err';
      }
    };
    document.getElementById('clear').onclick = async () => {
      try {
        const r = await fetch('/api/clear', { method: 'POST' });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || r.statusText);
        out.textContent = '已清空服务端会话记忆。';
        out.className = '';
      } catch (e) {
        out.textContent = '清空失败: ' + e.message;
        out.className = 'err';
      }
    };
  </script>
</body>
</html>"""


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)


class ChatResponse(BaseModel):
    answer: str
    ok: bool = True


app = FastAPI(title="ReAct Agent Web", version="1.0.0")


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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/api/chat", response_model=ChatResponse)
def api_chat(req: ChatRequest) -> ChatResponse:
    """同步执行整轮 ReAct，完成后一次性返回（非流式）。"""
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="message 不能为空")
    result = _get_agent().run(text)
    if result is None:
        return ChatResponse(
            answer="本次未得到 Finish 最终答案（可能达到最大步数或中途失败）。可查看运行终端日志。",
            ok=False,
        )
    return ChatResponse(answer=result, ok=True)


@app.post("/api/clear")
def api_clear() -> dict:
    _get_agent().clear_session()
    return {"ok": True}


@app.get("/api/health")
def api_health() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def api_config() -> dict:
    """当前模型档（不含任何密钥）。"""
    ag = _get_agent()
    llm = ag.llm_client
    return {
        "model_profile": llm.get_model_profile(),
        "model_id": llm.model,
        "max_steps": ag.max_steps,
    }
