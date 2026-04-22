import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from call_LLM import HelloAgentsLLM, get_default_model_ids
from tool.executor import ToolExecutor
from tool.search import search
from tool.current_time import get_current_time
from prompt import TOOL_AGENT_SYSTEM_PROMPT
from react_log import begin_trace_capture, end_trace_capture, trace_line

_EXIT_CMDS = frozenset({"/exit", "/quit", "exit", "quit", ":q", "/q"})
_HELP_CMDS = frozenset({"/help", "/?", "help"})
_CLEAR_CMDS = frozenset({"/clear", "/reset"})

REASONER_AGENT_MAX_STEPS = 10
CHAT_AGENT_MAX_STEPS = int(os.getenv("LLM_CHAT_AGENT_MAX_STEPS", "10"))

TOOL_PREVIEW_MAX = 400


def _agent_max_steps_for_llm(llm: HelloAgentsLLM) -> int:
    return REASONER_AGENT_MAX_STEPS if llm.use_reasoner else CHAT_AGENT_MAX_STEPS


class ReActAgent:
    """
    使用 OpenAI 兼容 Chat Completions 的 tool calling（tool_choice=auto）与多轮消息。
    会话记忆为不含 system 的 OpenAI 风格 message 列表。
    """

    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 10):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.session_memory: List[Dict[str, Any]] = []
        self.last_run_trace: List[str] = []

    def clear_session(self) -> None:
        self.session_memory.clear()

    def run(self, question: str) -> Optional[str]:
        self.last_run_trace = []
        cap_token = begin_trace_capture(self.last_run_trace)
        outcome: Optional[str] = None
        tools = self.tool_executor.to_openai_tools()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": TOOL_AGENT_SYSTEM_PROMPT},
            *list(self.session_memory),
            {"role": "user", "content": question.strip()},
        ]
        turn_start = len(messages) - 1

        try:
            current_step = 0
            while current_step < self.max_steps:
                current_step += 1
                trace_line(f"\n--- 第 {current_step} 步 ---")

                result = self.llm_client.complete_with_tools(messages, tools)
                if not result:
                    trace_line("错误：LLM未能返回有效响应。")
                    break

                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if result.tool_calls:
                    assistant_msg["tool_calls"] = result.tool_calls
                    assistant_msg["content"] = (
                        result.content if result.content is not None else ""
                    )
                else:
                    assistant_msg["content"] = (
                        result.content if result.content is not None else ""
                    )
                if result.reasoning_content:
                    assistant_msg["reasoning_content"] = result.reasoning_content

                messages.append(assistant_msg)

                if not result.tool_calls:
                    outcome = assistant_msg.get("content") or ""
                    trace_line(f"🎉 最终答案: {outcome}")
                    return outcome or None

                for tc in result.tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    args = fn.get("arguments") or "{}"
                    preview = args if len(args) <= 200 else args[:200] + "…"
                    trace_line(f"🛠 工具调用: {name}({preview})")
                    observation = self.tool_executor.invoke(name, args)
                    obs_preview = (
                        observation
                        if len(observation) <= 500
                        else observation[:500] + "…"
                    )
                    trace_line(f"👀 工具结果: {obs_preview}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": observation,
                        }
                    )

            trace_line("已达到最大步数，流程终止。")
            return None
        finally:
            try:
                self.session_memory.extend(messages[turn_start:])
            finally:
                end_trace_capture(cap_token)

    def run_stream(self, question: str, web_model: Optional[str] = None):
        """
        与 run() 相同对话逻辑，但通过 yield 产出 RayClaw 兼容的 SSE 事件 dict：
        delta / reasoning / tool / error / done。
        """
        self.last_run_trace = []
        cap_token = begin_trace_capture(self.last_run_trace)
        tools = self.tool_executor.to_openai_tools()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": TOOL_AGENT_SYSTEM_PROMPT},
            *list(self.session_memory),
            {"role": "user", "content": question.strip()},
        ]
        turn_start = len(messages) - 1
        saved_model: Optional[str] = None
        if web_model and web_model.strip():
            saved_model = self.llm_client.model
            self.llm_client.model = web_model.strip()
        try:
            current_step = 0
            while current_step < self.max_steps:
                current_step += 1
                trace_line(f"\n--- 第 {current_step} 步 ---")

                result = None
                for ev in self.llm_client.stream_complete_with_tools(messages, tools):
                    k = ev.get("kind")
                    if k == "delta":
                        yield {"type": "delta", "text": ev.get("text") or ""}
                    elif k == "reasoning":
                        yield {"type": "reasoning", "text": ev.get("text") or ""}
                    elif k == "complete":
                        result = ev.get("result")

                if result is None:
                    trace_line("错误：LLM未能返回有效响应。")
                    yield {"type": "error", "message": "LLM 调用失败或返回无效"}
                    yield {"type": "done"}
                    return

                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                if result.tool_calls:
                    assistant_msg["tool_calls"] = result.tool_calls
                    assistant_msg["content"] = (
                        result.content if result.content is not None else ""
                    )
                else:
                    assistant_msg["content"] = (
                        result.content if result.content is not None else ""
                    )
                if result.reasoning_content:
                    assistant_msg["reasoning_content"] = result.reasoning_content

                messages.append(assistant_msg)

                if not result.tool_calls:
                    yield {"type": "done"}
                    return

                for tc in result.tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name") or ""
                    args = fn.get("arguments") or "{}"
                    preview = args
                    if len(preview) > TOOL_PREVIEW_MAX:
                        preview = preview[:TOOL_PREVIEW_MAX] + "…"
                    trace_line(f"🛠 工具调用: {name}({args[:200]}{'…' if len(args) > 200 else ''})")
                    observation = self.tool_executor.invoke(name, args)
                    obs_preview = (
                        observation
                        if len(observation) <= TOOL_PREVIEW_MAX
                        else observation[:TOOL_PREVIEW_MAX] + "…"
                    )
                    trace_line(f"👀 工具结果: {obs_preview}")
                    yield {"type": "tool", "tool_name": name, "preview": obs_preview}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": observation,
                        }
                    )

            trace_line("已达到最大步数，流程终止。")
            yield {"type": "done"}
        finally:
            if saved_model is not None:
                self.llm_client.model = saved_model
            try:
                self.session_memory.extend(messages[turn_start:])
            finally:
                end_trace_capture(cap_token)


def build_agent(use_reasoner: bool = False) -> ReActAgent:
    llm = HelloAgentsLLM(use_reasoner=use_reasoner)
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool(
        "Search",
        search_desc,
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索引擎用的检索词或问题短语。",
                },
            },
            "required": ["query"],
        },
        lambda args: search(args.get("query", "")),
    )
    time_desc = (
        "从操作系统读取当前真实的本地日期与时间（含星期）。涉及「最新」「今年」「当前」等时效问题时，"
        "应先调用本工具确认日期，再构造 Search 的检索词。"
    )
    tool_executor.registerTool(
        "GetCurrentTime",
        time_desc,
        {"type": "object", "properties": {}},
        lambda _args: get_current_time(),
    )
    return ReActAgent(
        llm_client=llm,
        tool_executor=tool_executor,
        max_steps=_agent_max_steps_for_llm(llm),
    )


def _print_help() -> None:
    print(
        "  单行：在提示符后输入问题，回车即发送。\n"
        "  /multi     多行输入模式；输入完毕后单独一行输入 END 结束。\n"
        "  /model     查看当前模型档；/model chat 或 /model reasoner 切换。\n"
        "  /clear     清空多轮会话记忆（此前问答不再带入后续问题）。\n"
        "  /help      显示本说明\n"
        "  /exit      退出（也可用 quit、:q）\n"
        "  Ctrl+C     取消当前输入\n"
    )


def _prompt_model_choice() -> bool:
    chat_id, reasoner_id = get_default_model_ids()
    default = os.getenv("LLM_DEFAULT_PROFILE", "chat").strip().lower()
    default_is_reasoner = default in ("reasoner", "think", "thinking", "r1")
    default_hint = "2" if default_is_reasoner else "1"

    print(
        "选择运行模型（可在会话中用 /model 切换）：\n"
        f"  [1] 对话模型 — {chat_id}\n"
        f"  [2] 思考模型 — {reasoner_id}（链式推理 + 正式输出）\n"
    )
    try:
        choice = input(f"请输入 1 或 2（直接回车默认选 [{default_hint}]）: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消启动。")
        raise SystemExit(0) from None

    if not choice:
        return default_is_reasoner
    if choice == "2":
        return True
    if choice == "1":
        return False
    print("输入无效，已按默认处理。")
    return default_is_reasoner


def _handle_model_command(agent: ReActAgent, text: str) -> bool:
    parts = text.split(maxsplit=1)
    llm = agent.llm_client
    if len(parts) == 1:
        print(
            f"当前模型档: {llm.get_model_profile()}  （实际 id: {llm.model}）\n"
            "切换: /model chat  或  /model reasoner"
        )
        return True
    try:
        llm.set_model_profile(parts[1])
        agent.max_steps = _agent_max_steps_for_llm(llm)
        print(
            f"已切换为 {llm.get_model_profile()}  （{llm.model}）"
            f"  · 工具调用最大步数: {agent.max_steps}"
        )
    except ValueError as e:
        print(str(e))
    return True


def _read_user_message() -> Optional[str]:
    try:
        first = input("› ").rstrip("\n")
    except EOFError:
        return None
    except KeyboardInterrupt:
        print("^C")
        return ""

    if not first.strip():
        return ""

    if first.strip().lower() == "/multi":
        print("（多行模式：输入完毕后单独一行输入 END 结束）")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return None
            except KeyboardInterrupt:
                print("^C")
                return ""
            if line.strip().upper() == "END":
                break
            lines.append(line)
        return "\n".join(lines).strip()

    return first.strip()


def run_interactive_cli(use_reasoner: Optional[bool] = None) -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass

    print(
        "工具调用智能体（OpenAI tools）— 交互模式\n"
        "在 › 后输入问题并回车；长文本可先输入 /multi，结束时单独一行输入 END。\n"
        "命令：/help  /model  /clear  /exit\n"
    )
    if use_reasoner is None:
        use_reasoner = _prompt_model_choice()
    else:
        label = "思考模型 (reasoner)" if use_reasoner else "对话模型 (chat)"
        print(f"启动参数已指定：{label}\n")

    agent = build_agent(use_reasoner=use_reasoner)
    print(f"当前使用: {agent.llm_client.get_model_profile()} — {agent.llm_client.model}\n")

    while True:
        try:
            raw = _read_user_message()
        except KeyboardInterrupt:
            print("\n再见。")
            break

        if raw is None:
            print("\n再见。")
            break

        text = raw.strip()
        if not text:
            continue

        low = text.lower()
        if low in _EXIT_CMDS:
            print("再见。")
            break
        if low in _HELP_CMDS:
            _print_help()
            continue

        if low in _CLEAR_CMDS:
            agent.clear_session()
            print("已清空会话记忆，后续问题将不带入此前多轮内容。\n")
            continue

        if low.startswith("/model"):
            _handle_model_command(agent, text)
            continue

        print()
        agent.run(text)
        print("\n" + "─" * 48 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="工具调用智能体（交互式 CLI）")
    g = parser.add_mutually_exclusive_group()
    g.add_argument(
        "--chat",
        action="store_true",
        help="直接使用对话模型（如 deepseek-chat），跳过启动时选择",
    )
    g.add_argument(
        "--reasoner",
        action="store_true",
        help="直接使用思考模型（如 deepseek-reasoner），跳过启动时选择",
    )
    args = parser.parse_args()
    if args.reasoner:
        run_interactive_cli(use_reasoner=True)
    elif args.chat:
        run_interactive_cli(use_reasoner=False)
    else:
        run_interactive_cli(use_reasoner=None)
