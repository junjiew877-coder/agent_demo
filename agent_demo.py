import argparse
import os
from typing import List, Optional, Tuple

from call_LLM import HelloAgentsLLM, get_default_model_ids
from tool import ToolExecutor, search, get_current_time
import re
import sys
from prompt import REACT_PROMPT_TEMPLATE

_EXIT_CMDS = frozenset({"/exit", "/quit", "exit", "quit", ":q", "/q"})
_HELP_CMDS = frozenset({"/help", "/?", "help"})
_CLEAR_CMDS = frozenset({"/clear", "/reset"})

# 使用 deepseek-reasoner 时，单轮问答内 LLM 的最大调用步数（与 ReAct 循环一致）
REASONER_AGENT_MAX_STEPS = 10
CHAT_AGENT_MAX_STEPS = int(os.getenv("LLM_CHAT_AGENT_MAX_STEPS", "10"))


def _agent_max_steps_for_llm(llm: HelloAgentsLLM) -> int:
    return REASONER_AGENT_MAX_STEPS if llm.use_reasoner else CHAT_AGENT_MAX_STEPS


def _parse_tool_call_line(action_line: str) -> Tuple[Optional[str], Optional[str]]:
    """
    解析单行工具调用：ToolName[...] 或 Finish[...]。
    使用括号深度匹配第一个 [...]，避免贪婪正则吞到文末最后一个 ]。
    """
    s = (action_line or "").strip()
    m = re.match(r"^(\w+)\[", s)
    if not m:
        return None, None
    name = m.group(1)
    i = m.end()
    depth = 1
    start = i
    while i < len(s):
        c = s[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return name, s[start:i]
        i += 1
    return None, None


class ReActAgent:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 10):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history: List[str] = []
        # 跨用户输入保留，直至 /clear
        self.session_memory: List[str] = []

    def clear_session(self) -> None:
        """清空多轮会话记忆（不重置工具与模型配置）。"""
        self.session_memory.clear()

    def _format_history_for_prompt(self) -> str:
        """组合历史会话与当前问题的 ReAct 轨迹，供模板 History 字段使用。"""
        blocks: List[str] = []
        if self.session_memory:
            prev = "\n\n---\n\n".join(self.session_memory)
            blocks.append("【此前多轮会话】\n" + prev)
        if self.history:
            blocks.append("【当前问题的工具轨迹】\n" + "\n".join(self.history))
        if blocks:
            return "\n\n".join(blocks)
        return "（尚无历史；这是会话中的第一个问题。）"

    def _append_session_turn(
        self, question: str, final_answer: Optional[str], react_trace: List[str]
    ) -> None:
        """将本轮问答写入会话，供后续 query 使用。"""
        parts = [f"【用户】\n{question.strip()}"]
        if react_trace:
            parts.append("【本问题内工具轨迹】\n" + "\n".join(react_trace))
        if final_answer is not None:
            parts.append(f"【助手最终答复】\n{final_answer.strip()}")
        else:
            parts.append("【助手最终答复】\n（本轮未完成、未以 Finish 结束或已达步数上限。）")
        self.session_memory.append("\n".join(parts))

    def run(self, question: str) -> Optional[str]:
        self.history = []
        outcome: Optional[str] = None
        current_step = 0

        try:
            while current_step < self.max_steps:
                current_step += 1
                print(f"\n--- 第 {current_step} 步 ---")

                tools_desc = self.tool_executor.getAvailableTools()
                history_str = self._format_history_for_prompt()
                prompt = REACT_PROMPT_TEMPLATE.format(
                    tools=tools_desc, question=question, history=history_str
                )

                messages = [{"role": "user", "content": prompt}]
                response_text = self.llm_client.think(messages=messages)
                if not response_text:
                    print("错误：LLM未能返回有效响应。")
                    break

                thought, action = self._parse_output(response_text)
                if thought:
                    print(f"🤔 LLM输出：{thought}")
                if not action:
                    print("警告：未能解析出有效的Action，流程终止。")
                    break

                if action.lstrip().lower().startswith("finish"):
                    final_answer = self._parse_action_input(action)
                    print(f"🎉 最终答案: {final_answer}")
                    outcome = final_answer
                    return outcome

                tool_name, tool_input = self._parse_action(action)
                if not tool_name:
                    self.history.append("Observation: 无效的Action格式，请检查。")
                    continue

                print(f"🎬 行动: {tool_name}[{tool_input}]")
                tool_function = self.tool_executor.getTool(tool_name)
                observation = (
                    tool_function(tool_input)
                    if tool_function
                    else f"错误：未找到名为 '{tool_name}' 的工具。"
                )

                print(f"👀 观察: {observation}")
                # 只写入规范化后的 Action，避免把模型多写的伪 Observation/第二 Action 带进 History
                canonical = f"{tool_name}[{tool_input}]"
                self.history.append(f"Action: {canonical}")
                self.history.append(f"Observation: {observation}")

            print("已达到最大步数，流程终止。")
            return None
        finally:
            self._append_session_turn(question, outcome, list(self.history))

    def _parse_output(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        取 **最后一个** 行首的 `Action:` 之后的内容作为唯一动作（单行），避免模型
        在正文中多次书写 Action/Observation 时误把整段吞进一个 action。
        """
        raw = (text or "").strip("\n")
        if not raw:
            return None, None

        line_matches = list(
            re.finditer(r"(?mi)^\s*Action:\s*(.*)$", raw, flags=re.MULTILINE)
        )
        action_line: Optional[str] = None
        cut = 0
        if line_matches:
            last_m = line_matches[-1]
            action_line = (last_m.group(1) or "").strip()
            cut = last_m.start()
        else:
            idx = raw.lower().rfind("action:")
            if idx < 0:
                return None, None
            rest = raw[idx + len("action:") :].lstrip()
            action_line = rest.split("\n", 1)[0].strip()
            cut = idx

        prefix = raw[:cut].strip()
        thought: Optional[str] = None
        if prefix:
            thought = re.sub(
                r"(?is)^\s*Thought:\s*",
                "",
                prefix,
                count=1,
            ).strip()
            if not thought:
                thought = prefix

        if not action_line:
            return thought, None
        return thought, action_line

    def _parse_action(self, action_text: str) -> Tuple[Optional[str], Optional[str]]:
        return _parse_tool_call_line(action_text)

    def _parse_action_input(self, action_text: str) -> str:
        _, inner = _parse_tool_call_line(action_text)
        return inner if inner is not None else ""


def build_agent(use_reasoner: bool = False) -> ReActAgent:
    llm = HelloAgentsLLM(use_reasoner=use_reasoner)
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    time_desc = (
        "从操作系统读取当前真实的本地日期与时间（含星期）。涉及「最新」「今年」「当前」等时效问题时，"
        "应先调用本工具确认日期，再构造 Search 的检索词；参数可留空，例如 GetCurrentTime[]。"
    )
    tool_executor.registerTool("GetCurrentTime", time_desc, get_current_time)
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
    """交互选择是否使用思考模型。返回 True 表示 reasoner。"""
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
    """
    处理 /model 命令。返回 True 表示已处理（应跳过后续 agent.run）。
    """
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
            f"  · ReAct 最大步数: {agent.max_steps}"
        )
    except ValueError as e:
        print(str(e))
    return True


def _read_user_message() -> Optional[str]:
    """读取一条用户消息。EOF（Ctrl+Z+回车 / Ctrl+D）返回 None 表示结束会话。"""
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
        "ReAct 智能体 — 交互模式\n"
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReAct 智能体（交互式 CLI）")
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
