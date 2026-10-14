"""
决策链（LangChain 1.x LCEL 重写）

替代目标侧 LLMClient.call_with_functions 的手写 Function Calling 迭代循环。

调用路径：主模型 →（异常/超时/无效输出）→ 备用模型 →（仍失败）→ 降级
（降级后的确定性五信号兜底由策略层执行，见 LLMHedgeFundStrategy._fallback_with_hf，
  两者合起来构成三级 graceful degradation）

推理模型兼容：OpenAI 兼容端点的思考内容（reasoning_content / reasoning）
落在 AIMessage.additional_kwargs，此处提取用于日志与经验记录；
content 为空而 tool_calls 存在属正常情况。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .decision_tools import DECISION_TOOLS, ToolExecutor
from . import providers

logger = logging.getLogger(__name__)

_ACTION_MAP = {
    "SignalEntryLong": "enter_long",
    "SignalEntryShort": "enter_short",
    "SignalExit": "exit",
    "AdjustPosition": "adjust",
    "SignalHold": "hold",
    "SignalWait": "wait",
}

_NUDGE = (
    "你刚才没有调用任何决策函数。请立即从六个决策函数中"
    "（signal_entry_long / signal_entry_short / signal_exit / "
    "adjust_position / signal_hold / signal_wait）选择恰好一个并调用。"
)


@dataclass
class DecisionOutcome:
    """一轮决策的结果（LLM 成功或降级）"""
    tool_name: Optional[str] = None
    action: Optional[str] = None          # enter_long/enter_short/exit/adjust/hold/wait
    args: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    trade_score: Optional[float] = None
    degraded: bool = False
    degrade_reason: str = ""
    raw_content: str = ""                 # LLM 原文（含推理摘要，供经验日志）
    elapsed_ms: int = 0
    model_used: str = ""


def _extract_reasoning(ai_msg: AIMessage) -> str:
    """兼容推理模型：提取 reasoning_content/reasoning 字段（若有）"""
    kw = getattr(ai_msg, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning", "think"):
        val = kw.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


class DecisionChain:
    """一次决策 = system + user → bind_tools 模型 → tool_calls → ToolExecutor"""

    def __init__(self, llm_config: Dict[str, Any], tool_executor: ToolExecutor):
        self.executor = tool_executor
        primary = providers.get_llm(llm_config, "primary")
        backup = providers.get_llm(llm_config, "backup")

        self._models: List = []
        if primary is not None:
            self._models.append((primary.bind_tools(DECISION_TOOLS), primary.model_name))
        if backup is not None:
            self._models.append((backup.bind_tools(DECISION_TOOLS), backup.model_name))

        if not self._models:
            logger.error("主备模型均未配置，DecisionChain 只能走降级路径")

    # ------------------------------------------------------------------

    def decide(self, system_prompt: str, user_message: str) -> DecisionOutcome:
        """执行一轮决策：主 → 备 → 降级"""
        for bound, model_name in self._models:
            started = time.monotonic()
            try:
                outcome = self._invoke_model(bound, model_name, system_prompt, user_message)
                outcome.elapsed_ms = int((time.monotonic() - started) * 1000)
                if not outcome.degraded:
                    return outcome
                # 模型返回了但无有效决策（未调函数/参数被拒）→ 记录后尝试下一个模型
                logger.warning(f"[{model_name}] {outcome.degrade_reason}，尝试下一角色")
                last = outcome
            except Exception as e:
                logger.warning(f"[{model_name}] 调用异常: {e}", exc_info=True)
                last = DecisionOutcome(
                    degraded=True,
                    degrade_reason=f"模型调用异常: {e}",
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    model_used=model_name,
                )

        return last

    # ------------------------------------------------------------------

    def _invoke_model(
        self, bound, model_name: str, system_prompt: str, user_message: str
    ) -> DecisionOutcome:
        messages: List = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        ai_msg: AIMessage = bound.invoke(messages)

        reasoning = _extract_reasoning(ai_msg)
        content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        # 未调用函数：追加提醒重试一次
        if not tool_calls:
            logger.info(f"[{model_name}] 首次未调用决策函数，重试一次")
            messages = messages + [
                AIMessage(content=content or "(no content)"),
                HumanMessage(content=_NUDGE),
            ]
            ai_msg = bound.invoke(messages)
            reasoning = _extract_reasoning(ai_msg) or reasoning
            content = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                return DecisionOutcome(
                    degraded=True,
                    degrade_reason="模型两次均未调用决策函数",
                    raw_content=self._join_content(content, reasoning),
                    model_used=model_name,
                )

        # 多个调用时取最后一个（终态语义）
        tc = tool_calls[-1]
        name = tc.get("name", "")
        raw_args = tc.get("args") or {}

        result = self.executor.execute(name, dict(raw_args))
        raw_content = self._join_content(content, reasoning)

        if not result.get("success"):
            return DecisionOutcome(
                tool_name=name,
                degraded=True,
                degrade_reason=f"决策函数被拒绝: {result.get('message', '')}",
                raw_content=raw_content,
                model_used=model_name,
            )

        args = result.get("args", {})
        return DecisionOutcome(
            tool_name=name,
            action=_ACTION_MAP.get(name, name),
            args=args,
            confidence=float(args.get("confidence_score", 0.0)),
            reason=str(args.get("reason", "")),
            trade_score=args.get("trade_score"),
            raw_content=raw_content,
            model_used=model_name,
        )

    @staticmethod
    def _join_content(content: str, reasoning: str) -> str:
        if reasoning:
            shown = reasoning if len(reasoning) <= 2000 else reasoning[:2000] + "…"
            return f"{content}\n[推理摘要] {shown}"
        return content
