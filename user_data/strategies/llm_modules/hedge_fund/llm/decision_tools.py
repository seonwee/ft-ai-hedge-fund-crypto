"""
决策工具层（LangChain 1.x 重写）

替代目标侧手写的 JSON schema + TradingTools 参数校验：
- 6 个决策函数的参数用 pydantic 模型定义，langchain bind_tools 自动生成
  OpenAI tool schema（描述即文档），校验由 pydantic 声明式完成
- ToolExecutor 负责执行：校验通过后写入策略缓存（信号/杠杆/金额/仓位调整），
  缓存契约与 ft-userdata-llm-share 完全一致，策略层代码无缝复用

动作语义映射自源侧输出契约 {action: buy/sell/short/cover/hold, quantity,
confidence, reasoning}（portfolio_management_node.py:149）：
  buy → signal_entry_long；short → signal_entry_short
  sell/cover → signal_exit；hold → signal_hold（有仓）/ signal_wait（无仓）
  （加/减仓 → adjust_position，源侧无对应，为目标侧增强能力）
"""

import logging
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 参数模型（langchain bind_tools 直接消费）
# ---------------------------------------------------------------------------

class _DecisionBase(BaseModel):
    pair: str = Field(description="交易对，例如 BTC/USDT:USDT")
    confidence_score: float = Field(
        ge=1, le=100,
        description="决策置信度 (1-100)。>80 高信心，60-80 中等，<60 低信心",
    )
    reason: str = Field(
        min_length=10,
        description="决策理由，必须引用至少一个具体数值/指标，禁止空泛表述",
    )


class SignalEntryLong(_DecisionBase):
    """做多开仓，指定杠杆与投入金额"""
    leverage: float = Field(ge=1, le=100, description="杠杆倍数 (1-100)")
    key_support: float = Field(description="关键支撑位价格")
    key_resistance: float = Field(description="关键阻力位价格")
    rsi_value: float = Field(ge=0, le=100, description="当前 RSI 数值 (0-100)")
    trend_strength: str = Field(description="趋势强度评估: '强势' | '中等' | '弱势'")
    stake_amount: Optional[float] = Field(
        default=None, gt=0,
        description="本次投入的 USDT 保证金金额（留空则由策略按预算分配）",
    )


class SignalEntryShort(_DecisionBase):
    """做空开仓，指定杠杆与投入金额"""
    leverage: float = Field(ge=1, le=100, description="杠杆倍数 (1-100)")
    key_support: float = Field(description="关键支撑位价格")
    key_resistance: float = Field(description="关键阻力位价格")
    rsi_value: float = Field(ge=0, le=100, description="当前 RSI 数值 (0-100)")
    trend_strength: str = Field(description="趋势强度评估: '强势' | '中等' | '弱势'")
    stake_amount: Optional[float] = Field(
        default=None, gt=0,
        description="本次投入的 USDT 保证金金额（留空则由策略按预算分配）",
    )


class SignalExit(_DecisionBase):
    """平掉当前持仓，并对本次交易自我评价"""
    rsi_value: float = Field(ge=0, le=100, description="当前 RSI 数值 (0-100)")
    trade_score: float = Field(
        ge=0, le=100,
        description="对本次交易质量的自我评分 (0-100)：入场时机/持仓管理/盈亏/风控综合。90+优秀，70-90良好，50-70及格，<50差",
    )


class AdjustPosition(_DecisionBase):
    """加仓/减仓，调整现有持仓大小"""
    adjustment_pct: float = Field(
        description="调整百分比（正=加仓，负=减仓），如 50 表示加仓 50%，-30 表示减仓 30%；减仓幅度不得超过 99",
    )
    key_support: float = Field(description="关键支撑位价格")
    key_resistance: float = Field(description="关键阻力位价格")

    @field_validator("adjustment_pct")
    @classmethod
    def _check_pct(cls, v: float) -> float:
        if v == 0:
            raise ValueError("调整幅度不能为 0")
        if v < -99:
            raise ValueError("减仓幅度不得超过 99%（保留底仓）")
        return v


class SignalHold(_DecisionBase):
    """持仓不动（已有仓位时）"""
    rsi_value: float = Field(ge=0, le=100, description="当前 RSI 数值 (0-100)")


class SignalWait(_DecisionBase):
    """空仓观望（无仓位时）"""
    rsi_value: float = Field(ge=0, le=100, description="当前 RSI 数值 (0-100)")


DECISION_TOOLS: List[Type[BaseModel]] = [
    SignalEntryLong,
    SignalEntryShort,
    SignalExit,
    AdjustPosition,
    SignalHold,
    SignalWait,
]

TOOL_CLASS_BY_NAME: Dict[str, Type[BaseModel]] = {t.__name__: t for t in DECISION_TOOLS}

# 终止性函数：调用一个即完成本轮决策（对齐目标侧 FunctionExecutor.TERMINAL_FUNCTIONS）
TERMINAL_TOOLS = set(TOOL_CLASS_BY_NAME.keys())


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------

class ToolExecutor:
    """校验 LLM 工具调用参数并写入策略缓存（缓存契约同 ft-userdata-llm-share）"""

    def __init__(self, strategy_instance, max_leverage: float = 100):
        self.strategy = strategy_instance
        self.max_leverage = float(max_leverage)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行一个工具调用。

        Returns:
            {success: bool, message: str, action?: str, args?: dict}
            校验失败不抛异常，返回 success=False 由决策链降级处理。
        """
        tool_cls = TOOL_CLASS_BY_NAME.get(tool_name)
        if tool_cls is None:
            return {"success": False, "message": f"未知决策函数: {tool_name}"}

        try:
            model = tool_cls(**(arguments or {}))
        except ValidationError as e:
            msg = f"参数校验失败: {e.errors()[0].get('msg', str(e))}" if e.errors() else str(e)
            logger.warning(f"[{tool_name}] {msg}")
            return {"success": False, "message": msg, "action": None}

        args = model.model_dump()
        pair = args["pair"]

        # 杠杆二次截断（策略风控，比 pydantic 的 1-100 更紧）
        if "leverage" in args and args["leverage"] > self.max_leverage:
            logger.info(
                f"[{tool_name}] {pair} 杠杆 {args['leverage']} 超出策略上限 "
                f"{self.max_leverage}，已截断"
            )
            args["leverage"] = self.max_leverage

        handler = getattr(self, f"_exec_{tool_name}", None)
        if handler is None:
            return {"success": False, "message": f"执行器未实现: {tool_name}"}

        result = handler(pair, args)
        result.setdefault("action", tool_name)
        result["args"] = args
        return result

    # -- 各函数执行：写缓存（与目标侧 TradingTools 契约一致） -----------------

    def _exec_SignalEntryLong(self, pair: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.strategy._signal_cache[pair] = {"action": "enter_long", **args}
        self.strategy._leverage_cache[pair] = args["leverage"]
        if args.get("stake_amount"):
            self.strategy._stake_request_cache[pair] = args["stake_amount"]
        logger.info(
            f"[做多] {pair} | 置信度 {args['confidence_score']:.0f} | 杠杆 {args['leverage']}x"
            f" | 投入 {args.get('stake_amount') or '按预算'} | {args['reason']}"
        )
        return {"success": True, "message": f"做多信号已记录: {pair}"}

    def _exec_SignalEntryShort(self, pair: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.strategy._signal_cache[pair] = {"action": "enter_short", **args}
        self.strategy._leverage_cache[pair] = args["leverage"]
        if args.get("stake_amount"):
            self.strategy._stake_request_cache[pair] = args["stake_amount"]
        logger.info(
            f"[做空] {pair} | 置信度 {args['confidence_score']:.0f} | 杠杆 {args['leverage']}x"
            f" | 投入 {args.get('stake_amount') or '按预算'} | {args['reason']}"
        )
        return {"success": True, "message": f"做空信号已记录: {pair}"}

    def _exec_SignalExit(self, pair: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.strategy._signal_cache[pair] = {
            "action": "exit",
            "confidence_score": args["confidence_score"],
            "rsi_value": args["rsi_value"],
            "trade_score": args["trade_score"],
            "reason": args["reason"],
        }
        self.strategy._model_score_cache[pair] = args["trade_score"]
        logger.info(
            f"[平仓] {pair} | 置信度 {args['confidence_score']:.0f} | 自评 {args['trade_score']:.0f}/100 | {args['reason']}"
        )
        return {"success": True, "message": f"平仓信号已记录: {pair}"}

    def _exec_AdjustPosition(self, pair: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.strategy._position_adjustment_cache[pair] = {
            "adjustment_pct": args["adjustment_pct"],
            "confidence_score": args["confidence_score"],
            "key_support": args["key_support"],
            "key_resistance": args["key_resistance"],
            "reason": args["reason"],
        }
        direction = "加仓" if args["adjustment_pct"] > 0 else "减仓"
        logger.info(
            f"[{direction}] {pair} | 幅度 {args['adjustment_pct']:+.1f}% | 置信度 {args['confidence_score']:.0f} | {args['reason']}"
        )
        return {"success": True, "message": f"{direction}信号已记录: {pair}"}

    def _exec_SignalHold(self, pair: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.strategy._signal_cache[pair] = {
            "action": "hold",
            "confidence_score": args["confidence_score"],
            "rsi_value": args["rsi_value"],
            "reason": args["reason"],
        }
        logger.info(f"[持仓] {pair} | 置信度 {args['confidence_score']:.0f} | {args['reason']}")
        return {"success": True, "message": f"持仓不动: {pair}"}

    def _exec_SignalWait(self, pair: str, args: Dict[str, Any]) -> Dict[str, Any]:
        self.strategy._signal_cache[pair] = {
            "action": "wait",
            "confidence_score": args["confidence_score"],
            "rsi_value": args["rsi_value"],
            "reason": args["reason"],
        }
        logger.info(f"[观望] {pair} | 置信度 {args['confidence_score']:.0f} | {args['reason']}")
        return {"success": True, "message": f"空仓等待: {pair}"}
