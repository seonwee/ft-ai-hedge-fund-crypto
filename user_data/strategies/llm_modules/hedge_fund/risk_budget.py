"""
风险预算模块（移植自 ai-hedge-fund-crypto/src/graph/risk_management_node.py）

源侧纪律：单标的仓位 ≤ 组合总值 × 20%，且 ≤ 可用资金；
已有持仓占用预算，剩余额度 = 上限 − 该标的已用额度。

freqtrade 适配口径：
- 组合总值 = wallets 总权益（free + used，含持仓保证金）
- 单标的已用额度 = 该 pair 所有未平仓 trade 的 stake_amount（保证金）之和
  （源侧为现货成本基准；期货下以保证金占用表达资本分配，名义敞口由 leverage 决定，
   该约束限制的是"单标的资本占用"，与源侧语义一致）
- 可用资金 = wallets free 余额

分层原则（继承源侧）：风险预算是确定性代码硬约束，不是 LLM 的职责；
提示词中另有红线段做软约束，双保险。
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def compute_position_budget(
    total_equity: float,
    free_capital: float,
    open_trades: List[Dict[str, Any]],
    pair: str,
    max_position_pct: float = 20.0,
) -> Dict[str, float]:
    """
    计算某个 pair 的剩余仓位预算。

    Args:
        total_equity: 组合总值（free + used）
        free_capital: 可用资金（free）
        open_trades: 未平仓交易列表，每项至少含 {pair, stake_amount}
        pair: 目标交易对
        max_position_pct: 单标的仓位上限（占组合总值百分比）

    Returns:
        {total_equity, pair_used, position_limit, remaining_limit, free_capital, max_stake}
    """
    total_equity = float(total_equity or 0.0)
    free_capital = float(free_capital or 0.0)

    pair_used = sum(
        float(t.get("stake_amount", 0.0) or 0.0)
        for t in open_trades
        if t.get("pair") == pair
    )

    position_limit = total_equity * (float(max_position_pct) / 100.0)
    remaining_limit = position_limit - pair_used
    max_stake = max(0.0, min(remaining_limit, free_capital))

    return {
        "total_equity": total_equity,
        "pair_used": float(pair_used),
        "position_limit": float(position_limit),
        "remaining_limit": float(remaining_limit),
        "free_capital": free_capital,
        "max_stake": float(max_stake),
    }


def cap_stake(
    desired_stake: Optional[float],
    budget: Dict[str, float],
    min_stake: Optional[float] = None,
    pair: str = "",
) -> float:
    """
    用预算硬截断期望仓位。

    Args:
        desired_stake: LLM/默认逻辑期望的投入金额（None 表示用满剩余预算）
        budget: compute_position_budget 的返回值
        min_stake: 交易所最小下单金额
        pair: 用于日志

    Returns:
        截断后的合法 stake；无可用预算时返回 0（不开仓）
    """
    max_stake = budget.get("max_stake", 0.0)

    if max_stake <= 0:
        logger.info(
            f"{pair} 无剩余仓位预算: 上限 {budget.get('position_limit', 0):.2f}, "
            f"已用 {budget.get('pair_used', 0):.2f}, 可用 {budget.get('free_capital', 0):.2f}"
        )
        return 0.0

    stake = max_stake if desired_stake is None else float(desired_stake)

    if stake > max_stake:
        logger.info(
            f"{pair} 期望投入 {stake:.2f} 超出剩余预算 {max_stake:.2f}，已截断"
        )
        stake = max_stake

    if min_stake is not None and stake < float(min_stake):
        logger.info(f"{pair} 截断后金额 {stake:.2f} 低于最小下单 {min_stake:.2f}，放弃开仓")
        return 0.0

    return float(stake)


def budget_line_for_prompt(budget: Dict[str, float], pair: str) -> str:
    """预算信息格式化为提示词文本（用户消息中的头寸预算段）。"""
    return (
        f"【头寸预算 · {pair}】\n"
        f"  组合总值: {budget['total_equity']:.2f} USDT\n"
        f"  单标的预算上限: {budget['position_limit']:.2f} USDT\n"
        f"  该标的已占用: {budget['pair_used']:.2f} USDT\n"
        f"  剩余可投入: {budget['max_stake']:.2f} USDT（已受可用资金约束）"
    )
