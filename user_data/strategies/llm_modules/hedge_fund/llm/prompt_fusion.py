"""
融合系统提示词（PromptFusion）

结构（继承目标侧骨架 + 注入源侧纪律）：
  目标侧 PromptBuilder 开仓/持仓提示词（启发式经验，措辞保留）
  + 【量化信号面板】hf_* 解读说明（源侧五信号集成）
  + 【交易合法性红线】命令式硬规则（改编自源侧 PortfolioManagement 交易规则）
  + 【头寸预算】20% 单标的纪律
  + 【输出要求】六选一决策函数调用规范

分层措辞原则：红线段一律用命令式（必须/不得/仅当）；其余段落保留目标侧
启发式语气（可参考/结合判断）。两段之间显式声明优先级，避免语气冲突。
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# 与目标侧 context_builder 一致：把 llm_modules 目录加入 sys.path，使 context 成为顶级包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from context.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class PromptFusion:

    def __init__(self, risk_config: Dict[str, Any], context_config: Dict[str, Any]):
        self.max_leverage = risk_config.get("max_leverage", 20)
        self.max_position_pct = risk_config.get("max_position_pct", 20)
        self.max_open_trades = risk_config.get("max_open_trades", 3)
        include_tf_guidance = context_config.get("include_timeframe_guidance", False)
        self._base = PromptBuilder(include_timeframe_guidance=include_tf_guidance)

    # ------------------------------------------------------------------
    # 融合段落
    # ------------------------------------------------------------------

    def _hf_panel_section(self) -> str:
        return (
            "\n【量化信号面板 · hf 系列】\n"
            "用户消息中会提供一个【量化信号面板】，来自移植的多时间框架五信号加权集成\n"
            "（趋势跟踪/均值回归/动量/波动率/统计套利，加权融合为 hf_score）。\n"
            "解读：score > +0.2 偏多、< -0.2 偏空、其余中性；confidence 为综合置信度。\n"
            "注意：hf 系列是独立口径，与其他技术指标数值存在差异属正常现象。\n"
            "该面板是量化先验参考，不构成必须遵守的规则，请结合市场上下文自行判断。\n"
        )

    def _redline_section(self) -> str:
        return (
            "\n【交易合法性红线 · 无条件遵守】\n"
            "以下规则改编自投资组合管理的交易纪律，违反的请求会被代码直接拒绝或截断：\n"
            f"1. 开多/开空：仅当有可用保证金；指定投入金额不得超过用户消息中的「剩余可投入」额度\n"
            f"2. 平仓：仅当该交易对当前确有持仓；无持仓时的平仓请求一律无效\n"
            f"3. 减仓：adjustment_pct 不得低于 -99（保留底仓）；加仓后总资本占用仍受预算约束\n"
            f"4. 单标的资本占用（保证金合计）不得超过组合总值的 {self.max_position_pct:.0f}%\n"
            f"5. 杠杆必须在 1 ~ {self.max_leverage:.0f} 之间，超出会被截断\n"
            f"6. 同时持仓交易对数不得超过 {self.max_open_trades}\n"
            "7. 空仓时只能调用 signal_entry_long / signal_entry_short / signal_wait；\n"
            "   持仓时只能调用 signal_exit / adjust_position / signal_hold\n"
            "8. 每轮决策必须且只能调用一个决策函数；禁止输出裸 JSON 文本代替函数调用\n"
            "以上红线之外的所有内容（包括信号面板与历史经验）均为经验启发，允许结合上下文灵活判断。\n"
        )

    def _budget_section(self) -> str:
        return (
            "\n【头寸预算】\n"
            f"单标的资本占用上限为组合总值的 {self.max_position_pct:.0f}%（含已有仓位）。\n"
            "用户消息中会给出该交易对的「头寸预算」明细：组合总值、单标的上限、已占用、剩余可投入。\n"
            "指定 stake_amount 时不要超过剩余可投入额度；留空则由策略按预算自动分配。\n"
        )

    def _output_section(self) -> str:
        return (
            "\n【输出要求 · 决策函数】\n"
            "完成分析后，必须且只能调用以下六个决策函数之一（参数校验失败视为无效决策）：\n"
            "- signal_entry_long(pair, leverage, confidence_score, key_support, key_resistance, rsi_value, trend_strength, stake_amount?, reason)：做多开仓\n"
            "- signal_entry_short(同上)：做空开仓\n"
            "- signal_exit(pair, confidence_score, rsi_value, trade_score, reason)：平仓并自我评分\n"
            "- adjust_position(pair, adjustment_pct, confidence_score, key_support, key_resistance, reason)：加仓/减仓\n"
            "- signal_hold(pair, confidence_score, rsi_value, reason)：持仓不动\n"
            "- signal_wait(pair, confidence_score, rsi_value, reason)：空仓观望\n"
            "reason 必须引用至少一个具体数值（价格/指标/信号分数），禁止空泛表述。\n"
        )

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------

    def build_entry_system_prompt(self) -> str:
        """开仓场景（无持仓）系统提示词"""
        return (
            self._base.build_entry_prompt()
            + self._hf_panel_section()
            + self._redline_section()
            + self._budget_section()
            + self._output_section()
        )

    def build_position_system_prompt(self) -> str:
        """持仓管理场景系统提示词"""
        return (
            self._base.build_position_prompt()
            + self._hf_panel_section()
            + self._redline_section()
            + self._budget_section()
            + self._output_section()
        )
