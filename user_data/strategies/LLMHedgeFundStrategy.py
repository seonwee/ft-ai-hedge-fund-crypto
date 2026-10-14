"""
LLM Hedge Fund Strategy
基于 freqtrade + LangChain 1.x 的加密货币 AI 交易策略

移植自 ai-hedge-fund-crypto 的「五信号加权集成 + 风险预算 + LLM 下单」决策链，
并接入 ft-userdata-llm-share 的核心经验闭环。

关键设计：
- LLM 决策层用 LangChain 重写（bind_tools + pydantic 参数模型），支持一主一备模型
- 五信号层（hf_*）既是 LLM 上下文，也是 LLM 失效时的确定性兜底决策器
- 风险预算（单标的 ≤ 组合 20% 且 ≤ 可用资金）在 custom_stake_amount 硬截断
- 上下文不注入原始 K 线序列（raw_kline_history_points=0），只注入指标/账户/持仓/情绪/经验
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, merge_informative_pair

from llm_modules.indicators.indicator_calculator import IndicatorCalculator
from llm_modules.utils.config_loader import ConfigLoader
from llm_modules.utils.context_builder import ContextBuilder
from llm_modules.utils.position_tracker import PositionTracker
from llm_modules.utils.market_comparator import MarketStateComparator
from llm_modules.utils.decision_checker import DecisionQualityChecker

from llm_modules.experience.trade_logger import TradeLogger
from llm_modules.experience.experience_manager import ExperienceManager
from llm_modules.experience.trade_reviewer import TradeReviewer

from llm_modules.learning.historical_query import HistoricalQueryEngine
from llm_modules.learning.pattern_analyzer import PatternAnalyzer
from llm_modules.learning.self_reflection import SelfReflectionEngine
from llm_modules.learning.trade_evaluator import TradeEvaluator

from llm_modules.hedge_fund.signals import (
    add_hf_columns, compute_signal_panel, format_signal_panel_for_prompt,
)
from llm_modules.hedge_fund.risk_budget import compute_position_budget, cap_stake
from llm_modules.hedge_fund.llm.decision_tools import ToolExecutor
from llm_modules.hedge_fund.llm.decision_chain import DecisionChain
from llm_modules.hedge_fund.llm.prompt_fusion import PromptFusion

logger = logging.getLogger(__name__)


class LLMHedgeFundStrategy(IStrategy):
    """
    五信号 + LLM 决策的加密货币期货策略。

    主周期可配置（默认 1h），辅助周期 30m/4h/1d。
    """

    INTERFACE_VERSION = 3
    can_short = True
    timeframe = '1h'

    # 1h * 250 ≈ 10 天；确保 hf 动量（最长 126 周期）与 Hurst 窗口稳定
    startup_candle_count = 300

    # 极端止损兜底；实际平仓由 LLM 决策
    stoploss = -0.99
    use_custom_stoploss = False

    position_adjustment_enable = True
    max_entry_position_adjustment = 10

    order_types = {
        'entry': 'market',
        'exit': 'market',
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        logger.info("=" * 60)
        logger.info("LLM Hedge Fund Strategy - 正在初始化...")
        logger.info("=" * 60)

        # 主周期可配置（默认 1h）
        if config.get("timeframe"):
            self.timeframe = config["timeframe"]

        # 1. 配置
        self.config_loader = ConfigLoader()
        self.llm_config = self.config_loader.get_llm_config()
        self.risk_config = self.config_loader.get_risk_config()
        self.experience_config = self.config_loader.get_experience_config()
        self.context_config = self.config_loader.get_context_config()
        self.hf_config = self.config_loader.get_hf_config()

        # 2. 经验闭环（核心）
        trade_log_path = self.experience_config.get(
            'trade_log_path', './user_data/logs/trade_experience.jsonl')
        self.historical_query = HistoricalQueryEngine(trade_log_path)
        self.pattern_analyzer = PatternAnalyzer(min_sample_size=5)
        self.self_reflection = SelfReflectionEngine()
        self.trade_evaluator = TradeEvaluator()

        # 3. 上下文构建器（注入学习组件）
        self.context_builder = ContextBuilder(
            context_config=self.context_config,
            historical_query_engine=self.historical_query,
            pattern_analyzer=self.pattern_analyzer,
            tradable_balance_ratio=config.get('tradable_balance_ratio', 1.0),
            max_open_trades=config.get('max_open_trades', 3),
        )

        # 4. 决策层（LangChain）
        self.tool_executor = ToolExecutor(
            self, max_leverage=self.risk_config.get('max_leverage', 20))
        self.decision_chain = DecisionChain(self.llm_config, self.tool_executor)

        # 5. 融合提示词
        self.prompt_fusion = PromptFusion(self.risk_config, self.context_config)
        self.entry_system_prompt = self.prompt_fusion.build_entry_system_prompt()
        self.position_system_prompt = self.prompt_fusion.build_position_system_prompt()

        # 6. 经验写回
        self.trade_logger = TradeLogger(self.experience_config)
        self.experience_manager = ExperienceManager(
            trade_logger=self.trade_logger,
            self_reflection_engine=self.self_reflection,
            trade_evaluator=self.trade_evaluator,
        )

        # 7. 增强模块
        self.position_tracker = PositionTracker()
        self.market_comparator = MarketStateComparator()
        self.decision_checker = DecisionQualityChecker()
        self.trade_reviewer = TradeReviewer()

        # 8. 缓存（契约同 ft-userdata-llm-share）
        self._signal_cache: Dict[str, Dict[str, Any]] = {}
        self._leverage_cache: Dict[str, float] = {}
        self._stake_request_cache: Dict[str, float] = {}
        self._position_adjustment_cache: Dict[str, Dict[str, Any]] = {}
        self._model_score_cache: Dict[str, float] = {}
        self._pending_entry_states: Dict[str, Dict[str, Any]] = {}
        self._degrade_count = 0

        logger.info("✓ 策略初始化完成")
        logger.info(f"  - 主周期: {self.timeframe}")
        logger.info(f"  - LLM: {self.llm_config.get('model')}（备: "
                    f"{(self.llm_config.get('backup') or {}).get('model', '未配置')}）")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 技术指标 + 多周期
    # ------------------------------------------------------------------

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = IndicatorCalculator.add_all_indicators(dataframe)

        # 五信号层：写入 8 个 hf_* 数值列 + 生成信号面板
        dataframe, panel = add_hf_columns(
            dataframe,
            weights=self.hf_config.get('strategy_weights'),
            threshold=self.hf_config.get('signal_threshold', 0.2),
            hurst_window=self.hf_config.get('hurst_window', 126),
            min_candles=self.hf_config.get('min_candles', 200),
        )
        # 信号面板缓存到实例，供决策与降级使用（按 pair 存最新一版）
        self._last_panel = panel

        # 多周期：30m/4h/1d 的 TA-Lib 指标（hf 信号仅主周期计算，避免口径混乱）
        dataframe = merge_informative_pair(
            dataframe, self.dp.get_pair_dataframe(metadata['pair'], '30m'), self.timeframe,
            '30m', ffill=True)
        dataframe = merge_informative_pair(
            dataframe, self.dp.get_pair_dataframe(metadata['pair'], '4h'), self.timeframe,
            '4h', ffill=True)
        dataframe = merge_informative_pair(
            dataframe, self.dp.get_pair_dataframe(metadata['pair'], '1d'), self.timeframe,
            '1d', ffill=True)

        for tf in ('30m', '4h', '1d'):
            IndicatorCalculator.add_all_indicators(dataframe)

        return dataframe

    # ------------------------------------------------------------------
    # 开仓决策（LLM + hf 兜底）
    # ------------------------------------------------------------------

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']

        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''

        if len(dataframe) < self.startup_candle_count:
            return dataframe

        try:
            current_trades = Trade.get_open_trades()
            exchange = self._get_exchange()

            market_context = self.context_builder.build_market_context(
                dataframe=dataframe, metadata=metadata, wallets=self.wallets,
                current_trades=current_trades, exchange=exchange,
                position_tracker=self.position_tracker,
                market_comparator=self.market_comparator,
            )

            panel = self._last_panel
            hf_score = panel['score'] if panel else 0.0
            hf_confidence = panel['confidence'] if panel else 0.0
            signal_block = format_signal_panel_for_prompt(panel)

            budget = compute_position_budget(
                total_equity=self.wallets.get_total('USDT'),
                free_capital=self.wallets.get_free('USDT'),
                open_trades=self._trades_as_dicts(current_trades),
                pair=pair,
                max_position_pct=self.risk_config.get('max_position_pct', 20),
            )

            decision_request = self._build_decision_request(
                "entry", market_context, signal_block, budget, pair)

            outcome = self.decision_chain.decide(
                self.entry_system_prompt, decision_request)

            if outcome.degraded:
                self._fallback_with_hf(
                    dataframe, pair, "entry", outcome, hf_score, hf_confidence)
            else:
                self._apply_entry_outcome(dataframe, pair, outcome)

            # 经验记录
            self._log_decision(pair, "entry", outcome, panel)

        except Exception as e:
            logger.error(f"开仓决策失败 {pair}: {e}", exc_info=True)
            # 异常兜底：用五信号层
            panel = getattr(self, '_last_panel', None)
            self._fallback_with_hf(
                dataframe, pair, "entry",
                None, panel['score'] if panel else 0.0,
                panel['confidence'] if panel else 0.0)

        return dataframe

    # ------------------------------------------------------------------
    # 平仓决策（有持仓才走）
    # ------------------------------------------------------------------

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']

        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        dataframe.loc[:, 'exit_tag'] = ''

        if len(dataframe) < self.startup_candle_count:
            return dataframe

        try:
            current_trades = Trade.get_open_trades()
            pair_trades = [t for t in current_trades if t.pair == pair]
            if not pair_trades:
                return dataframe

            exchange = self._get_exchange()

            # 更新持仓追踪与开仓状态关联
            self._update_position_tracking(dataframe, pair, pair_trades)

            market_context = self.context_builder.build_market_context(
                dataframe=dataframe, metadata=metadata, wallets=self.wallets,
                current_trades=current_trades, exchange=exchange,
                position_tracker=self.position_tracker,
                market_comparator=self.market_comparator,
            )

            panel = self._last_panel
            signal_block = format_signal_panel_for_prompt(panel)

            decision_request = self._build_decision_request(
                "exit", market_context, signal_block, None, pair)

            outcome = self.decision_chain.decide(
                self.position_system_prompt, decision_request)

            if outcome.degraded:
                self._fallback_exit_rule(dataframe, pair, outcome, pair_trades)
            elif outcome.action == "exit":
                dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
                dataframe.loc[dataframe.index[-1], 'exit_short'] = 1
                dataframe.loc[dataframe.index[-1], 'exit_tag'] = outcome.reason
                logger.info(f"🔚 {pair} 平仓 | 自评 {outcome.trade_score or '—'}/100 | {outcome.reason}")
            elif outcome.action == "hold":
                logger.info(f"💎 {pair} 继续持有 | {outcome.reason}")
            # adjust 由 adjust_trade_position 回调消费

            self._log_decision(pair, "exit", outcome, panel)
            self._check_decision_quality(pair, outcome, pair_trades, dataframe)

        except Exception as e:
            logger.error(f"平仓决策失败 {pair}: {e}", exc_info=True)

        return dataframe

    # ------------------------------------------------------------------
    # 决策落地辅助
    # ------------------------------------------------------------------

    def _apply_entry_outcome(self, dataframe, pair, outcome):
        if outcome.action == "enter_long":
            dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
            dataframe.loc[dataframe.index[-1], 'enter_tag'] = outcome.reason
            logger.info(f"📈 {pair} 做多 | 置信度 {outcome.confidence:.0f} | {outcome.reason}")
        elif outcome.action == "enter_short":
            dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
            dataframe.loc[dataframe.index[-1], 'enter_tag'] = outcome.reason
            logger.info(f"📉 {pair} 做空 | 置信度 {outcome.confidence:.0f} | {outcome.reason}")
        elif outcome.action == "wait":
            logger.info(f"⏸️  {pair} 观望 | {outcome.reason}")
        else:
            logger.info(f"⏸️  {pair} 未明确信号")

    def _fallback_with_hf(self, dataframe, pair, phase, outcome, hf_score, hf_confidence):
        """LLM 不可用/无效时的五信号兜底（仅开仓方向；平仓走独立规则）"""
        reason = "LLM 降级·五信号融合"
        if outcome is not None:
            reason += f"（{outcome.degrade_reason}）"
        self._degrade_count += 1

        if phase == "entry":
            if hf_score > self.hf_config.get('signal_threshold', 0.2):
                dataframe.loc[dataframe.index[-1], 'enter_long'] = 1
                dataframe.loc[dataframe.index[-1], 'enter_tag'] = f"{reason} score={hf_score:+.3f}"
                logger.info(f"⚠️ {pair} [降级] 五信号偏多 score={hf_score:+.3f}")
            elif hf_score < -self.hf_config.get('signal_threshold', 0.2):
                dataframe.loc[dataframe.index[-1], 'enter_short'] = 1
                dataframe.loc[dataframe.index[-1], 'enter_tag'] = f"{reason} score={hf_score:+.3f}"
                logger.info(f"⚠️ {pair} [降级] 五信号偏空 score={hf_score:+.3f}")
            else:
                logger.info(f"⏸️  {pair} [降级] 五信号中性，观望")

    def _fallback_exit_rule(self, dataframe, pair, outcome, pair_trades):
        """LLM 失效时的平仓兜底：风控止损 + 利润保护"""
        if not pair_trades:
            return
        trade = pair_trades[0]
        metrics = self.position_tracker.get_position_metrics(trade.id)
        max_loss = metrics.get('max_loss_pct', 0) if metrics else 0
        max_profit = metrics.get('max_profit_pct', 0) if metrics else 0
        drawdown = metrics.get('drawdown_from_peak_pct', 0) if metrics else 0

        reason = None
        if max_loss <= self.risk_config.get('emergency_stop_loss', -0.15) * 100:
            reason = "LLM 降级·风控止损"
        elif max_profit > 3 and drawdown < -3:
            reason = "LLM 降级·利润保护"

        if reason:
            dataframe.loc[dataframe.index[-1], 'exit_long'] = 1
            dataframe.loc[dataframe.index[-1], 'exit_short'] = 1
            dataframe.loc[dataframe.index[-1], 'exit_tag'] = reason
            logger.info(f"⚠️ {pair} [降级平仓] {reason} MFE {max_profit:+.2f}% MAE {max_loss:+.2f}%")

    # ------------------------------------------------------------------
    # 上下文与经验辅助
    # ------------------------------------------------------------------

    def _build_decision_request(self, action_type, market_context, signal_block, budget, pair):
        parts = [
            f"请分析当前情况，决策: {'是否开仓(做多或做空)' if action_type == 'entry' else '是否平仓/调整仓位'}",
            "",
            "=" * 50,
            market_context,
            "",
            signal_block,
        ]
        if budget is not None:
            parts.append("")
            parts.append(
                f"【头寸预算 · {pair}】组合总值 {budget['total_equity']:.2f} USDT | "
                f"单标的上限 {budget['position_limit']:.2f} | 已占用 {budget['pair_used']:.2f} | "
                f"剩余可投入 {budget['max_stake']:.2f}")
        parts.append("")
        parts.append("决策后必须且只能调用一个决策函数，立即停止。")
        return "\n".join(parts)

    def _log_decision(self, pair, action_type, outcome, panel):
        try:
            self.experience_manager.log_decision_with_context(
                pair=pair, action=action_type,
                decision=outcome.raw_content or outcome.reason,
                reasoning=outcome.reason,
                confidence=(outcome.confidence / 100.0),
                market_context={"hf_panel": panel} if panel else {},
                function_calls=[],
            )
        except Exception as e:
            logger.debug(f"记录决策失败: {e}")

    def _check_decision_quality(self, pair, outcome, pair_trades, dataframe):
        if not pair_trades:
            return
        try:
            trade = pair_trades[0]
            price = dataframe.iloc[-1]['close']
            if trade.is_short:
                profit_pct = (trade.open_rate - price) / trade.open_rate * trade.leverage * 100
            else:
                profit_pct = (price - trade.open_rate) / trade.open_rate * trade.leverage * 100
            dtype = 'exit' if outcome.action == 'exit' else 'hold'
            check = self.decision_checker.record_decision(
                pair=pair, decision_type=dtype, reason=outcome.reason, profit_pct=profit_pct)
            for w in (check.get('warnings') or []):
                if w.get('level') == 'high':
                    logger.warning(f"[决策质量] {w.get('message')}")
        except Exception as e:
            logger.debug(f"决策质量检查失败: {e}")

    def _update_position_tracking(self, dataframe, pair, pair_trades):
        if dataframe.empty:
            return
        price = dataframe.iloc[-1]['close']
        for trade in pair_trades:
            try:
                self.position_tracker.update_position(
                    trade_id=trade.id, pair=pair, current_price=price,
                    open_price=trade.open_rate, is_short=trade.is_short,
                    leverage=trade.leverage, decision_type='check', decision_reason='')
                temp_key = f"{pair}_{trade.open_rate}"
                pending = getattr(self, '_pending_entry_states', {})
                if temp_key in pending:
                    st = pending.pop(temp_key)
                    self.market_comparator.save_entry_state(
                        trade_id=trade.id, pair=pair, price=trade.open_rate,
                        indicators=st.get('indicators', {}),
                        entry_reason=st.get('entry_tag', ''),
                        trend_alignment='', market_sentiment='')
            except Exception as e:
                logger.debug(f"更新持仓追踪失败: {e}")

    def _get_exchange(self):
        if hasattr(self, 'dp') and self.dp:
            return getattr(self.dp, '_exchange', None) or getattr(self.dp, 'exchange', None)
        return None

    @staticmethod
    def _trades_as_dicts(trades):
        return [{"pair": t.pair, "stake_amount": t.stake_amount} for t in trades]

    # ------------------------------------------------------------------
    # freqtrade 回调
    # ------------------------------------------------------------------

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        stake_request = self._stake_request_cache.pop(pair, None)

        budget = compute_position_budget(
            total_equity=self.wallets.get_total('USDT'),
            free_capital=self.wallets.get_free('USDT'),
            open_trades=self._trades_as_dicts(Trade.get_open_trades()),
            pair=pair,
            max_position_pct=self.risk_config.get('max_position_pct', 20),
        )

        desired = stake_request if stake_request else (proposed_stake or max_stake)
        stake = cap_stake(desired, budget, min_stake=min_stake, pair=pair)

        if stake <= 0:
            return 0.0
        return min(stake, max_stake)

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        if pair in self._leverage_cache:
            val = self._leverage_cache.pop(pair)
            return min(val, max_leverage)
        return min(self.risk_config.get('default_leverage', 3), max_leverage)

    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs):
        pair = trade.pair
        info = self._position_adjustment_cache.pop(pair, None)
        if not info:
            return None

        pct = info.get('adjustment_pct', 0)
        adjustment_stake = trade.stake_amount * (pct / 100.0)

        if pct > 0:
            budget = compute_position_budget(
                total_equity=self.wallets.get_total('USDT'),
                free_capital=self.wallets.get_free('USDT'),
                open_trades=self._trades_as_dicts(Trade.get_open_trades()),
                pair=pair,
                max_position_pct=self.risk_config.get('max_position_pct', 20),
            )
            adjustment_stake = min(adjustment_stake, budget['max_stake'], max_stake)
            if min_stake and adjustment_stake < min_stake:
                logger.warning(f"{pair} 加仓金额 {adjustment_stake:.2f} 低于最小下单")
                return None
            logger.info(f"{pair} 加仓 {pct:.1f}% = {adjustment_stake:.2f} USDT | {info.get('reason', '')}")
            return adjustment_stake
        elif pct < 0:
            max_reduce = -trade.stake_amount * 0.99
            adjustment_stake = max(adjustment_stake, max_reduce)
            logger.info(f"{pair} 减仓 {abs(pct):.1f}% = {adjustment_stake:.2f} USDT | {info.get('reason', '')}")
            return adjustment_stake
        return None

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs):
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty:
                return True
            latest = dataframe.iloc[-1]
            indicators = {
                'atr': latest.get('atr', 0), 'rsi': latest.get('rsi', 50),
                'ema_20': latest.get('ema_20', 0), 'ema_50': latest.get('ema_50', 0),
                'macd': latest.get('macd', 0), 'adx': latest.get('adx', 0),
            }
            self._pending_entry_states[f"{pair}_{rate}"] = {
                'pair': pair, 'rate': rate, 'indicators': indicators,
                'entry_tag': entry_tag or '', 'side': side, 'time': current_time,
            }
        except Exception as e:
            logger.error(f"confirm_trade_entry 失败: {e}")
        return True

    def confirm_trade_exit(self, pair, trade, order_type, amount, rate, time_in_force,
                           exit_reason, current_time, **kwargs):
        try:
            position_metrics = self.position_tracker.get_position_metrics(trade.id)

            if trade.is_short:
                profit_pct = (trade.open_rate - rate) / trade.open_rate * trade.leverage * 100
            else:
                profit_pct = (rate - trade.open_rate) / trade.open_rate * trade.leverage * 100

            duration_minutes = int((current_time.replace(tzinfo=None) - trade.open_date.replace(tzinfo=None)).total_seconds() / 60)

            if self.trade_reviewer:
                review = self.trade_reviewer.generate_trade_review(
                    pair=pair, side='short' if trade.is_short else 'long',
                    entry_price=trade.open_rate, exit_price=rate,
                    entry_reason=getattr(trade, 'enter_tag', '') or '',
                    exit_reason=exit_reason, profit_pct=profit_pct,
                    duration_minutes=duration_minutes, leverage=trade.leverage,
                    position_metrics=position_metrics or {}, market_changes={})
                report = self.trade_reviewer.format_review_report(review)
                logger.info(f"\n{report}")

            if self.experience_manager:
                model_score = self._model_score_cache.pop(pair, None)
                market_condition = (
                    f"MFE {position_metrics.get('max_profit_pct', 0):+.2f}% / "
                    f"MAE {position_metrics.get('max_loss_pct', 0):+.2f}% / "
                    f"持仓 {duration_minutes}分钟"
                    + (f" / 模型评分 {model_score:.0f}/100" if model_score else "")
                ) if position_metrics else ""

                self.experience_manager.log_trade_completion(
                    trade_id=trade.id, pair=pair,
                    side='short' if trade.is_short else 'long',
                    entry_time=trade.open_date, entry_price=trade.open_rate,
                    entry_reason=getattr(trade, 'enter_tag', '') or '未记录',
                    exit_time=current_time.replace(tzinfo=None), exit_price=rate,
                    exit_reason=exit_reason, profit_pct=profit_pct,
                    profit_abs=trade.stake_amount * profit_pct / 100,
                    leverage=trade.leverage, stake_amount=trade.stake_amount,
                    max_drawdown=position_metrics.get('max_loss_pct', 0) if position_metrics else 0,
                    market_condition=market_condition,
                    position_metrics=position_metrics or {}, market_changes={})
                logger.info(f"✓ 交易 {trade.id} 已记录")

            self.position_tracker.remove_position(trade.id)
            self.market_comparator.remove_entry_state(trade.id)
        except Exception as e:
            logger.error(f"生成交易复盘失败: {e}", exc_info=True)
        return True
