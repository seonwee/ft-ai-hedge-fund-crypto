"""
市场状态对比模块 - 对比开仓时vs当前的市场变化
提供环境变化信息，让LLM判断开仓逻辑是否仍然成立
"""
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MarketStateComparator:
    """市场状态对比器 - 纯信息对比，无行为限制"""

    def __init__(self):
        # 存储开仓时的市场状态: {trade_id: market_state}
        self.entry_states: Dict[int, Dict[str, Any]] = {}

    def save_entry_state(
        self,
        trade_id: int,
        pair: str,
        price: float,
        indicators: Dict[str, float],
        entry_reason: str,
        trend_alignment: str = "",
        market_sentiment: str = ""
    ):
        """
        保存开仓时的完整市场状态

        Args:
            trade_id: 交易ID
            pair: 交易对
            price: 开仓价格
            indicators: 技术指标字典
            entry_reason: 开仓理由（完整）
            trend_alignment: 多时间框架趋势对齐情况
            market_sentiment: 市场情绪信息
        """
        self.entry_states[trade_id] = {
            'time': datetime.now(),
            'pair': pair,
            'price': price,
            'indicators': indicators.copy(),
            'entry_reason': entry_reason,
            'trend_alignment': trend_alignment,
            'market_sentiment': market_sentiment
        }
        logger.info(f"保存开仓状态 #{trade_id}: {pair} @ {price}")

    def compare_with_entry(
        self,
        trade_id: int,
        current_price: float,
        current_indicators: Dict[str, float],
        current_trend: str = "",
        current_sentiment: str = ""
    ) -> Dict[str, Any]:
        """
        对比当前状态与开仓时的变化

        返回详细的对比分析，供LLM参考
        不做任何决策，只提供信息
        """

        if trade_id not in self.entry_states:
            return {'error': '未找到开仓状态记录'}

        entry = self.entry_states[trade_id]

        # 价格变化
        price_change_pct = (current_price - entry['price']) / entry['price'] * 100

        # 指标变化
        indicator_changes = {}
        for key, entry_val in entry['indicators'].items():
            if key in current_indicators:
                current_val = current_indicators[key]
                if entry_val != 0:
                    change_pct = (current_val - entry_val) / abs(entry_val) * 100
                    indicator_changes[key] = {
                        'entry': entry_val,
                        'current': current_val,
                        'change': current_val - entry_val,
                        'change_pct': change_pct,
                        'direction': 'increased' if change_pct > 0 else ('decreased' if change_pct < 0 else 'unchanged')
                    }

        # 识别重大变化
        significant_changes = []

        # ATR变化 (波动性) - 重要指标
        if 'atr' in indicator_changes:
            atr_change = indicator_changes['atr']['change_pct']
            if abs(atr_change) > 50:
                direction = '激增' if atr_change > 0 else '骤降'
                significant_changes.append(
                    f"市场波动性{direction}{abs(atr_change):.1f}% (ATR: {entry['indicators']['atr']:.2f} → {current_indicators.get('atr', 0):.2f})"
                )

        # 趋势变化 (EMA)
        if 'ema_20' in indicator_changes and 'ema_50' in indicator_changes:
            entry_ema_diff = entry['indicators']['ema_20'] - entry['indicators']['ema_50']
            current_ema_diff = current_indicators.get('ema_20', 0) - current_indicators.get('ema_50', 0)

            # 趋势反转检测
            if (entry_ema_diff > 0 and current_ema_diff < 0) or (entry_ema_diff < 0 and current_ema_diff > 0):
                significant_changes.append(
                    "趋势发生反转 (EMA20与EMA50交叉，从{}变为{})".format(
                        '金叉' if entry_ema_diff > 0 else '死叉',
                        '金叉' if current_ema_diff > 0 else '死叉'
                    )
                )

        # RSI极端变化
        if 'rsi' in indicator_changes:
            entry_rsi = entry['indicators']['rsi']
            current_rsi = current_indicators.get('rsi', 50)

            # RSI从一个极端区域移动到另一个
            if (entry_rsi < 40 and current_rsi > 60) or (entry_rsi > 60 and current_rsi < 40):
                significant_changes.append(
                    f"RSI发生极端转变 ({entry_rsi:.1f} → {current_rsi:.1f})"
                )

        # MACD变化
        if 'macd' in indicator_changes and 'macd_signal' in indicator_changes:
            entry_macd_diff = entry['indicators']['macd'] - entry['indicators']['macd_signal']
            current_macd_diff = current_indicators.get('macd', 0) - current_indicators.get('macd_signal', 0)

            if (entry_macd_diff > 0 and current_macd_diff < 0) or (entry_macd_diff < 0 and current_macd_diff > 0):
                significant_changes.append(
                    "MACD信号交叉 (动能方向发生改变)"
                )

        # ADX变化 (趋势强度)
        if 'adx' in indicator_changes:
            entry_adx = entry['indicators']['adx']
            current_adx = current_indicators.get('adx', 0)
            adx_change_pct = indicator_changes['adx']['change_pct']

            if entry_adx >= 20 and current_adx < 18 and adx_change_pct <= -25:
                significant_changes.append(
                    f"趋势强度显著减弱 (ADX: {entry_adx:.1f} → {current_adx:.1f}, {adx_change_pct:+.1f}%)"
                )
            elif entry_adx <= 20 and current_adx > 25 and adx_change_pct >= 25:
                significant_changes.append(
                    f"趋势强度显著增强 (ADX: {entry_adx:.1f} → {current_adx:.1f}, {adx_change_pct:+.1f}%)"
                )

        # 持仓时长
        time_elapsed_hours = (datetime.now() - entry['time']).total_seconds() / 3600

        # 趋势对齐变化
        trend_alignment_changed = False
        if entry.get('trend_alignment') and current_trend:
            if entry['trend_alignment'] != current_trend:
                trend_alignment_changed = True
                significant_changes.append(
                    f"多时间框架趋势对齐变化: {entry['trend_alignment']} → {current_trend}"
                )

        return {
            'time_elapsed_hours': time_elapsed_hours,
            'time_elapsed_str': f"{time_elapsed_hours:.1f}小时" if time_elapsed_hours < 24 else f"{time_elapsed_hours/24:.1f}天",
            'price_change_pct': price_change_pct,
            'indicator_changes': indicator_changes,
            'significant_changes': significant_changes,
            'has_significant_changes': len(significant_changes) > 0,
            'entry_time': entry['time'].strftime("%Y-%m-%d %H:%M:%S"),
            'entry_price': entry['price'],
            'entry_reason': entry['entry_reason'],
            'entry_trend_alignment': entry.get('trend_alignment', '未知'),
            'current_trend_alignment': current_trend,
            'trend_alignment_changed': trend_alignment_changed,
            'entry_sentiment': entry.get('market_sentiment', ''),
            'current_sentiment': current_sentiment
        }

    def generate_comparison_text(self, trade_id: int, current_price: float, current_indicators: Dict[str, float], current_trend: str = "", current_sentiment: str = "") -> str:
        """
        生成对比分析文本

        用于添加到LLM的context中
        """
        comparison = self.compare_with_entry(trade_id, current_price, current_indicators, current_trend, current_sentiment)

        if 'error' in comparison:
            return f"市场状态对比: {comparison['error']}"

        text_parts = [
            "【开仓 vs 当前原始数据】",
            f"trade_id: {trade_id}",
            f"entry_time: {comparison['entry_time']}",
            f"current_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"holding_hours: {comparison['time_elapsed_hours']:.2f}",
            f"entry_price: {comparison['entry_price']:.6f}",
            f"current_price: {current_price:.6f}",
            f"price_change_pct: {comparison['price_change_pct']:+.2f}%",
            ""
        ]

        # 关键指标对比（原始数据）
        text_parts.append("关键指标对比 (entry → current | Δ%):")
        key_indicators = ['atr', 'rsi', 'macd', 'adx', 'ema_20', 'ema_50']
        for key in key_indicators:
            if key in comparison['indicator_changes']:
                change = comparison['indicator_changes'][key]
                text_parts.append(
                    f"  {key.upper()}: {change['entry']:.4f} → {change['current']:.4f} ({change['change_pct']:+.2f}%)"
                )

        text_parts.append("")

        text_parts.append("趋势对齐记录:")
        text_parts.append(f"  entry_alignment: {comparison.get('entry_trend_alignment', '未知')}")
        text_parts.append(f"  current_alignment: {comparison.get('current_trend_alignment', '未知')}")
        text_parts.append(f"  alignment_changed: {comparison['trend_alignment_changed']}")

        text_parts.append("")

        # 原始开仓理由
        text_parts.append("开仓理由 (原文):")
        reason_lines = comparison['entry_reason'].split('\n')
        for line in reason_lines:
            if line.strip():
                text_parts.append(f"  {line.strip()}")

        return "\n".join(text_parts)

    def remove_entry_state(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """
        移除开仓状态记录

        返回最终的开仓状态数据用于复盘
        """
        if trade_id in self.entry_states:
            final_state = self.entry_states[trade_id].copy()
            del self.entry_states[trade_id]
            logger.info(f"移除开仓状态 #{trade_id}")
            return final_state
        return None

    def get_entry_state(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """获取开仓状态"""
        return self.entry_states.get(trade_id)

    def get_all_entry_states(self) -> Dict[int, Dict[str, Any]]:
        """获取所有开仓状态"""
        return self.entry_states.copy()
