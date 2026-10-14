"""
市场分析器模块
负责分析市场结构、指标趋势、成交量等
"""
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """市场技术分析器"""

    def analyze_market_structure(
        self,
        dataframe: pd.DataFrame,
        lookback: int = 80,
        compare_window: int = 40,
        swing_threshold_pct: float = 1.5
    ) -> Dict[str, Any]:
        """
        分析市场结构（支撑/阻力/趋势）

        Args:
            dataframe: OHLCV数据
            lookback: 近期窗口，越小越关注最新走势
            compare_window: 对比窗口，用于和更早的数据比较
            swing_threshold_pct: 认定新高/新低的百分比阈值

        Returns:
            市场结构分析结果
        """
        if len(dataframe) < max(lookback, compare_window) + 5:
            return {"structure": "数据不足"}

        lookback = min(lookback, len(dataframe))
        compare_window = min(compare_window, max(0, len(dataframe) - lookback))
        if compare_window == 0:
            compare_window = min(max(10, lookback // 2), lookback)

        extended = dataframe.tail(lookback + compare_window)
        recent = extended.tail(lookback)
        baseline = extended.head(compare_window)

        highs = recent['high']
        lows = recent['low']
        closes = recent['close']

        swing_high = highs.max()
        swing_low = lows.min()
        current_price = closes.iloc[-1]

        baseline_high = baseline['high'].max() if not baseline.empty else highs.head(len(highs)//2).max()
        baseline_low = baseline['low'].min() if not baseline.empty else lows.head(len(lows)//2).min()

        swing_threshold_pct = max(0.5, swing_threshold_pct)
        high_break_pct = 0.0
        low_break_pct = 0.0

        if baseline_high and baseline_high > 0:
            high_break_pct = (swing_high - baseline_high) / baseline_high * 100
        if baseline_low and baseline_low > 0:
            low_break_pct = (baseline_low - swing_low) / baseline_low * 100

        recent_trend_pct = (current_price - closes.iloc[0]) / closes.iloc[0] * 100

        if high_break_pct > swing_threshold_pct and low_break_pct > swing_threshold_pct:
            structure = "扩张结构（高低点同时刷新）"
        elif high_break_pct > swing_threshold_pct:
            structure = "上升结构（Higher Highs）"
        elif low_break_pct > swing_threshold_pct:
            structure = "下降结构（Lower Lows）"
        else:
            slope_threshold = swing_threshold_pct / 2
            if abs(recent_trend_pct) <= slope_threshold:
                structure = "盘整结构（震荡）"
            elif recent_trend_pct > 0:
                structure = "上升通道（温和趋势）"
            else:
                structure = "下降通道（温和趋势）"

        distance_to_high = ((swing_high - current_price) / current_price) * 100
        distance_to_low = ((current_price - swing_low) / current_price) * 100

        return {
            "structure": structure,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "distance_to_high_pct": distance_to_high,
            "distance_to_low_pct": distance_to_low,
            "range_pct": ((swing_high - swing_low) / swing_low) * 100 if swing_low else 0,
            "high_break_pct": high_break_pct,
            "low_break_pct": low_break_pct,
            "recent_trend_pct": recent_trend_pct
        }

    def analyze_indicator_trends(self, dataframe: pd.DataFrame) -> Dict[str, str]:
        """
        分析指标变化趋势

        Args:
            dataframe: 包含技术指标的数据

        Returns:
            各指标趋势描述
        """
        if len(dataframe) < 5:
            return {}

        latest = dataframe.iloc[-1]
        prev = dataframe.iloc[-2]
        prev_5 = dataframe.iloc[-5]

        trends = {}

        # EMA交叉状态
        if 'ema_20' in latest and 'ema_50' in latest:
            ema20_now = latest['ema_20']
            ema50_now = latest['ema_50']
            ema20_prev = prev['ema_20']
            ema50_prev = prev['ema_50']

            if ema20_now > ema50_now and ema20_prev <= ema50_prev:
                trends['ema_cross'] = "刚刚金叉"
            elif ema20_now < ema50_now and ema20_prev >= ema50_prev:
                trends['ema_cross'] = "刚刚死叉"
            elif ema20_now > ema50_now:
                trends['ema_cross'] = "金叉持续中"
            else:
                trends['ema_cross'] = "死叉持续中"

        # MACD柱状图趋势
        if 'macd_hist' in latest:
            macd_hist_now = latest['macd_hist']
            macd_hist_prev = prev['macd_hist']

            if macd_hist_now > macd_hist_prev:
                trends['macd_histogram'] = "增强（多头）" if macd_hist_now > 0 else "减弱（空头弱化）"
            else:
                trends['macd_histogram'] = "减弱（多头弱化）" if macd_hist_now > 0 else "增强（空头）"

        # RSI趋势
        if 'rsi' in latest:
            rsi_now = latest['rsi']
            rsi_5ago = prev_5['rsi']

            if rsi_now > rsi_5ago + 5:
                trends['rsi_trend'] = "上升（动能增强）"
            elif rsi_now < rsi_5ago - 5:
                trends['rsi_trend'] = "下降（动能减弱）"
            else:
                trends['rsi_trend'] = "平稳"

        # ADX趋势强度
        if 'adx' in latest:
            adx_now = latest['adx']
            adx_prev = prev['adx']

            if adx_now > adx_prev:
                trends['adx_direction'] = f"上升（趋势增强，当前{adx_now:.1f}）"
            else:
                trends['adx_direction'] = f"下降（趋势减弱，当前{adx_now:.1f}）"

        return trends

    def analyze_timeframe_alignment(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        分析多时间框架趋势对齐

        Args:
            dataframe: 包含多时间框架指标的数据

        Returns:
            时间框架对齐分析结果
        """
        if len(dataframe) < 2:
            return {"alignment": "数据不足", "trends": {}}

        latest = dataframe.iloc[-1]
        trends = {}

        # 横盘判断阈值：EMA差距小于0.3%视为横盘
        consolidation_threshold = 0.003

        # 30分钟趋势
        if 'ema_20' in latest and 'ema_50' in latest:
            ema_diff = (latest['ema_20'] - latest['ema_50']) / latest['ema_50']
            if abs(ema_diff) < consolidation_threshold:
                trends['30m'] = "横盘"
            elif ema_diff > 0:
                trends['30m'] = "上涨"
            else:
                trends['30m'] = "下跌"

        # 1小时趋势
        if 'ema_20_1h' in latest and 'ema_50_1h' in latest:
            ema_diff = (latest['ema_20_1h'] - latest['ema_50_1h']) / latest['ema_50_1h']
            if abs(ema_diff) < consolidation_threshold:
                trends['1h'] = "横盘"
            elif ema_diff > 0:
                trends['1h'] = "上涨"
            else:
                trends['1h'] = "下跌"

        # 4小时趋势
        if 'ema_20_4h' in latest and 'ema_50_4h' in latest:
            ema_diff = (latest['ema_20_4h'] - latest['ema_50_4h']) / latest['ema_50_4h']
            if abs(ema_diff) < consolidation_threshold:
                trends['4h'] = "横盘"
            elif ema_diff > 0:
                trends['4h'] = "上涨"
            else:
                trends['4h'] = "下跌"

        # 1天趋势
        if 'ema_20_1d' in latest and 'ema_50_1d' in latest:
            ema_diff = (latest['ema_20_1d'] - latest['ema_50_1d']) / latest['ema_50_1d']
            if abs(ema_diff) < consolidation_threshold:
                trends['1d'] = "横盘"
            elif ema_diff > 0:
                trends['1d'] = "上涨"
            else:
                trends['1d'] = "下跌"

        # 判断对齐情况
        if not trends:
            return {"alignment": "无趋势数据", "trends": {}}

        uptrend_count = sum(1 for t in trends.values() if t == "上涨")
        downtrend_count = sum(1 for t in trends.values() if t == "下跌")
        consolidation_count = sum(1 for t in trends.values() if t == "横盘")
        total_count = len(trends)

        if uptrend_count == total_count:
            alignment = "完全对齐 - 强势上涨"
        elif downtrend_count == total_count:
            alignment = "完全对齐 - 强势下跌"
        elif consolidation_count == total_count:
            alignment = "完全对齐 - 横盘整理"
        elif consolidation_count >= total_count / 2:
            alignment = f"横盘为主（{consolidation_count}/{total_count}）- 等待突破"
        elif uptrend_count > downtrend_count:
            alignment = f"多数上涨（{uptrend_count}/{total_count}）"
        elif downtrend_count > uptrend_count:
            alignment = f"多数下跌（{downtrend_count}/{total_count}）"
        else:
            alignment = "趋势分歧（震荡）"

        return {
            "trends": trends,
            "alignment": alignment,
            "strength": uptrend_count / total_count if uptrend_count > downtrend_count else -downtrend_count / total_count
        }

    def analyze_volume_trend(self, dataframe: pd.DataFrame, lookback: int = 100) -> Dict[str, Any]:
        """
        分析成交量趋势

        Args:
            dataframe: OHLCV数据
            lookback: 回溯K线数量，默认100根（30分钟框架约50小时）

        Returns:
            成交量趋势分析结果
        """
        if len(dataframe) < lookback:
            return {"trend": "数据不足"}

        recent = dataframe.tail(lookback)
        volumes = recent['volume']

        # 计算成交量移动平均
        volume_ma = volumes.mean()
        current_volume = volumes.iloc[-1]

        # 成交量趋势
        first_half_avg = volumes.head(lookback//2).mean()
        second_half_avg = volumes.tail(lookback//2).mean()

        if second_half_avg > first_half_avg * 1.2:
            trend = "持续放量"
        elif second_half_avg < first_half_avg * 0.8:
            trend = "持续缩量"
        else:
            trend = "平稳"

        # 当前成交量相对于平均值
        volume_ratio = current_volume / volume_ma

        if volume_ratio > 1.5:
            current_status = "异常放量"
        elif volume_ratio > 1.2:
            current_status = "明显放量"
        elif volume_ratio < 0.7:
            current_status = "明显缩量"
        else:
            current_status = "正常"

        return {
            "trend": trend,
            "current_status": current_status,
            "volume_ratio": volume_ratio,
            "current_vs_avg": f"{(volume_ratio - 1) * 100:+.1f}%"
        }
