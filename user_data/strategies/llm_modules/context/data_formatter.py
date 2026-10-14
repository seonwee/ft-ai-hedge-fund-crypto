"""
数据格式化器模块
负责格式化历史K线、指标数据等
"""
import logging
import math
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class DataFormatter:
    """数据格式化器"""

    def get_indicator_history(
        self,
        dataframe: pd.DataFrame,
        lookback: int = 100,
        display_points: int = 20
    ) -> Dict[str, List]:
        """
        获取关键指标的历史序列

        Args:
            dataframe: 包含技术指标的数据
            lookback: 回溯K线数量
            display_points: 显示的数据点数量

        Returns:
            指标历史数据字典
        """
        if len(dataframe) < lookback:
            lookback = len(dataframe)

        recent = dataframe.tail(lookback)
        history = {}

        # 选择关键指标提供历史序列
        key_indicators = ['close', 'rsi', 'macd', 'macd_hist', 'ema_20', 'ema_50', 'adx', 'volume']

        for ind in key_indicators:
            if ind in recent.columns:
                values = recent[ind].tolist()
                # 只保留最近的值，格式化为简洁形式
                trimmed = values[-display_points:]
                history[f"{ind}_recent"] = [
                    round(float(v), 2) if pd.notna(v) else None
                    for v in trimmed
                ]

        return history

    def get_raw_kline_history(
        self,
        dataframe: pd.DataFrame,
        count: int,
        extra_fields: Optional[List[str]] = None,
        compact: bool = False,
        stride: int = 1,
        max_rows: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        返回指定数量的原始K线数据文本/表格

        Args:
            dataframe: OHLCV数据
            count: 需要的K线数量
            extra_fields: 额外要显示的字段
            compact: 是否使用紧凑格式
            stride: 步长（用于采样）
            max_rows: 最大行数限制

        Returns:
            格式化的K线历史数据
        """
        result = {"rows": [], "header": None}

        if count <= 0 or dataframe.empty:
            return result

        available_cols = set(dataframe.columns)
        base_fields = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in available_cols]
        if len(base_fields) < 4:
            return result

        extra = [
            field for field in self._ensure_list(extra_fields)
            if field in available_cols and field not in base_fields
        ]

        columns = ['time'] + base_fields + extra

        stride = max(1, stride)
        total_available = min(len(dataframe), count)
        if max_rows:
            max_rows = max(1, max_rows)
            if total_available > max_rows:
                stride = max(stride, math.ceil(total_available / max_rows))

        fetch_count = total_available * stride if stride > 1 else total_available
        subset = dataframe.tail(fetch_count)

        if subset.empty:
            return result

        if stride > 1 and len(subset) > count:
            subset = subset.iloc[::-1][::stride].iloc[::-1]

        subset = subset.tail(total_available)
        if subset.empty:
            return result

        if compact:
            result['header'] = ",".join(columns)
        result['stride'] = stride

        for _, row in subset.iterrows():
            time_str = self.format_timestamp(row.get('date'), row.name)

            if compact:
                values = [time_str]
                for col in columns[1:]:
                    values.append(self.format_number(row.get(col)))
                result['rows'].append(",".join(values))
            else:
                pieces = [
                    f"O:{self.format_number(row.get('open'))}",
                    f"H:{self.format_number(row.get('high'))}",
                    f"L:{self.format_number(row.get('low'))}",
                    f"C:{self.format_number(row.get('close'))}"
                ]

                if 'volume' in base_fields:
                    pieces.append(f"V:{self.format_number(row.get('volume'))}")

                if extra:
                    extras = [
                        f"{field}:{self.format_number(row.get(field))}"
                        for field in extra
                    ]
                    pieces.append(" ".join(extras))

                result['rows'].append(f"{time_str} | {' '.join(pieces)}")

        return result

    def normalize_multi_timeframe_config(
        self,
        cfg: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        将多时间框架配置标准化为 {tf: {candles:int, fields:list}}

        Args:
            cfg: 原始配置
            config: 上下文配置对象

        Returns:
            标准化的配置字典
        """
        normalized: Dict[str, Dict[str, Any]] = {}
        if not cfg:
            return normalized

        default_fields = self._ensure_list(config.get("default_multi_timeframe_fields", []))

        if isinstance(cfg, dict):
            items = cfg.items()
        elif isinstance(cfg, list):
            items = []
            for entry in cfg:
                if isinstance(entry, dict) and entry.get('timeframe'):
                    tf = entry['timeframe']
                    data = entry.copy()
                    data.pop('timeframe', None)
                    items.append((tf, data))
        else:
            return normalized

        for tf, settings in items:
            if not tf:
                continue

            candles = 0
            fields = default_fields
            stride = 1
            compact = None
            max_rows = None

            if isinstance(settings, dict):
                candles = settings.get('candles') or settings.get('count') or settings.get('points') or settings.get('length') or 0
                fields = self._ensure_list(settings.get('fields') or settings.get('extra_fields') or default_fields)
                stride = settings.get('stride', 1)
                compact = settings.get('compact')
                max_rows = settings.get('max_rows')
            else:
                candles = settings
                fields = default_fields

            try:
                candles = int(candles)
            except (TypeError, ValueError):
                candles = 0

            try:
                stride = int(stride)
            except (TypeError, ValueError):
                stride = 1

            stride = max(1, stride)

            normalized[str(tf)] = {
                'candles': max(0, candles),
                'fields': fields,
                'stride': stride,
                'compact': compact,
                'max_rows': max_rows
            }

        return normalized

    def format_timestamp(self, timestamp: Any, fallback_index: Any) -> str:
        """
        格式化时间戳为字符串

        Args:
            timestamp: 时间戳对象
            fallback_index: 备用索引

        Returns:
            格式化的时间字符串
        """
        if isinstance(timestamp, pd.Timestamp):
            ts = timestamp.to_pydatetime()
        elif isinstance(timestamp, datetime):
            ts = timestamp
        elif timestamp is None or (isinstance(timestamp, float) and pd.isna(timestamp)):
            if isinstance(fallback_index, pd.Timestamp):
                ts = fallback_index.to_pydatetime()
            elif isinstance(fallback_index, datetime):
                ts = fallback_index
            else:
                ts = None
        else:
            return str(timestamp)

        if isinstance(ts, datetime):
            return ts.strftime("%Y-%m-%d %H:%M")

        return str(timestamp if timestamp is not None else fallback_index)

    def format_number(self, value: Any, decimals: int = 2) -> str:
        """
        格式化数字为字符串

        Args:
            value: 数值
            decimals: 小数位数

        Returns:
            格式化的数字字符串
        """
        if value is None or pd.isna(value):
            return "-"

        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        formatted = f"{number:.{decimals}f}"
        if '.' in formatted:
            formatted = formatted.rstrip('0').rstrip('.')
        return formatted or "0"

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        """
        确保值为列表类型

        Args:
            value: 任意值

        Returns:
            列表
        """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]
