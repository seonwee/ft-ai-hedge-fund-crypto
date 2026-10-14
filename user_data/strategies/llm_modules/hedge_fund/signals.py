"""
五信号集成层（移植自 ai-hedge-fund-crypto/src/indicators/general_indicators.py）

忠实保留源侧触发逻辑与阈值，产出：
1. dataframe 上的 8 个聚合数值列（仅最新 K 线）：
   hf_score / hf_confidence / hf_trend / hf_mr / hf_mom / hf_vol / hf_stat / hf_adx
2. 完整信号面板 dict（各子信号 signal/confidence/metrics），供 LLM 提示词使用

与源侧的差异（均为修复性改动，已在 PORT_FEASIBILITY_LANGCHAIN.md 记录）：
- calculate_adx 改为纯函数，不再向入参 DataFrame 写 12 个中间列（源侧会污染）
- Hurst 指数改为滚动窗口（源侧对整段序列计算，历史回放存在前视偏差）
- 权重与阈值外置到 config（源侧硬编码）

注意：本模块为独立 pandas 口径（hf_ 前缀），与 TA-Lib 指标列口径不同属正常，
已在提示词中向 LLM 说明。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

SIGNAL_VALUES = {"bullish": 1, "neutral": 0, "bearish": -1}

DEFAULT_WEIGHTS = {
    "trend": 0.25,
    "mean_reversion": 0.20,
    "momentum": 0.25,
    "volatility": 0.15,
    "stat_arb": 0.15,
}


@dataclass
class SignalResult:
    signal: str
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 基础算子（源侧口径，纯 pandas）
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Cutler's RSI（源侧口径：rolling mean，与 TA-Lib 的 Wilder 平滑不同）"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).fillna(0.0)
    loss = (-delta.where(delta < 0, 0.0)).fillna(0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _bollinger(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    sma = df["close"].rolling(window).mean()
    std_dev = df["close"].rolling(window).std()
    return sma + std_dev * 2, sma - std_dev * 2


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """纯函数版 ADX（源侧口径：ewm span 平滑；不写中间列到入参 df）"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    up_move = df["high"] - df["high"].shift()
    down_move = df["low"].shift() - df["low"]
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    plus_di = 100 * (plus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
    minus_di = 100 * (minus_dm.ewm(span=period).mean() / tr.ewm(span=period).mean())
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(span=period).mean()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def _hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
    """Hurst 指数（R/S 法）。传入的 series 应已截取为滚动窗口。"""
    prices = np.asarray(price_series, dtype=float)
    lags = range(2, max_lag)
    tau = [
        max(1e-8, float(np.sqrt(np.std(prices[lag:] - prices[:-lag]))))
        for lag in lags
    ]
    try:
        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(reg[0])
    except (ValueError, RuntimeWarning):
        return 0.5


# ---------------------------------------------------------------------------
# 五个子信号（触发条件与源侧一致）
# ---------------------------------------------------------------------------

def calculate_trend_signals(df: pd.DataFrame) -> SignalResult:
    ema_8 = _ema(df["close"], 8)
    ema_21 = _ema(df["close"], 21)
    ema_55 = _ema(df["close"], 55)
    adx = _adx(df, 14)

    short_trend = ema_8.iloc[-1] > ema_21.iloc[-1]
    medium_trend = ema_21.iloc[-1] > ema_55.iloc[-1]
    trend_strength = float(adx.iloc[-1]) / 100.0

    if short_trend and medium_trend:
        signal, confidence = "bullish", trend_strength
    elif (not short_trend) and (not medium_trend):
        signal, confidence = "bearish", trend_strength
    else:
        signal, confidence = "neutral", 0.5

    return SignalResult(
        signal=signal,
        confidence=float(confidence),
        metrics={
            "adx": float(adx.iloc[-1]),
            "trend_strength": trend_strength,
            "ema_8": float(ema_8.iloc[-1]),
            "ema_21": float(ema_21.iloc[-1]),
            "ema_55": float(ema_55.iloc[-1]),
        },
    )


def calculate_mean_reversion_signals(df: pd.DataFrame) -> SignalResult:
    ma_50 = df["close"].rolling(window=50).mean()
    std_50 = df["close"].rolling(window=50).std()
    z_score = (df["close"] - ma_50) / std_50

    bb_upper, bb_lower = _bollinger(df, 20)
    rsi_14 = _rsi(df, 14)
    rsi_28 = _rsi(df, 28)

    bb_range = bb_upper.iloc[-1] - bb_lower.iloc[-1]
    price_vs_bb = (
        (df["close"].iloc[-1] - bb_lower.iloc[-1]) / bb_range if bb_range > 1e-12 else 0.5
    )

    z_last = float(z_score.iloc[-1])
    if z_last < -2 and price_vs_bb < 0.2:
        signal, confidence = "bullish", min(abs(z_last) / 4, 1.0)
    elif z_last > 2 and price_vs_bb > 0.8:
        signal, confidence = "bearish", min(abs(z_last) / 4, 1.0)
    else:
        signal, confidence = "neutral", 0.5

    return SignalResult(
        signal=signal,
        confidence=float(confidence),
        metrics={
            "z_score": z_last,
            "price_vs_bb": float(price_vs_bb),
            "rsi_14": float(rsi_14.iloc[-1]),
            "rsi_28": float(rsi_28.iloc[-1]),
        },
    )


def calculate_momentum_signals(df: pd.DataFrame) -> SignalResult:
    returns = df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    volume_ma = df["volume"].rolling(21).mean()
    volume_momentum = df["volume"] / volume_ma.replace(0, np.nan)

    momentum_score = float(
        (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]
    )
    volume_confirmation = bool(volume_momentum.iloc[-1] > 1.0) if not np.isnan(volume_momentum.iloc[-1]) else False

    if momentum_score > 0.05 and volume_confirmation:
        signal, confidence = "bullish", min(abs(momentum_score) * 5, 1.0)
    elif momentum_score < -0.05 and volume_confirmation:
        signal, confidence = "bearish", min(abs(momentum_score) * 5, 1.0)
    else:
        signal, confidence = "neutral", 0.5

    return SignalResult(
        signal=signal,
        confidence=float(confidence),
        metrics={
            "momentum_1m": float(mom_1m.iloc[-1]),
            "momentum_3m": float(mom_3m.iloc[-1]),
            "momentum_6m": float(mom_6m.iloc[-1]),
            "volume_momentum": float(volume_momentum.iloc[-1]) if not np.isnan(volume_momentum.iloc[-1]) else 0.0,
        },
    )


def calculate_volatility_signals(df: pd.DataFrame) -> SignalResult:
    returns = df["close"].pct_change()
    hist_vol = returns.rolling(21).std() * math.sqrt(365)
    vol_ma = hist_vol.rolling(63).mean()
    vol_regime = hist_vol / vol_ma.replace(0, np.nan)
    vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

    atr = _atr(df, 14)
    atr_ratio = atr / df["close"]

    current_vol_regime = float(vol_regime.iloc[-1]) if not np.isnan(vol_regime.iloc[-1]) else 1.0
    vol_z = float(vol_z_score.iloc[-1]) if not np.isnan(vol_z_score.iloc[-1]) else 0.0

    if current_vol_regime < 0.8 and vol_z < -1:
        signal, confidence = "bullish", min(abs(vol_z) / 3, 1.0)
    elif current_vol_regime > 1.2 and vol_z > 1:
        signal, confidence = "bearish", min(abs(vol_z) / 3, 1.0)
    else:
        signal, confidence = "neutral", 0.5

    return SignalResult(
        signal=signal,
        confidence=float(confidence),
        metrics={
            "historical_volatility": float(hist_vol.iloc[-1]),
            "volatility_regime": current_vol_regime,
            "volatility_z_score": vol_z,
            "atr_ratio": float(atr_ratio.iloc[-1]),
        },
    )


def calculate_stat_arb_signals(
    df: pd.DataFrame, hurst_window: int = 126
) -> SignalResult:
    returns = df["close"].pct_change()
    skew = returns.rolling(63).skew()
    kurt = returns.rolling(63).kurt()

    # 滚动窗口 Hurst（修复源侧全序列前视偏差）
    window = df["close"].tail(min(hurst_window, len(df)))
    hurst = _hurst_exponent(window)

    skew_last = float(skew.iloc[-1]) if not np.isnan(skew.iloc[-1]) else 0.0

    if hurst < 0.4 and skew_last > 1:
        signal, confidence = "bullish", (0.5 - hurst) * 2
    elif hurst < 0.4 and skew_last < -1:
        signal, confidence = "bearish", (0.5 - hurst) * 2
    else:
        signal, confidence = "neutral", 0.5

    return SignalResult(
        signal=signal,
        confidence=float(max(min(confidence, 1.0), 0.0)),
        metrics={
            "hurst_exponent": float(hurst),
            "skewness": skew_last,
            "kurtosis": float(kurt.iloc[-1]) if not np.isnan(kurt.iloc[-1]) else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# 加权融合（源侧 weighted_signal_combination，阈值外置）
# ---------------------------------------------------------------------------

def weighted_signal_combination(
    signals: Dict[str, SignalResult], weights: Dict[str, float], threshold: float = 0.2
) -> Dict[str, Any]:
    weighted_sum = 0.0
    total_confidence = 0.0

    for strategy, result in signals.items():
        numeric_signal = SIGNAL_VALUES[result.signal]
        weight = weights.get(strategy, 0.0)
        weighted_sum += numeric_signal * weight * result.confidence
        total_confidence += weight * result.confidence

    final_score = weighted_sum / total_confidence if total_confidence > 0 else 0.0

    if final_score > threshold:
        signal = "bullish"
    elif final_score < -threshold:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "signal": signal,
        "score": float(final_score),
        "confidence": float(abs(final_score)),
    }


# ---------------------------------------------------------------------------
# 对外主入口
# ---------------------------------------------------------------------------

def compute_signal_panel(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.2,
    hurst_window: int = 126,
    min_candles: int = 200,
) -> Optional[Dict[str, Any]]:
    """
    计算五信号面板。数据不足时返回 None（调用方跳过 hf 列与面板）。

    返回结构（对齐源侧 analyst_signals.technical_analyst_agent 的语义）：
    {
      "signal": "bullish/neutral/bearish", "score": float, "confidence": float,
      "sub_signals": {trend/mean_reversion/momentum/volatility/stat_arb:
                       {signal, confidence, metrics}}
    }
    """
    if df is None or len(df) < min_candles:
        return None

    weights = weights or DEFAULT_WEIGHTS

    trend = calculate_trend_signals(df)
    mean_reversion = calculate_mean_reversion_signals(df)
    momentum = calculate_momentum_signals(df)
    volatility = calculate_volatility_signals(df)
    stat_arb = calculate_stat_arb_signals(df, hurst_window=hurst_window)

    sub_signals = {
        "trend": trend,
        "mean_reversion": mean_reversion,
        "momentum": momentum,
        "volatility": volatility,
        "stat_arb": stat_arb,
    }

    combined = weighted_signal_combination(sub_signals, weights, threshold)
    combined["sub_signals"] = {
        name: {"signal": r.signal, "confidence": round(r.confidence, 4), "metrics": {
            k: round(v, 6) for k, v in r.metrics.items()
        }}
        for name, r in sub_signals.items()
    }
    return combined


def add_hf_columns(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.2,
    hurst_window: int = 126,
    min_candles: int = 200,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    在 dataframe 最新 K 线上写入 8 个 hf_* 聚合数值列（float），并返回完整信号面板。

    仅写最新一行：本策略只对最新 K 线做 LLM 决策（与 ft-userdata-llm-share 模式一致），
    历史行保持 NaN，避免向 LLM 上下文倾倒无意义数值。
    """
    panel = compute_signal_panel(
        df, weights=weights, threshold=threshold,
        hurst_window=hurst_window, min_candles=min_candles,
    )
    if panel is None:
        return df, None

    sub = panel["sub_signals"]
    last_idx = df.index[-1]
    df.loc[last_idx, "hf_score"] = float(panel["score"])
    df.loc[last_idx, "hf_confidence"] = float(panel["confidence"])
    df.loc[last_idx, "hf_trend"] = float(SIGNAL_VALUES[sub["trend"]["signal"]])
    df.loc[last_idx, "hf_mr"] = float(SIGNAL_VALUES[sub["mean_reversion"]["signal"]])
    df.loc[last_idx, "hf_mom"] = float(SIGNAL_VALUES[sub["momentum"]["signal"]])
    df.loc[last_idx, "hf_vol"] = float(SIGNAL_VALUES[sub["volatility"]["signal"]])
    df.loc[last_idx, "hf_stat"] = float(SIGNAL_VALUES[sub["stat_arb"]["signal"]])
    df.loc[last_idx, "hf_adx"] = float(sub["trend"]["metrics"]["adx"])

    return df, panel


_SIGNAL_CN = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}

_SUB_NAME_CN = {
    "trend": "趋势跟踪",
    "mean_reversion": "均值回归",
    "momentum": "动量",
    "volatility": "波动率",
    "stat_arb": "统计套利",
}

# 面板里保留的关键指标（中文名），其余 metrics 折叠掉，减少噪声
_METRIC_CN = {
    "adx": "ADX",
    "trend_strength": "趋势强度",
    "ema_8": "EMA8",
    "ema_21": "EMA21",
    "ema_55": "EMA55",
    "z_score": "Z分数",
    "price_vs_bb": "布林位置",
    "rsi_14": "RSI14",
    "rsi_28": "RSI28",
    "momentum_1m": "1月动量",
    "momentum_3m": "3月动量",
    "momentum_6m": "6月动量",
    "volume_momentum": "量能比",
    "historical_volatility": "年化波动率",
    "volatility_regime": "波动区间",
    "volatility_z_score": "波动Z分数",
    "atr_ratio": "ATR比率",
    "hurst_exponent": "Hurst指数",
    "skewness": "偏度",
    "kurtosis": "峰度",
}


def _fmt_metrics(metrics: Dict[str, float]) -> str:
    parts = []
    for k, v in metrics.items():
        label = _METRIC_CN.get(k)
        if label is None:
            continue
        if isinstance(v, float):
            parts.append(f"{label}={v:.3f}" if abs(v) < 100 else f"{label}={v:.2f}")
        else:
            parts.append(f"{label}={v}")
    return "，".join(parts)


def format_signal_panel_for_prompt(panel: Dict[str, Any]) -> str:
    """把信号面板格式化为提示词文本块（中文、清晰、独立口径）。"""
    if not panel:
        return "（五信号层数据不足，未生成）"

    sub = panel["sub_signals"]
    score = float(panel["score"])
    combined_cn = _SIGNAL_CN.get(panel["signal"], panel["signal"])

    lines = [
        "【量化信号面板 · 五信号加权集成】",
        f"综合：{combined_cn}（score {score:+.2f}，满分±1，越过 ±0.2 才转多/空）",
        "子信号（各维度独立判断，加权融合后得综合 score）：",
    ]
    for name, sig in sub.items():
        s = _SIGNAL_CN.get(sig["signal"], sig["signal"])
        conf = float(sig["confidence"]) * 100
        metrics = _fmt_metrics(sig["metrics"])
        line = f"  · {_SUB_NAME_CN.get(name, name)}：{s}（置信{conf:.0f}%）"
        if metrics:
            line += f"｜{metrics}"
        lines.append(line)
    lines.append(
        "（hf 为独立 pandas 口径，与上方 TA-Lib 指标数值不同属正常；仅供参考，非硬规则）"
    )
    return "\n".join(lines)
