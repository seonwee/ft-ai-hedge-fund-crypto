"""
技术指标计算器模块
统一计算所有技术指标，避免重复代码
"""
import logging
import pandas as pd
import talib.abstract as ta

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """技术指标计算器"""

    @staticmethod
    def add_trend_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加趋势指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了趋势指标的DataFrame
        """
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)

        return dataframe

    @staticmethod
    def add_momentum_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加动量指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了动量指标的DataFrame
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']

        return dataframe

    @staticmethod
    def add_volatility_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加波动率指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了波动率指标的DataFrame
        """
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upper'] = bollinger['upperband']
        dataframe['bb_middle'] = bollinger['middleband']
        dataframe['bb_lower'] = bollinger['lowerband']

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    @staticmethod
    def add_trend_strength_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加趋势强度指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了趋势强度指标的DataFrame
        """
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    @staticmethod
    def add_volume_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加成交量指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了成交量指标的DataFrame
        """
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)

        return dataframe

    @staticmethod
    def add_all_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        添加所有技术指标

        Args:
            dataframe: OHLCV数据

        Returns:
            添加了所有指标的DataFrame
        """
        dataframe = IndicatorCalculator.add_trend_indicators(dataframe)
        dataframe = IndicatorCalculator.add_momentum_indicators(dataframe)
        dataframe = IndicatorCalculator.add_volatility_indicators(dataframe)
        dataframe = IndicatorCalculator.add_trend_strength_indicators(dataframe)
        dataframe = IndicatorCalculator.add_volume_indicators(dataframe)

        return dataframe
