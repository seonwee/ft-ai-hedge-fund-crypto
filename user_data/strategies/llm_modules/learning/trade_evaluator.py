"""
交易评价系统
在平仓时自动评估交易质量，生成结构化评价
"""
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TradeEvaluator:
    """交易质量评价器"""

    def __init__(self):
        """初始化评价器"""
        # 评价权重配置
        self.weights = {
            'profit': 0.35,          # 盈亏结果
            'risk_management': 0.25,  # 风控质量
            'timing': 0.20,           # 时机把握
            'efficiency': 0.20        # 资金效率
        }

        logger.info("交易评价器已初始化")

    def evaluate_trade(
        self,
        trade_id: int,
        pair: str,
        side: str,
        entry_price: float,
        exit_price: float,
        entry_time: datetime,
        exit_time: datetime,
        profit_pct: float,
        leverage: float,
        stake_amount: float,
        entry_reason: str,
        exit_reason: str,
        position_metrics: Optional[Dict[str, Any]] = None,
        market_changes: Optional[Dict[str, Any]] = None,
        model_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        评估交易质量

        Args:
            trade_id: 交易ID
            pair: 交易对
            side: 方向
            entry_price: 入场价格
            exit_price: 出场价格
            entry_time: 入场时间
            exit_time: 出场时间
            profit_pct: 盈亏百分比（考虑杠杆）
            leverage: 杠杆
            stake_amount: 投入金额
            entry_reason: 入场理由
            exit_reason: 出场理由
            position_metrics: 持仓指标
            market_changes: 市场变化
            model_score: 模型自我评分 (0-100)，如果提供则使用模型评分

        Returns:
            评价结果字典
        """
        try:
            # 【优先使用模型自我评分】
            if model_score is not None:
                total_score = max(0, min(100, model_score))
                grade = self._calculate_grade(total_score)

                # 简化评价，基于模型评分
                logger.info(f"使用模型自我评分: {total_score:.2f}/100")

                return {
                    'trade_id': trade_id,
                    'pair': pair,
                    'side': side,
                    'profit_pct': profit_pct,
                    'total_score': round(total_score, 2),
                    'grade': grade,
                    'scores': {
                        'model_self_evaluation': round(total_score, 2)
                    },
                    'reward': self._calculate_reward(profit_pct, total_score),
                    'comments': {
                        'source': 'model_self_evaluation',
                        'strengths': [],
                        'weaknesses': [],
                        'suggestions': []
                    },
                    'timestamp': datetime.now().isoformat()
                }

            # 【降级方案：使用固定算法评分】
            # 1. 盈亏评分 (0-100)
            profit_score = self._evaluate_profit(profit_pct)

            # 2. 风控评分 (0-100)
            risk_score = self._evaluate_risk_management(
                profit_pct=profit_pct,
                leverage=leverage,
                position_metrics=position_metrics
            )

            # 3. 时机评分 (0-100)
            timing_score = self._evaluate_timing(
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                profit_pct=profit_pct,
                leverage=leverage,
                position_metrics=position_metrics,
                exit_reason=exit_reason
            )

            # 4. 效率评分 (0-100)
            efficiency_score = self._evaluate_efficiency(
                entry_time=entry_time,
                exit_time=exit_time,
                profit_pct=profit_pct,
                position_metrics=position_metrics
            )

            # 5. 综合评分
            total_score = (
                profit_score * self.weights['profit'] +
                risk_score * self.weights['risk_management'] +
                timing_score * self.weights['timing'] +
                efficiency_score * self.weights['efficiency']
            )

            # 6. 评级
            grade = self._calculate_grade(total_score)

            # 7. 生成评语和建议
            comments = self._generate_comments(
                profit_pct=profit_pct,
                profit_score=profit_score,
                risk_score=risk_score,
                timing_score=timing_score,
                efficiency_score=efficiency_score,
                position_metrics=position_metrics
            )

            # 8. 奖励/惩罚值
            reward = self._calculate_reward(
                profit_pct=profit_pct,
                total_score=total_score
            )

            return {
                'trade_id': trade_id,
                'pair': pair,
                'side': side,
                'profit_pct': profit_pct,
                'total_score': round(total_score, 2),
                'grade': grade,
                'scores': {
                    'profit': round(profit_score, 2),
                    'risk_management': round(risk_score, 2),
                    'timing': round(timing_score, 2),
                    'efficiency': round(efficiency_score, 2)
                },
                'reward': round(reward, 4),
                'comments': comments,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"交易评价失败: {e}", exc_info=True)
            return {
                'trade_id': trade_id,
                'error': str(e),
                'total_score': 50,
                'grade': 'C',
                'reward': 0
            }

    def _evaluate_profit(self, profit_pct: float) -> float:
        """
        评估盈亏质量 (0-100)

        Args:
            profit_pct: 盈亏百分比

        Returns:
            盈亏评分
        """
        if profit_pct >= 15:
            return 100
        elif profit_pct >= 10:
            return 90
        elif profit_pct >= 5:
            return 75
        elif profit_pct >= 2:
            return 60
        elif profit_pct >= 0:
            return 50
        elif profit_pct >= -2:
            return 40
        elif profit_pct >= -5:
            return 25
        elif profit_pct >= -10:
            return 10
        else:
            return 0

    def _evaluate_risk_management(
        self,
        profit_pct: float,
        leverage: float,
        position_metrics: Optional[Dict[str, Any]]
    ) -> float:
        """
        评估风控质量 (0-100)

        考虑:
        - 最大回撤控制
        - 杠杆使用合理性
        - 止损执行
        """
        score = 50.0

        if not position_metrics:
            return score

        mae = position_metrics.get('max_loss_pct', 0)
        mfe = position_metrics.get('max_profit_pct', 0)

        # 1. 最大浮亏控制 (±20分)
        if mae > -5:
            score += 20  # 浮亏很小
        elif mae > -10:
            score += 10  # 浮亏中等
        elif mae > -15:
            score += 0   # 浮亏较大
        else:
            score -= 20  # 浮亏巨大

        # 2. 盈利回撤控制 (±15分)
        if mfe > 5 and profit_pct > 0:
            drawdown_pct = (mfe - profit_pct) / mfe * 100
            if drawdown_pct < 20:
                score += 15  # 很好地保护了利润
            elif drawdown_pct < 50:
                score += 5   # 有一定回撤
            else:
                score -= 10  # 回撤严重

        # 3. 杠杆合理性 (±15分)
        if leverage <= 3:
            score += 15  # 保守杠杆
        elif leverage <= 5:
            score += 10  # 中等杠杆
        elif leverage <= 10:
            score += 0   # 高杠杆
        else:
            score -= 10  # 极高杠杆

        return max(0, min(100, score))

    def _evaluate_timing(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        profit_pct: float,
        leverage: float,
        position_metrics: Optional[Dict[str, Any]],
        exit_reason: str
    ) -> float:
        """
        评估时机把握 (0-100)

        考虑:
        - 入场位置质量
        - 出场时机选择

        Args:
            side: 方向 ('long' | 'short')
            entry_price: 入场价格
            exit_price: 出场价格
            profit_pct: 实际盈亏百分比（已考虑杠杆和方向）
            leverage: 杠杆倍数
            position_metrics: 持仓指标
            exit_reason: 出场原因
        """
        score = 50.0

        if not position_metrics:
            return score

        mae = position_metrics.get('max_loss_pct', 0)
        mfe = position_metrics.get('max_profit_pct', 0)

        # 1. 入场位置 (±25分)
        if mae > -2:
            score += 25  # 入场位置极佳
        elif mae > -5:
            score += 15  # 入场位置不错
        elif mae > -10:
            score += 5   # 入场位置一般
        else:
            score -= 10  # 入场位置很差

        # 2. 出场时机 (±25分)
        # 使用profit_pct与mfe比较（都已考虑杠杆和方向）
        if mfe > 0 and abs(mfe - profit_pct) < 2:
            # 接近峰值出场（最大盈利与实际盈利相差<2%）
            score += 25
        elif "止盈" in exit_reason or "目标" in exit_reason:
            # 主动止盈
            score += 15
        elif "止损" in exit_reason or "反转" in exit_reason:
            # 被动止损/反转
            if mfe > 5:
                score -= 15  # 曾有盈利但被动出场
            else:
                score += 5   # 及时止损

        return max(0, min(100, score))

    def _evaluate_efficiency(
        self,
        entry_time: datetime,
        exit_time: datetime,
        profit_pct: float,
        position_metrics: Optional[Dict[str, Any]]
    ) -> float:
        """
        评估资金效率 (0-100)

        考虑:
        - 持仓时间
        - 收益/时间比
        - hold次数
        """
        score = 50.0

        # 计算持仓时长
        duration_hours = (exit_time - entry_time).total_seconds() / 3600

        # 1. 时间效率 (±25分)
        profit_per_hour = profit_pct / max(duration_hours, 0.1)

        if profit_per_hour > 1:
            score += 25  # 极高效率
        elif profit_per_hour > 0.5:
            score += 15  # 高效率
        elif profit_per_hour > 0.1:
            score += 5   # 中等效率
        elif profit_per_hour > 0:
            score += 0   # 低效率
        else:
            score -= 15  # 负效率

        # 2. hold次数 (±25分)
        if position_metrics:
            hold_count = position_metrics.get('hold_count', 0)

            if hold_count <= 3:
                score += 25  # 决策果断
            elif hold_count <= 5:
                score += 10  # 决策一般
            elif hold_count <= 10:
                score += 0   # 决策犹豫
            else:
                score -= 15  # 过度犹豫

        return max(0, min(100, score))

    def _calculate_grade(self, total_score: float) -> str:
        """计算评级"""
        if total_score >= 90:
            return 'A+'
        elif total_score >= 80:
            return 'A'
        elif total_score >= 70:
            return 'B+'
        elif total_score >= 60:
            return 'B'
        elif total_score >= 50:
            return 'C'
        elif total_score >= 40:
            return 'D'
        else:
            return 'F'

    def _generate_comments(
        self,
        profit_pct: float,
        profit_score: float,
        risk_score: float,
        timing_score: float,
        efficiency_score: float,
        position_metrics: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成评语和建议"""
        strengths = []
        weaknesses = []
        suggestions = []

        # 分析优势
        if profit_score >= 75:
            strengths.append("盈利表现优秀")
        if risk_score >= 70:
            strengths.append("风险控制良好")
        if timing_score >= 70:
            strengths.append("时机把握准确")
        if efficiency_score >= 70:
            strengths.append("资金使用高效")

        # 分析弱点
        if profit_score < 50:
            weaknesses.append("盈利能力不足")
            suggestions.append("重新评估入场条件和信号质量")

        if risk_score < 50:
            weaknesses.append("风险控制较差")
            suggestions.append("设置更严格的止损和仓位管理")

        if timing_score < 50:
            weaknesses.append("时机把握欠佳")
            suggestions.append("改进入场和出场信号识别")

        if efficiency_score < 50:
            weaknesses.append("资金效率偏低")
            suggestions.append("避免长时间持有低效仓位")

        # 针对性建议
        if position_metrics:
            mfe = position_metrics.get('max_profit_pct', 0)
            if mfe > 5 and profit_pct < mfe - 5:
                suggestions.append(f"曾有{mfe:+.1f}%浮盈，下次考虑在峰值附近分批止盈")

            hold_count = position_metrics.get('hold_count', 0)
            if hold_count > 8:
                suggestions.append("hold次数过多，要么是入场位不佳，要么是缺乏明确退出策略")

        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'suggestions': suggestions
        }

    def _calculate_reward(self, profit_pct: float, total_score: float) -> float:
        """
        计算奖励/惩罚值

        Args:
            profit_pct: 盈亏百分比
            total_score: 综合评分

        Returns:
            奖励值 (正数=奖励, 负数=惩罚)
        """
        # 基于盈亏的奖励
        profit_reward = profit_pct / 100

        # 基于评分的奖励加成
        score_multiplier = total_score / 50 - 1  # 评分50时为0，100时为+1，0时为-1

        # 综合奖励
        total_reward = profit_reward * (1 + score_multiplier * 0.5)

        return total_reward

    def get_evaluation_summary(self, evaluations: list) -> Dict[str, Any]:
        """
        获取评价摘要统计

        Args:
            evaluations: 评价列表

        Returns:
            统计摘要
        """
        if not evaluations:
            return {'count': 0}

        total_count = len(evaluations)
        avg_score = sum(e.get('total_score', 0) for e in evaluations) / total_count
        avg_reward = sum(e.get('reward', 0) for e in evaluations) / total_count

        grade_counts = {}
        for e in evaluations:
            grade = e.get('grade', 'C')
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        return {
            'count': total_count,
            'avg_score': round(avg_score, 2),
            'avg_reward': round(avg_reward, 4),
            'grade_distribution': grade_counts
        }
