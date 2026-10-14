"""
决策质量检查模块 - 反思和自我纠错
提供决策一致性检查和警告，引导LLM反思而非限制行为
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DecisionQualityChecker:
    """决策质量检查器 - 提供反思提示，无强制限制"""

    def __init__(self):
        # 决策缓存: {pair: [decisions]}
        self.decision_cache: Dict[str, List[Dict]] = {}

    def record_decision(
        self,
        pair: str,
        decision_type: str,
        reason: str,
        profit_pct: float = 0,
        confidence: float = 0
    ):
        """
        记录决策

        Args:
            pair: 交易对
            decision_type: 决策类型 ('hold', 'exit', 'entry_long', 'entry_short', etc.)
            reason: 决策理由
            profit_pct: 当前盈亏百分比
            confidence: 置信度
        """
        if pair not in self.decision_cache:
            self.decision_cache[pair] = []

        self.decision_cache[pair].append({
            'time': datetime.now(),
            'type': decision_type,
            'reason': reason[:200],  # 限制长度
            'profit_pct': profit_pct,
            'confidence': confidence
        })

        # 只保留最近50次决策
        if len(self.decision_cache[pair]) > 50:
            self.decision_cache[pair] = self.decision_cache[pair][-50:]

    def check_decision_consistency(
        self,
        pair: str,
        decision_type: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        检查决策一致性

        检测:
        1. 是否连续多次使用相同理由
        2. 理由是否与市场表现矛盾
        3. 是否陷入确认偏差

        返回警告信息，不阻止决策
        """

        if pair not in self.decision_cache:
            return {'status': 'ok', 'warnings': []}

        history = self.decision_cache[pair]

        # 分析最近的hold决策
        recent_holds = [d for d in history[-15:] if d['type'] in ['hold', 'signal_hold']]

        if len(recent_holds) < 3:
            return {'status': 'ok', 'warnings': []}

        # 检测重复理由
        reasons = [h['reason'] for h in recent_holds]
        reason_counts = {}
        for r in reasons:
            key = r[:50]  # 前50字符作为key
            reason_counts[key] = reason_counts.get(key, 0) + 1

        max_count = max(reason_counts.values()) if reason_counts else 0
        most_common = max(reason_counts, key=reason_counts.get) if reason_counts else ""

        warnings = []

        # 警告1: 连续使用相同理由
        if max_count >= 5:
            warnings.append({
                'level': 'high',
                'type': 'stuck_in_loop',
                'message': f'你已连续{max_count}次使用相似理由hold',
                'detail': '市场可能在用价格告诉你：你的判断需要重新评估',
                'repeated_reason': most_common,
                'suggestion': '问自己: 这个理由是否仍然有效？市场是否已经证明我错了？'
            })
        elif max_count >= 3:
            warnings.append({
                'level': 'medium',
                'type': 'repeated_reasoning',
                'message': f'检测到理由重复{max_count}次',
                'detail': '重复理由不等于理由正确',
                'repeated_reason': most_common,
                'suggestion': '寻找反面证据：有哪些信号与你的判断相悖？'
            })

        # 警告2: 连续hold但盈亏无改善
        if len(recent_holds) >= 6:
            # 检查盈亏是否在原地打转
            profits = [h['profit_pct'] for h in recent_holds[-6:]]
            profit_range = max(profits) - min(profits)

            if profit_range < 3.0:  # 盈亏波动小于3%
                warnings.append({
                    'level': 'medium',
                    'type': 'no_progress',
                    'message': f'连续{len(recent_holds)}次hold，但盈亏在原地打转(波动仅{profit_range:.1f}%)',
                    'detail': '持仓没有进展可能说明市场在告诉你：趋势可能不如你想的那么强',
                    'suggestion': '重新评估: 如果现在是空仓，我会在这个位置开仓吗？'
                })

        # 警告3: 连续hold且盈利持续回撤
        if len(recent_holds) >= 4:
            recent_4_profits = [h['profit_pct'] for h in recent_holds[-4:]]
            # 检查是否持续下降
            is_declining = all(recent_4_profits[i] < recent_4_profits[i-1] for i in range(1, len(recent_4_profits)))

            if is_declining and recent_4_profits[0] > 0 and recent_4_profits[-1] < recent_4_profits[0] * 0.5:
                warnings.append({
                    'level': 'high',
                    'type': 'profit_erosion',
                    'message': f'盈利持续回撤: {recent_4_profits[0]:.1f}% → {recent_4_profits[-1]:.1f}%',
                    'detail': '连续hold过程中盈利回撤超过50%',
                    'suggestion': '考虑: 盈利回撤是正常回调还是趋势反转的早期信号？'
                })

        return {
            'status': 'warning' if warnings else 'ok',
            'warnings': warnings,
            'hold_count': len(recent_holds),
            'repeated_reason_count': max_count,
            'decision_diversity': len(reason_counts) / len(reasons) if reasons else 1.0
        }

    def generate_reflection_prompt(
        self,
        decision_type: str,
        current_state: Dict[str, Any],
        consistency_check: Dict[str, Any]
    ) -> str:
        """
        生成反思提示

        基于决策一致性检查结果，生成引导LLM反思的提问
        不是命令，而是提问
        """

        if consistency_check['status'] == 'ok':
            return ""

        prompt_parts = ["【决策质量反思】"]

        for warning in consistency_check['warnings']:
            prompt_parts.append("")

            if warning['level'] == 'high':
                prompt_parts.append(f"[严重警告] {warning['message']}")
            elif warning['level'] == 'medium':
                prompt_parts.append(f"[提醒] {warning['message']}")
            else:
                prompt_parts.append(f"[注意] {warning['message']}")

            if 'detail' in warning:
                prompt_parts.append(f"   {warning['detail']}")

            if warning['type'] == 'stuck_in_loop':
                prompt_parts.append("")
                prompt_parts.append("   重复理由:")
                prompt_parts.append(f"   \"{warning['repeated_reason']}\"")
                prompt_parts.append("")
                prompt_parts.append("   深度反思提问:")
                prompt_parts.append("   1. 这个理由从第一次到现在，市场环境是否已经改变？")
                prompt_parts.append("   2. 你是否在寻找支持自己观点的证据，而忽略了反面证据？")
                prompt_parts.append("   3. 如果市场继续不按你的预期走，你的plan B是什么？")
                prompt_parts.append("   4. 假设你是空仓，现在会在这个位置开仓吗？为什么？")

            elif warning['type'] == 'no_progress':
                prompt_parts.append("")
                prompt_parts.append("   思考:")
                prompt_parts.append("   • 趋势可能没有你想象的那么强")
                prompt_parts.append("   • 市场在横盘震荡，持仓可能不是最优资金使用")
                prompt_parts.append("   • 是否有更好的交易机会？")

            elif warning['type'] == 'profit_erosion':
                prompt_parts.append("")
                prompt_parts.append("   关键问题:")
                prompt_parts.append("   • 盈利回撤是正常回调还是趋势反转的早期信号？")
                prompt_parts.append("   • 你在hold的过程中，哪些条件变好了？哪些变坏了？")
                prompt_parts.append("   • 如果盈利继续回撤到负值，你会后悔没有在盈利时止盈吗？")

            if 'suggestion' in warning:
                prompt_parts.append("")
                prompt_parts.append(f"   建议: {warning['suggestion']}")

        prompt_parts.append("")
        prompt_parts.append("注意: 以上是反思提示，不是强制要求。")
        prompt_parts.append("你仍然可以选择任何决策，但请基于深入思考，而非重复模式。")

        return "\n".join(prompt_parts)

    def get_decision_summary(self, pair: str) -> Dict[str, Any]:
        """
        获取决策摘要

        用于分析决策模式
        """
        if pair not in self.decision_cache:
            return {'total': 0}

        history = self.decision_cache[pair]

        # 统计不同类型的决策
        decision_types = {}
        for d in history:
            dt = d['type']
            decision_types[dt] = decision_types.get(dt, 0) + 1

        # 统计hold决策的理由多样性
        hold_decisions = [d for d in history if d['type'] in ['hold', 'signal_hold']]
        if hold_decisions:
            reasons = [h['reason'][:50] for h in hold_decisions]
            unique_reasons = len(set(reasons))
            diversity_score = unique_reasons / len(hold_decisions)
        else:
            diversity_score = 1.0

        return {
            'total': len(history),
            'decision_types': decision_types,
            'hold_count': len(hold_decisions),
            'diversity_score': diversity_score,
            'recent_decisions': history[-5:]
        }

    def clear_pair_history(self, pair: str):
        """清除交易对的决策历史"""
        if pair in self.decision_cache:
            del self.decision_cache[pair]
            logger.info(f"已清除 {pair} 的决策历史")

    def clear_all_history(self):
        """清除所有决策历史"""
        self.decision_cache.clear()
        logger.info("已清除所有决策历史")
