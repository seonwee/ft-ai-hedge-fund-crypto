# 实施计划与状态

> 记录本次移植的实施计划与完成状态。可行性分析见工作区根目录
> `PORT_FEASIBILITY_LANGCHAIN.md`，设计详见 `PORT_DESIGN.md`。

## 已确认决策（2026-10-14）

| # | 决策 |
|---|---|
| 1 | LangChain 全部用最新 1.x API |
| 2 | 风险管理节点移植（单标的 ≤ 组合 20% 且 ≤ 可用资金） |
| 3 | 目标侧 requests 版 LLMClient 用 LangChain 重写替换 |
| 4 | 能用 LangChain 的全部重写 |
| 5 | LLM 上下文不注入原始 K 线 |
| 6 | 独立项目 `ft-ai-hedge-fund-crypto/`（本目录即 git 仓库根） |
| 7 | 主周期可配置，默认 1h；辅助周期 30m/4h/1d |
| 8 | 云端 OpenAI 兼容 API，provider/model/base_url 全可配，主备双模型 |
| 9 | 经验闭环采用「核心闭环」（无 RAG/向量/奖励学习） |
| 10 | 不做回测，dry_run 优先 |

## 阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| A 脚手架 | 目录结构、Dockerfile.custom、docker-compose、config.example.json、.env.example | ✅ |
| B 五信号层 | `hedge_fund/signals.py`（hf_* 列 + 信号面板，Hurst 滚动窗口，纯函数 ADX） | ✅ |
| C LangChain 决策层 | `providers.py` / `decision_tools.py` / `decision_chain.py` / `prompt_fusion.py` | ✅ |
| D 风险预算 | `hedge_fund/risk_budget.py`（20% 硬截断） | ✅ |
| E 核心闭环模块 | 从 ft-userdata-llm-share 拷贝 context/experience/learning/utils/indicators | ✅ |
| F 策略主体 | `LLMHedgeFundStrategy.py`（populate_* / 回调 / 降级 / 经验挂接） | ✅ |
| G 本地验证 | 信号层 / 风险预算 / 决策工具 / 提示词 全部单测通过 | ✅ |
| H 容器验证 | docker 内 dry_run 观察（需真实 api key） | ⏳ 待执行 |
| I 首次提交推送 | git init + 关联远程 + 首次 push | ⏳ 进行中 |

## 剩余工作（上线前）

1. 填写真实云端 api key，`docker compose up --build` 启动
2. 容器内 `python -c "import langchain_openai, freqtrade"` 通过
3. 真实模型端点验证 `tool_choice='auto'` 的 function calling 能力
4. dry_run 观察 ≥7 天：胜率 / 调用次数 / 成本 / 延迟 P95
5. 降级演练：断 LLM → 备用模型 → 五信号兜底，三级各验一次
