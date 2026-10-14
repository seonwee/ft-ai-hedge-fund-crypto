# ft-ai-hedge-fund-crypto

> 基于 **freqtrade** + **LangChain 1.x** 的加密货币 AI 交易策略，将
> [ai-hedge-fund-crypto](https://github.com/51bitquant/ai-hedge-fund-crypto) 的
> 「五信号加权集成 + 风险预算 + LLM 下单」决策链移植到 freqtrade 实盘框架，
> 并接入 ft-userdata-llm-share 的核心经验闭环。

## 核心特性

- **五信号量化先验**：趋势跟踪 / 均值回归 / 动量 / 波动率 / 统计套利，加权融合为
  `hf_score`（`llm_modules/hedge_fund/signals.py`，纯 pandas，忠实保留源侧触发逻辑）
- **LangChain 1.x 全量 LLM 决策层**：`bind_tools` + pydantic 参数模型 + 主备模型降级
  （`llm_modules/hedge_fund/llm/`）
- **风险预算**：单标的资本占用 ≤ 组合总值 20% 且 ≤ 可用资金，`custom_stake_amount`
  代码硬截断 + 提示词红线双保险
- **LLM 失效优雅降级**：主模型 → 备用模型 → 五信号确定性层兜底，策略永不裸奔
- **核心经验闭环**：决策日志 / 交易复盘 / 历史统计注入 / 持仓 MFE-MAE 追踪
  （无 RAG / 向量检索 / 奖励学习）
- **上下文不注入原始 K 线**：只注入指标 / 账户 / 持仓 / 情绪 / 经验

## 目录结构

```
ft-ai-hedge-fund-crypto/
├── docker-compose.yml          # Docker 编排
├── Dockerfile.custom           # freqtrade:stable + langchain
├── .env.example                # 环境变量模板（api key）
├── user_data/
│   ├── config.example.json     # freqtrade 配置（含 llm/hf/risk/experience/context 段）
│   ├── logs/                   # 决策/交易/反思日志（运行时生成）
│   └── strategies/
│       ├── LLMHedgeFundStrategy.py
│       └── llm_modules/
│           ├── hedge_fund/     # ★ 新增：五信号 + 风险预算 + LangChain 决策层
│           ├── context/        # 上下文构建（拷贝自 ft-userdata-llm-share）
│           ├── experience/     # 交易日志 + 复盘
│           ├── learning/       # 历史查询 + 模式分析 + 自我反思（核心闭环）
│           ├── utils/          # config / context_builder / position_tracker 等
│           └── indicators/     # TA-Lib 指标计算器
```

## 快速开始

1. 准备云端 API key（OpenAI 兼容格式，如 DeepSeek），写入 `.env` 或
   `user_data/config.json` 的 `llm_config` 段（config.json 已 gitignore）。
2. 复制配置：
   ```bash
   cp user_data/config.example.json user_data/config.json
   cp .env.example .env   # 填入 LLM_API_KEY / LLM_BACKUP_API_KEY
   ```
3. 启动（dry_run 默认开启）：
   ```bash
   docker compose up --build
   ```
   Web UI: http://localhost:8087 （用户名 `freqtrader`）

## 关键配置（`config.json`）

| 段 | 说明 |
|---|---|
| `llm_config` | 主模型 `api_base/model/timeout` + `backup` 备用模型 |
| `hf_config` | 五信号权重、`signal_threshold`(±0.2)、`hurst_window`(126) |
| `risk_management` | `max_position_pct`(20)、`default_leverage`、`max_leverage` |
| `experience_config` | 决策/交易日志路径 |
| `context_config` | `raw_kline_history_points: 0`（不注入原始 K 线） |

主周期通过 freqtrade 配置 `timeframe` 覆盖（默认 1h），辅助周期固定 30m/4h/1d。

## 开发环境

本地开发用 uv 环境（Python 3.14 + langchain 1.x），无需 freqtrade：

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/Scripts/python.exe langchain-core langchain-openai pandas
```

信号层 / 风险预算 / 决策工具可脱离 freqtrade 单测；策略主体需在容器内验证。

## 风险提示

- 加密货币期货波动剧烈，务必先在 `dry_run` 下验证 ≥7 天再考虑实盘
- LLM 决策具有随机性，不保证盈利，需人工监控
- 建议单笔风险 ≤ 5%、杠杆 ≤ 10x
- 详见 `PORT_FEASIBILITY_LANGCHAIN.md`（工作区根目录）的可行性分析与陷阱清单

## 许可证

MIT
