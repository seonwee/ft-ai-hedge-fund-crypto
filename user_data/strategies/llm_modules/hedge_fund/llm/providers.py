"""
LLM Provider 层（LangChain 1.x 重写）

替代两处旧实现：
- 源侧 ai-hedge-fund-crypto/src/llm/__init__.py（langchain 0.3.x 旧 API）
- 目标侧 ft-userdata-llm-share 的 requests 版 LLMClient 的传输部分

设计：
- 统一走 OpenAI 兼容格式（ChatOpenAI + base_url），云端任意兼容供应商可配
- 一主一备两个模型实例；api_key 支持配置值或环境变量注入
  （LLM_API_KEY / LLM_BACKUP_API_KEY），密钥不落 config.json
"""

import logging
import os
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_ENV_KEY_MAP = {
    "primary": "LLM_API_KEY",
    "backup": "LLM_BACKUP_API_KEY",
}


def _resolve_api_key(cfg: Dict[str, Any], role: str) -> str:
    """配置值优先，其次环境变量；都缺失时给占位符并告警（调用必失败，走降级链）。"""
    api_key = cfg.get("api_key") or ""
    if not api_key:
        api_key = os.getenv(_ENV_KEY_MAP.get(role, "LLM_API_KEY"), "")
    if not api_key:
        logger.warning(f"LLM[{role}] 未配置 api_key（配置与环境变量均缺失），该角色不可用")
        api_key = "not-configured"
    return api_key


def get_llm(llm_config: Dict[str, Any], role: str = "primary") -> Optional[ChatOpenAI]:
    """
    构造指定角色的 ChatOpenAI 实例。

    Args:
        llm_config: config.json 的 llm_config 段；backup 角色读其 backup 子段
        role: 'primary' | 'backup'

    Returns:
        ChatOpenAI 实例；配置缺失时返回 None（调用方走降级链）
    """
    if role == "backup":
        cfg = llm_config.get("backup") or {}
        if not cfg or not cfg.get("model"):
            logger.info("未配置备用模型，降级链跳过 backup 角色")
            return None
    else:
        cfg = llm_config or {}
        if not cfg.get("model"):
            logger.error("llm_config 未配置主模型，LLM 决策层不可用")
            return None

    instance = ChatOpenAI(
        model=cfg["model"],
        base_url=cfg.get("api_base", "https://api.openai.com/v1"),
        api_key=_resolve_api_key(cfg, role),
        timeout=cfg.get("timeout", 120),
        max_retries=int(cfg.get("max_retries", 2)),
    )
    logger.info(f"LLM[{role}] 就绪: {cfg['model']} @ {cfg.get('api_base')}")
    return instance


def describe(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """配置摘要（打日志用，不含密钥）。"""
    primary = llm_config or {}
    backup = primary.get("backup", {}) or {}
    return {
        "primary": {"model": primary.get("model"), "api_base": primary.get("api_base")},
        "backup": {"model": backup.get("model"), "api_base": backup.get("api_base")},
    }
