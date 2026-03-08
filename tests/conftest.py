"""
全局测试配置
============
- 代理修复: 从 .env 读取 EMBED_API_URL，将其中的 CGNAT 段 IP 加入 no_proxy。
- real_config fixture: 从真实 config/default.yaml + .env 加载 Config，供远程测试复用。
"""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path

import pytest
from dotenv import dotenv_values

# ---------------------------------------------------------------------------
# 代理修复: 将内网 API 地址加入 no_proxy（模块加载时立即执行）
# ---------------------------------------------------------------------------

_CGNAT_NETWORK = ipaddress.IPv4Network("100.64.0.0/10")


def _fix_no_proxy() -> None:
    """从 .env 读取 API URL，将 CGNAT 段 IP 加入 no_proxy / NO_PROXY。

    httpx/urllib 的 no_proxy 不支持 CIDR 记法，必须逐个添加具体 IP。
    """
    if not (os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")):
        return

    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if not env_file.exists():
        return

    env_vars = dotenv_values(str(env_file))

    # 收集所有需要绕过代理的 IP
    hosts_to_add: list[str] = []
    for url_key in ("EMBED_API_URL", "LLM_API_URL", "VLM_API_URL"):
        url = env_vars.get(url_key, "")
        if not url:
            continue
        match = re.match(r"https?://([^:/]+)", url)
        if not match:
            continue
        host = match.group(1)
        try:
            if ipaddress.IPv4Address(host) in _CGNAT_NETWORK:
                hosts_to_add.append(host)
        except (ipaddress.AddressValueError, ValueError):
            pass

    if not hosts_to_add:
        return

    for var in ("no_proxy", "NO_PROXY"):
        current = os.environ.get(var, "")
        for host in hosts_to_add:
            if host not in current:
                separator = "," if current else ""
                current = f"{current}{separator}{host}"
        os.environ[var] = current


# 模块加载时执行代理修复
_fix_no_proxy()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_config():
    """从真实 config/default.yaml + .env 加载 Config（session 级缓存）。

    用于远程 API 测试，需要 .env 中配置有效的 API 密钥。
    """
    from video_tree_trm.config import Config

    project_root = Path(__file__).resolve().parent.parent
    yaml_path = project_root / "config" / "default.yaml"
    env_path = project_root / ".env"

    return Config.load(str(yaml_path), env_path=str(env_path))
