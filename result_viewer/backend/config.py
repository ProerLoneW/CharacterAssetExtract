# -*- coding: utf-8 -*-
"""运行时配置加载(文档三 §6.1 viewer.runtime.json 的实现,文件名 viewer_config.json)。

配置只描述部署与路径:监听、base_path、文档路径、前端构建产物目录、允许的运行、
每个运行的工作区根/来源根/产物绑定/路径重写。不复制模块中文名称或业务解释——
那些只来自文档二。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(Exception):
    pass


@dataclass
class RuntimeConfig:
    listen_host: str
    service_port: int
    base_path: str
    doc_path: Path
    frontend_dist: Path | None
    allowed_runs: list[str]
    runs: dict[str, dict]  # run_id -> 原始运行配置(由 run_registry 进一步解析)
    config_dir: Path
    poll_interval_sec: float = 2.0
    debounce_ms: int = 300
    raw: dict = field(default_factory=dict)


def load_config(path: Path) -> RuntimeConfig:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    base = path.parent

    def resolve(p: str | None) -> Path | None:
        if not p:
            return None
        q = Path(p)
        return q if q.is_absolute() else (base / q).resolve()

    required = ["service_port", "doc_path", "allowed_runs", "runs"]
    for key in required:
        if key not in raw:
            raise ConfigError(f"配置缺少必填字段: {key}")

    doc_path = resolve(raw["doc_path"])
    if doc_path is None or not doc_path.is_file():
        raise ConfigError(f"doc_path 指向的文档不存在: {raw['doc_path']}")

    frontend_dist = resolve(raw.get("frontend_dist"))

    base_path = str(raw.get("base_path", "") or "").strip()
    if base_path and not base_path.startswith("/"):
        base_path = "/" + base_path
    if base_path.endswith("/"):
        base_path = base_path.rstrip("/")

    allowed = raw["allowed_runs"]
    runs = raw["runs"]
    if not isinstance(allowed, list) or not all(isinstance(x, str) for x in allowed):
        raise ConfigError("allowed_runs 必须是字符串数组")
    if not isinstance(runs, dict):
        raise ConfigError("runs 必须是对象")
    for run_id in allowed:
        if run_id not in runs:
            raise ConfigError(f"allowed_runs 中的 {run_id} 没有对应的 runs 定义")

    return RuntimeConfig(
        listen_host=str(raw.get("listen_host", "127.0.0.1")),
        service_port=int(raw["service_port"]),
        base_path=base_path,
        doc_path=doc_path,
        frontend_dist=frontend_dist,
        allowed_runs=list(allowed),
        runs=runs,
        config_dir=base,
        poll_interval_sec=float(raw.get("poll_interval_sec", 2.0)),
        debounce_ms=int(raw.get("debounce_ms", 300)),
        raw=raw,
    )
