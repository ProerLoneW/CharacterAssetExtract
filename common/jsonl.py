# -*- coding: utf-8 -*-
"""JSONL 读写与按主键更新的通用工具。

所有中间结果落盘为 JSONL，支持：
  - 全量读写
  - 按主键 upsert（单步重跑时只更新本步骤负责的字段）
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Callable, Dict, Iterable, Iterator, List, Type, TypeVar

T = TypeVar("T")


def read_jsonl(path: str) -> Iterator[dict]:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, records: Iterable[dict]) -> int:
    """原子写：先写临时文件再 rename，避免中断产生半个文件。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)) or ".",
                               suffix=".tmp")
    n = 0
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(path: str, cls: Type[T]) -> List[T]:
    return [cls.from_dict(d) for d in read_jsonl(path)]


def load_records_by_key(path: str, cls: Type[T], key: str) -> Dict[str, T]:
    return {getattr(r, key): r for r in load_records(path, cls)}


def upsert_records(path: str, cls: Type[T], updates: Iterable[T], key: str) -> int:
    """按主键合并写回：存在的记录整体替换，不存在的追加。保持原文件顺序。"""
    updates = list(updates)
    upd_map = {getattr(u, key): u for u in updates}
    seen = set()
    merged: List[T] = []
    for old in load_records(path, cls):
        k = getattr(old, key)
        if k in upd_map:
            merged.append(upd_map[k])
            seen.add(k)
        else:
            merged.append(old)
    for u in updates:
        k = getattr(u, key)
        if k not in seen:
            merged.append(u)
    return write_jsonl(path, [r.to_dict() for r in merged])
