# -*- coding: utf-8 -*-
"""流水线日志与计时基建。

- setup_logging：控制台 + <ws>/logs/run_<ts>.log 双通道 logger；
- StreamToLogger：把 sys.stdout 的 print 旁路进日志文件（模块内 print 不动）；
- StageTimer：阶段计时器，累积 {stage, segment_id, seconds} 并原子写
  <ws>/logs/timings.json（每记一条重写一次，中断不丢）。
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional


def setup_logging(ws, name: str = "runner", level: int = logging.INFO):
    """返回 (logger, log_file_path)。重复调用不叠加 handler。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if getattr(logger, "_ws_configured", False):
        return logger, logger._log_file  # type: ignore[attr-defined]

    log_dir = ws.logs_dir
    log_file = os.path.join(log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    logger._ws_configured = True  # type: ignore[attr-defined]
    logger._log_file = log_file    # type: ignore[attr-defined]
    return logger, log_file


class StreamToLogger(io.TextIOBase):
    """sys.stdout 替身：控制台照常输出，同时按行写入日志文件。"""

    def __init__(self, log_file: str, original=None):
        self.original = original if original is not None else sys.__stdout__
        self._fh = open(log_file, "a", encoding="utf-8")
        self._buf = ""

    def write(self, s: str) -> int:
        self.original.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._fh.write(time.strftime("%H:%M:%S ") + line + "\n")
            self._fh.flush()
        return len(s)

    def flush(self) -> None:
        self.original.flush()
        self._fh.flush()

    def close(self) -> None:
        self.flush()
        self._fh.close()


class StageTimer:
    """阶段计时：with timer.stage("asr", segment="segment_003"): ..."""

    def __init__(self, ws, out_name: str = "timings.json"):
        self.ws = ws
        self.out_path = os.path.join(ws.logs_dir, out_name)
        self.records: list = []
        self._t_all = time.time()

    def _flush(self) -> None:
        data = {
            "elapsed_total": round(time.time() - self._t_all, 1),
            "records": self.records,
        }
        tmp = self.out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.out_path)

    @contextmanager
    def stage(self, name: str, segment: Optional[str] = None,
              logger=None):
        label = f"{name}@{segment}" if segment else name
        if logger:
            logger.info("===== 阶段 %s 开始 =====", label)
        else:
            print(f"===== 阶段 {label} 开始 =====", flush=True)
        t0 = time.time()
        err = ""
        try:
            yield
        except Exception as e:
            err = str(e)[:200]
            raise
        finally:
            dt = time.time() - t0
            rec = {"stage": name, "segment_id": segment or "",
                   "seconds": round(dt, 1), "error": err}
            self.records.append(rec)
            self._flush()
            msg = f"===== 阶段 {label} 结束，耗时 {dt/60:.2f} 分钟 ====="
            if err:
                msg += f"（失败：{err}）"
            if logger:
                logger.info(msg)
            else:
                print(msg, flush=True)

    def run_timed(self, name: str, segment: Optional[str], fn, *args,
                  logger=None):
        """在 stage 计时内执行 fn(*args)（供后台线程使用）。"""
        with self.stage(name, segment=segment, logger=logger):
            return fn(*args)

    def summary(self) -> dict:
        """按 stage 汇总总耗时。"""
        out: dict = {}
        for r in self.records:
            out[r["stage"]] = round(out.get(r["stage"], 0.0) + r["seconds"], 1)
        out["total"] = round(time.time() - self._t_all, 1)
        return out
