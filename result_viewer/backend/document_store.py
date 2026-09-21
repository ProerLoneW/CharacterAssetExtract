# -*- coding: utf-8 -*-
"""文档快照存储与热加载(文档三 §5.2/§5.3/§5.4)。

- 轮询文档元信息(mtime/size),变化后计算 SHA-256,内容稳定才编译;
- 候选快照校验失败时保留最后可用版本,并记录带位置的错误;
- 快照整体原子替换(单引用赋值 + 锁),产生新 doc_revision;
- 保留当前与上一可用版本供在途请求。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .document_parser import CompileResult, DocumentParser


@dataclass
class DocSnapshot:
    doc_revision: str
    compiled_at: str
    spec: dict | None
    errors: list[dict]
    source_name: str

    @property
    def ok(self) -> bool:
        return self.spec is not None


class DocumentStore:
    def __init__(self, doc_path: Path, parser: DocumentParser, poll_interval_sec: float = 2.0, debounce_ms: int = 300):
        self.doc_path = Path(doc_path)
        self.parser = parser
        self.poll_interval_sec = poll_interval_sec
        self.debounce_ms = debounce_ms
        self._lock = threading.Lock()
        self._current: DocSnapshot | None = None
        self._previous: DocSnapshot | None = None
        self._last_error: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observers: list[Callable[[DocSnapshot], None]] = []

    # -- 生命周期 -----------------------------------------------------------

    def initial_compile(self) -> None:
        """首次编译:成功则可用;失败保留诊断外壳(spec=None)。"""
        result, source_name = self._read_and_compile()
        if result.ok:
            self._publish(result, source_name)
        else:
            self._last_error = [e.to_dict() for e in result.errors]
            self._current = DocSnapshot(
                doc_revision=result.content_sha256,
                compiled_at=_now_iso(),
                spec=None,
                errors=[e.to_dict() for e in result.errors],
                source_name=source_name,
            )

    def start_polling(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._poll_loop, name="doc-hot-reload", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def add_observer(self, fn: Callable[[DocSnapshot], None]) -> None:
        self._observers.append(fn)

    # -- 读取 -----------------------------------------------------------

    @property
    def current(self) -> DocSnapshot | None:
        with self._lock:
            return self._current

    @property
    def previous(self) -> DocSnapshot | None:
        with self._lock:
            return self._previous

    @property
    def last_error(self) -> list[dict]:
        with self._lock:
            return list(self._last_error)

    def revision(self) -> str | None:
        snap = self.current
        return snap.doc_revision if snap else None

    def snapshot_for(self, doc_revision: str | None) -> DocSnapshot | None:
        """在途请求按期望版本取快照;仅接受当前与上一可用版本。"""
        if doc_revision is None:
            return self.current
        for snap in (self.current, self.previous):
            if snap and snap.doc_revision == doc_revision and snap.ok:
                return snap
        return None

    # -- 内部 -----------------------------------------------------------

    def _poll_loop(self) -> None:
        last_sig: tuple[float, int] | None = None
        last_sha: str | None = None
        while not self._stop.wait(self.poll_interval_sec):
            try:
                stat = self.doc_path.stat()
                sig = (stat.st_mtime, stat.st_size)
            except OSError:
                continue
            if sig == last_sig:
                continue
            last_sig = sig
            time.sleep(self.debounce_ms / 1000.0)
            result, source_name = self._read_and_compile()
            if result.content_sha256 == last_sha:
                continue
            last_sha = result.content_sha256
            if result.ok:
                self._publish(result, source_name)
                self._last_error = []
            else:
                # 保留旧快照,只记录错误(文档三 §5.4)
                self._last_error = [e.to_dict() for e in result.errors]

    def _read_and_compile(self) -> tuple[CompileResult, str]:
        try:
            text = self.doc_path.read_text(encoding="utf-8")
        except OSError as exc:
            err = CompileResult(ok=False, content_sha256="")
            from .document_parser import DocumentError

            err.errors.append(DocumentError("DOCUMENT_UNREADABLE", f"文档读取失败: {exc}"))
            return err, self.doc_path.name
        return self.parser.compile(text), self.doc_path.name

    def _publish(self, result: CompileResult, source_name: str) -> None:
        snap = DocSnapshot(
            doc_revision=result.content_sha256,
            compiled_at=_now_iso(),
            spec=result.spec,
            errors=[],
            source_name=source_name,
        )
        with self._lock:
            self._previous = self._current if (self._current and self._current.ok) else self._previous
            self._current = snap
        for fn in list(self._observers):
            try:
                fn(snap)
            except Exception:
                pass


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
