# -*- coding: utf-8 -*-
"""运行注册表与产物绑定(文档三 §6.1/§6.2)。

- RunDescriptor:一次可浏览的运行(工作区根、布局 profile、来源根、产物绑定)。
- ArtifactBinding:产物 ID -> 受控来源根内的相对路径;custom 占位符按产物显式绑定。
- 路径重写:旧 manifest 中的绝对路径通过已登记规则映射回本地受控根;
  无规则命中的路径不产生本地 media_id,保留 provenance 说明(不偷偷跨运行拼接)。
- 媒体边界:所有对外暴露的文件必须 resolve 后位于某个已登记来源根内(文档三 §8)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


class BindingError(Exception):
    pass


@dataclass
class SourceRoot:
    root_id: str
    path: Path

    def contains(self, candidate: Path) -> bool:
        try:
            candidate.relative_to(self.path)
            return True
        except ValueError:
            return False


@dataclass
class ArtifactBinding:
    artifact_id: str
    run_id: str
    profile: str
    root_id: str
    relative_path: str
    source_run_id: str | None = None
    producer_module_id: str | None = None
    schema_hint: str | None = None
    provenance_status: str = "local"

    def absolute_path(self, roots: dict[str, SourceRoot]) -> Path | None:
        root = roots.get(self.root_id)
        if root is None:
            return None
        return (root.path / self.relative_path).resolve()


@dataclass
class RunDescriptor:
    run_id: str
    video_id: str | None
    title: str
    layout_profile: str
    route_id: str | None
    created_at: str | None
    produced_code_revision: str | None
    roots: dict[str, SourceRoot] = field(default_factory=dict)
    rewrites: list[tuple[re.Pattern[str], str]] = field(default_factory=list)
    bindings: dict[str, ArtifactBinding] = field(default_factory=dict)

    # -- 绑定解析 -----------------------------------------------------------

    def binding_for(self, artifact_id: str) -> ArtifactBinding | None:
        """显式绑定优先;本实现所有布局均按显式绑定工作,未绑定即 unconfigured。"""
        return self.bindings.get(artifact_id)

    def resolve_binding_path(self, artifact_id: str) -> Path | None:
        b = self.binding_for(artifact_id)
        if b is None:
            return None
        p = b.absolute_path(self.roots)
        return p if (p is not None and p.exists()) else None

    # -- 媒体路径安全 -------------------------------------------------------

    def media_target(self, root_id: str, relative_path: str) -> Path | None:
        """把 (root_id, rel) 解析为受控文件;越界或不存在返回 None。"""
        root = self.roots.get(root_id)
        if root is None:
            return None
        rel = Path(relative_path.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            return None
        candidate = (root.path / rel).resolve()
        if not root.contains(candidate):
            return None
        if not candidate.is_file():
            return None
        return candidate

    def rewrite_media_path(self, absolute_or_unknown: str) -> tuple[str, Path] | None:
        """把旧 manifest 绝对路径重写为 (root_id, 本地绝对路径)。

        命中规则但文件不存在时返回 None(media 层显示 MEDIA_MISSING);
        无规则命中同样返回 None —— 显式拒绝优于磁盘碰运气。
        """
        if not absolute_or_unknown:
            return None
        text = absolute_or_unknown.replace("\\", "/")
        # Server runs can reference real paths (including explicitly registered
        # symlink targets), without needing a portable-checkout rewrite first.
        direct = Path(text)
        if direct.is_absolute():
            target = direct.resolve()
            for root_id, root in self.roots.items():
                if root.contains(target) and target.is_file():
                    return (root_id, target)
        for pattern, replacement in self.rewrites:
            if pattern.search(text):
                rewritten = pattern.sub(replacement, text)
                if rewritten.startswith("{run_root}/"):
                    rel = rewritten[len("{run_root}/"):]
                    target = self.media_target("run_root", rel)
                    if target is not None:
                        return ("run_root", target)
                return None
        return None


class RunRegistry:
    def __init__(self, config):
        self._config = config
        self.runs: dict[str, RunDescriptor] = {}
        for run_id in config.allowed_runs:
            self.runs[run_id] = self._build_run(run_id, config.runs[run_id])

    def get(self, run_id: str) -> RunDescriptor | None:
        return self.runs.get(run_id)

    def all_runs(self) -> list[RunDescriptor]:
        return [self.runs[r] for r in self._config.allowed_runs if r in self.runs]

    def _build_run(self, run_id: str, raw: dict) -> RunDescriptor:
        base = self._config.config_dir
        workspace = raw.get("workspace_root")
        if not workspace:
            raise BindingError(f"运行 {run_id} 缺少 workspace_root")

        def resolve(p: str) -> Path:
            q = Path(p)
            return q.resolve() if q.is_absolute() else (base / q).resolve()

        desc = RunDescriptor(
            run_id=run_id,
            video_id=raw.get("video_id"),
            title=str(raw.get("title", run_id)),
            layout_profile=str(raw.get("layout_profile", "custom")),
            route_id=raw.get("route_id"),
            created_at=raw.get("created_at"),
            produced_code_revision=raw.get("produced_code_revision"),
        )
        for root in raw.get("source_roots", []):
            rid = root["root_id"]
            desc.roots[rid] = SourceRoot(root_id=rid, path=resolve(root["path"]))
        if "run_root" not in desc.roots:
            desc.roots["run_root"] = SourceRoot("run_root", resolve(workspace))

        for rule in raw.get("path_rewrites", []):
            # 配置允许 JS 风格 $1 分组引用;转换为 Python re.sub 的 \g<1>
            replacement = re.sub(r"\$(\d+)", r"\\g<\1>", str(rule["replacement"]))
            desc.rewrites.append((re.compile(rule["pattern"]), replacement))

        for artifact_id, b in (raw.get("artifact_bindings") or {}).items():
            desc.bindings[artifact_id] = ArtifactBinding(
                artifact_id=artifact_id,
                run_id=run_id,
                profile=desc.layout_profile,
                root_id=b.get("root_id", "run_root"),
                relative_path=str(b["relative_path"]).replace("\\", "/"),
                source_run_id=b.get("source_run_id"),
                producer_module_id=b.get("producer_module_id"),
                schema_hint=b.get("schema_hint"),
                provenance_status=b.get("provenance_status", "local"),
            )
        return desc
