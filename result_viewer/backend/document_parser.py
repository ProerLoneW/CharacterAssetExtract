# -*- coding: utf-8 -*-
"""PipelineDoc 1.0 解析器(文档一 §6、文档三 §5.1 的 M0 实现)。

职责:把 docs/02_全流程Pipeline说明.md 解析为不可变 PipelineSpec 快照。
- 使用 CommonMark 解析器(markdown-it-py)建立 token 树,不使用跨全文贪婪正则猜结构。
- 严格 JSON:fence 完整 info string 必须是 ``json pipeline-system`` / ``json pipeline-module``,
  普通 ``json`` 示例不参与契约解析。
- 严格校验:未知键拒绝、重复键拒绝、NaN/Infinity 拒绝、枚举/引用/顺序/路径占位符检查、
  依赖无环、adapter/renderer 能力检查、标题与元数据一致性检查。
- 所有错误携带文档行号与字段路径,编译失败则整份候选快照被拒绝。

本模块不读取运行数据,也不产生任何业务产物。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from markdown_it import MarkdownIt

# ---------------------------------------------------------------------------
# 契约常量(来自文档一 §6 / 文档二实例)
# ---------------------------------------------------------------------------

EXPECTED_DOC_ID = "extract-full-pipeline"
SUPPORTED_SCHEMA_MAJOR = 1

MODULE_SECTION_HEADINGS = (
    "作用与判断",
    "输入与前置条件",
    "处理过程",
    "产物与消费",
    "异常、重跑与成本",
    "展示与验收",
)

LIFECYCLES = {"active", "optional", "legacy", "deprecated", "external"}
IMPLEMENTATIONS = {"existing", "external", "planned"}
ARTIFACT_KINDS = {"json", "jsonl", "structured", "directory", "media"}
TIMEBASES = {"source", "global_local", "shot_local", "segment_local", "mixed", "none"}
EVIDENCE_KINDS = {"code", "sample", "design", "unknown"}
MODULE_FILTER_KINDS = {"scope", "shot", "subject", "status", "variant", "segment"}

# 文档二中出现过的受控路径占位符;新占位符属于协议变更,必须显式扩展此表。
KNOWN_PATH_PLACEHOLDERS = {
    "video", "scope", "shot", "subject", "output", "source_video", "asr_file",
    "impl", "workspace", "tracklets", "identity", "asset_bank",
    "variant_output", "bundle_root", "output_file",
}

NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")

SYSTEM_FENCE_INFO = "json pipeline-system"
MODULE_FENCE_INFO = "json pipeline-module"

MODULE_OPEN_RE = re.compile(r"^<!--\s*module:([a-z][a-z0-9_-]*)\s*-->$")
MODULE_CLOSE_RE = re.compile(r"^<!--\s*/module:([a-z][a-z0-9_-]*)\s*-->$")
SECTION_OPEN_RE = re.compile(r"^<!--\s*section:([a-z][a-z0-9_-]*)\s*-->$")
SECTION_CLOSE_RE = re.compile(r"^<!--\s*/section:([a-z][a-z0-9_-]*)\s*-->$")
H2_MODULE_RE = re.compile(r"^(\d+)\s*·\s*(.+)$")

SAFE_LINK_SCHEMES = {"http", "https", "mailto"}


# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------


@dataclass
class DocumentError:
    """一条可定位的文档校验错误。不回传服务器秘密路径,只回传文档内位置。"""

    code: str
    message: str
    line: int | None = None
    field: str | None = None
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "field": self.field,
            "context": self.context,
        }


@dataclass
class CompileResult:
    """编译输出:spec 或错误列表,二者互斥。"""

    ok: bool
    spec: dict[str, Any] | None = None
    errors: list[DocumentError] = field(default_factory=list)
    content_sha256: str = ""

    def error_summary(self) -> str:
        return "; ".join(
            f"[{e.code}] line {e.line or '?'}"
            + (f" field {e.field}" if e.field else "")
            + f": {e.message}"
            for e in self.errors[:8]
        )


# ---------------------------------------------------------------------------
# 严格 JSON 解析(重复键 / NaN / Infinity 拒绝)
# ---------------------------------------------------------------------------


class _DuplicateKeyError(ValueError):
    def __init__(self, key: str):
        super().__init__(f"duplicate JSON key: {key!r}")
        self.key = key


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateKeyError(key)
        seen[key] = value
    return seen


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {name}")


def parse_strict_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_no_duplicate_keys,
        parse_constant=_reject_constant,
    )


# ---------------------------------------------------------------------------
# 安全内容块(文档三 §5.1:只允许有限的 Markdown 节点类型)
# ---------------------------------------------------------------------------


def _inline_text(tokens: list[Any]) -> str:
    """把 inline token 的 children 展开为纯文本;链接保留 [文字](href) 形式供前端受控渲染。"""
    parts: list[str] = []
    href: str | None = None
    for tok in tokens or []:
        if tok.type == "link_open":
            href = (tok.attrs.get("href") if tok.attrs else "") or ""
            parts.append("[")
        elif tok.type == "link_close":
            parts.append(f"]({href})" if href else "]")
            href = None
        elif tok.type == "text":
            parts.append(tok.content)
        elif tok.type == "code_inline":
            parts.append(tok.content)
        elif tok.type == "softbreak":
            parts.append(" ")
    return "".join(parts).strip()


def _inline_links(tokens: list[Any]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for tok in tokens or []:
        if tok.type == "link_open":
            href = tok.attrs.get("href", "") if tok.attrs else ""
            links.append({"href": href})
    return links


def _validate_link_schemes(blocks: list[dict[str, Any]], errors: list[DocumentError], base_line: int) -> None:
    for block in blocks:
        if block["type"] == "paragraph":
            for link in block.get("links", []):
                href = link.get("href", "")
                if href.startswith("#") or not href:
                    continue
                scheme = href.split(":", 1)[0].lower() if ":" in href else ""
                if scheme and scheme not in SAFE_LINK_SCHEMES:
                    errors.append(
                        DocumentError(
                            "UNSAFE_LINK",
                            f"链接协议不受支持: {href[:80]}",
                            line=base_line,
                        )
                    )


def _collect_blocks(tokens: list[Any], start: int, end: int, errors: list[DocumentError], base_line: int) -> list[dict[str, Any]]:
    """把 [start,end) 的 token 切片折叠为安全内容块列表。"""
    blocks: list[dict[str, Any]] = []
    i = start
    while i < end:
        tok = tokens[i]
        ttype = tok.type
        line = (tok.map or [base_line])[0] + 1 if tok.map else base_line
        if ttype == "paragraph_open":
            inline = tokens[i + 1]
            blocks.append(
                {
                    "type": "paragraph",
                    "text": _inline_text(inline.children),
                    "links": _inline_links(inline.children),
                    "line": line,
                }
            )
            i += 3  # paragraph_open, inline, paragraph_close
            continue
        if ttype in ("bullet_list_open", "ordered_list_open"):
            items: list[str] = []
            ordered = ttype == "ordered_list_open"
            j = i + 1
            depth = 0
            while j < end:
                t = tokens[j]
                if t.type in ("bullet_list_open", "ordered_list_open"):
                    depth += 1
                elif t.type in ("bullet_list_close", "ordered_list_close"):
                    if depth == 0:
                        break
                    depth -= 1
                elif t.type == "paragraph_open" and depth == 0:
                    items.append(_inline_text(tokens[j + 1].children))
                j += 1
            blocks.append({"type": "list", "ordered": ordered, "items": items, "line": line})
            i = j + 1
            continue
        if ttype == "fence":
            info = (tok.info or "").strip()
            if info and info.split()[0] in ("json", "text"):
                blocks.append(
                    {
                        "type": "code",
                        "lang": info.split()[0],
                        "text": tok.content.rstrip("\n"),
                        "line": line,
                    }
                )
            # 其他语言的 fence 属于普通示例,不进入安全内容树
            i += 1
            continue
        if ttype == "table_open":
            header: list[str] = []
            rows: list[list[str]] = []
            current: list[str] = []
            in_head = False
            j = i + 1
            while j < end and tokens[j].type != "table_close":
                t = tokens[j]
                if t.type == "thead_open":
                    in_head = True
                elif t.type == "thead_close":
                    in_head = False
                elif t.type == "tr_open":
                    current = []
                elif t.type == "inline" and tokens[j - 1].type in ("th_open", "td_open"):
                    current.append(_inline_text(t.children))
                elif t.type == "tr_close":
                    if in_head:
                        header = current
                    else:
                        rows.append(current)
                j += 1
            blocks.append({"type": "table", "header": header, "rows": rows, "line": line})
            i = j + 1
            continue
        if ttype == "heading_open" and tok.tag in ("h3", "h4"):
            text = _inline_text(tokens[i + 1].children)
            if tok.tag == "h3" and text.strip() in MODULE_SECTION_HEADINGS:
                # 模块内的六个固定三级标题是结构,不是内容
                i += 3
                continue
            blocks.append({"type": "subheading", "text": text, "line": line})
            i += 3
            continue
        # 其余(token/pair 包装)跳过一个;通用包装 token 跳两个
        if getattr(tok, "nesting", 0) == 1 and ttype.endswith("_open"):
            # 跳过未知结构的整个配对,避免误读
            inner = ttype[: -len("_open")] + "_close"
            j = i + 1
            depth = 0
            while j < end:
                t = tokens[j]
                if t.type == ttype:
                    depth += 1
                elif t.type == inner:
                    if depth == 0:
                        break
                    depth -= 1
                j += 1
            i = j + 1
            continue
        i += 1
    return blocks


# ---------------------------------------------------------------------------
# 主解析器
# ---------------------------------------------------------------------------


class DocumentParser:
    """把文档二 Markdown 编译为 PipelineSpec。

    能力集合(adapter/renderer 注册名)由外部注入,解析器本身不依赖具体实现。
    """

    def __init__(self, adapter_capabilities: set[str], renderer_capabilities: set[str]):
        self.adapter_capabilities = adapter_capabilities
        self.renderer_capabilities = renderer_capabilities
        # commonmark + table:文档二正文含 GFM 表格(文档一 §6.1 允许的受控节点之一)
        self._md = MarkdownIt("commonmark").enable("table")

    # -- 公共入口 -----------------------------------------------------------

    def compile(self, text: str) -> CompileResult:
        import hashlib

        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        errors: list[DocumentError] = []
        tokens = self._md.parse(text)

        structure = self._scan_structure(tokens, errors)
        if errors:
            return CompileResult(ok=False, errors=errors, content_sha256=sha)

        system = self._parse_system(structure, errors)
        modules = self._parse_modules(structure, errors)
        if errors:
            return CompileResult(ok=False, errors=errors, content_sha256=sha)

        self._cross_validate(system, modules, structure, errors)
        if errors:
            return CompileResult(ok=False, errors=errors, content_sha256=sha)

        spec = {
            "content_sha256": sha,
            "system": system,
            "modules": [m["meta"] | {"sections": m["sections"], "line": m["line"]} for m in modules],
            "content_sections": structure["sections_content"],
            "module_count": len(modules),
        }
        return CompileResult(ok=True, spec=spec, content_sha256=sha)

    # -- 结构扫描 -----------------------------------------------------------

    def _scan_structure(self, tokens: list[Any], errors: list[DocumentError]) -> dict[str, Any]:
        """线性扫描 token,定位模块/章节边界、系统 fence、模块 fence 与六段标题。"""
        modules: dict[str, dict[str, Any]] = {}
        open_module: str | None = None
        sections: dict[str, list[Any]] = {}
        open_section: str | None = None
        system_fences: list[dict[str, Any]] = []

        for idx, tok in enumerate(tokens):
            line = (tok.map[0] + 1) if tok.map else None
            if tok.type == "html_block":
                content = (tok.content or "").strip()
                m = MODULE_OPEN_RE.match(content)
                if m:
                    if open_module is not None:
                        errors.append(
                            DocumentError("NESTED_MODULE", f"模块 {m.group(1)} 嵌套在未闭合模块 {open_module} 内", line=line)
                        )
                        continue
                    if m.group(1) in modules:
                        errors.append(DocumentError("DUPLICATE_MODULE_ID", f"重复模块标记: {m.group(1)}", line=line))
                    open_module = m.group(1)
                    modules[open_module] = {
                        "marker_line": line,
                        "fence": None,
                        "h2": None,
                        "h3": [],
                        "token_start": idx,
                        "sections_content": {},
                    }
                    continue
                m = MODULE_CLOSE_RE.match(content)
                if m:
                    if open_module != m.group(1):
                        errors.append(
                            DocumentError(
                                "MODULE_MARKUP_MISMATCH",
                                f"模块结束标记 {m.group(1)} 与开始标记 {open_module} 不一致",
                                line=line,
                            )
                        )
                    open_module = None
                    continue
                m = SECTION_OPEN_RE.match(content)
                if m:
                    if open_section is not None:
                        errors.append(DocumentError("NESTED_SECTION", f"章节 {m.group(1)} 嵌套在 {open_section} 内", line=line))
                    open_section = m.group(1)
                    sections[open_section] = []
                    continue
                m = SECTION_CLOSE_RE.match(content)
                if m:
                    open_section = None
                    continue

            if open_module is None and open_section is not None and tok.type == "fence":
                info = (tok.info or "").strip()
                # pipeline-system fence 合法位置就在 system-index 章节内(文档一 §6.1);
                # 其他协议 fence 出现在普通章节才算误放。
                misplaced = info == MODULE_FENCE_INFO or (info == SYSTEM_FENCE_INFO and open_section != "system-index")
                if misplaced:
                    errors.append(
                        DocumentError("MISPLACED_FENCE", f"协议 fence({info})出现在普通章节 {open_section} 内", line=line)
                    )
                    continue
                if info == SYSTEM_FENCE_INFO:
                    system_fences.append(
                        {"content": tok.content, "line": line, "token_index": idx}
                    )
                    continue

            if tok.type == "fence":
                info = (tok.info or "").strip()
                line_no = (tok.map[0] + 1) if tok.map else None
                if info == SYSTEM_FENCE_INFO:
                    system_fences.append({"content": tok.content, "line": line_no, "token_index": idx})
                elif info == MODULE_FENCE_INFO:
                    if open_module is None:
                        errors.append(DocumentError("ORPHAN_MODULE_FENCE", "pipeline-module fence 不在任何模块标记内", line=line_no))
                    else:
                        mod = modules[open_module]
                        if mod["fence"] is not None:
                            errors.append(
                                DocumentError("DUPLICATE_MODULE_FENCE", f"模块 {open_module} 含多个 pipeline-module fence", line=line_no)
                            )
                        mod["fence"] = {"content": tok.content, "line": line_no, "token_index": idx}
                continue

            if tok.type == "heading_open":
                inline = tokens[idx + 1] if idx + 1 < len(tokens) else None
                text = _inline_text(inline.children) if inline is not None else ""
                if open_module is not None:
                    if tok.tag == "h2":
                        modules[open_module]["h2"] = {"text": text, "line": line}
                    elif tok.tag == "h3":
                        modules[open_module]["h3"].append({"text": text, "line": line})
                continue

        if open_module is not None:
            errors.append(DocumentError("UNCLOSED_MODULE", f"模块 {open_module} 没有结束标记"))
        if open_section is not None:
            errors.append(DocumentError("UNCLOSED_SECTION", f"章节 {open_section} 没有结束标记"))
        if len(system_fences) != 1:
            errors.append(
                DocumentError(
                    "SYSTEM_FENCE_COUNT",
                    f"期望恰好 1 个 `{SYSTEM_FENCE_INFO}` fence,实际 {len(system_fences)} 个",
                    line=system_fences[0]["line"] if system_fences else None,
                )
            )

        # 收集模块与章节的正文块
        all_tokens = tokens
        for mod_id, mod in modules.items():
            start = mod["token_start"]
            end = None
            for j in range(start + 1, len(all_tokens)):
                t = all_tokens[j]
                if t.type == "html_block" and MODULE_CLOSE_RE.match((t.content or "").strip()):
                    end = j
                    break
            if end is None:
                end = len(all_tokens)
            mod["token_end"] = end
            if mod["fence"] is not None:
                f_start = mod["fence"]["token_index"]
                blocks = _collect_blocks(all_tokens, f_start + 1, end, errors, mod["marker_line"] or 0)
                mod["sections_content"] = blocks
        for sec_id, _rows in sections.items():
            pass  # 章节正文在下面统一收集

        # 章节正文:重新扫描 section 标记区间
        sec_ranges: list[tuple[str, int, int]] = []
        cur: tuple[str, int] | None = None
        for idx, tok in enumerate(tokens):
            if tok.type != "html_block":
                continue
            content = (tok.content or "").strip()
            m = SECTION_OPEN_RE.match(content)
            if m:
                cur = (m.group(1), idx)
                continue
            m = SECTION_CLOSE_RE.match(content)
            if m and cur and cur[0] == m.group(1):
                sec_ranges.append((cur[0], cur[1], idx))
                cur = None
        sections_content: dict[str, list[dict[str, Any]]] = {}
        for sec_id, s, e in sec_ranges:
            sections_content[sec_id] = _collect_blocks(tokens, s + 1, e, errors, s + 1)
            _validate_link_schemes(sections_content[sec_id], errors, s + 1)

        return {
            "modules": modules,
            "sections_content": sections_content,
            "system_fence": system_fences[0] if system_fences else None,
        }

    # -- 系统元数据 -----------------------------------------------------------

    def _parse_system(self, structure: dict[str, Any], errors: list[DocumentError]) -> dict[str, Any]:
        fence = structure["system_fence"]
        if fence is None:
            return {}
        data = self._load_json(fence["content"], fence["line"], "pipeline-system", errors)
        if data is None or not isinstance(data, dict):
            if data is not None:
                errors.append(DocumentError("SYSTEM_TYPE", "pipeline-system 必须是 JSON 对象", line=fence["line"]))
            return {}
        base_line = fence["line"]
        self._check_keys(
            data,
            required=[
                "doc_id", "schema_version", "content_version", "updated_at", "title", "summary",
                "code_baseline", "worktree_dirty", "evidence_section", "default_route",
                "groups", "routes", "layout_profiles",
            ],
            optional=["sample_run"],
            where="pipeline-system",
            errors=errors,
            line=base_line,
        )
        if data.get("doc_id") != EXPECTED_DOC_ID:
            errors.append(
                DocumentError("DOC_ID_MISMATCH", f"doc_id 必须为 {EXPECTED_DOC_ID}", line=base_line, field="doc_id")
            )
        sv = data.get("schema_version")
        if not isinstance(sv, str) or not SEMVER_RE.match(sv):
            errors.append(DocumentError("BAD_SCHEMA_VERSION", "schema_version 必须形如 X.Y.Z", line=base_line, field="schema_version"))
        else:
            major = int(sv.split(".")[0])
            if major != SUPPORTED_SCHEMA_MAJOR:
                errors.append(
                    DocumentError(
                        "UNSUPPORTED_PROTOCOL",
                        f"协议主版本 {major} 不受支持(当前支持 {SUPPORTED_SCHEMA_MAJOR}.x)",
                        line=base_line,
                        field="schema_version",
                    )
                )
        for f in ("content_version", "title", "summary"):
            if not isinstance(data.get(f), str) or not data.get(f):
                errors.append(DocumentError("BAD_FIELD", f"{f} 必须是非空字符串", line=base_line, field=f))
        if not DATE_RE.match(str(data.get("updated_at", ""))):
            errors.append(DocumentError("BAD_FIELD", "updated_at 必须是 YYYY-MM-DD", line=base_line, field="updated_at"))
        if not (isinstance(data.get("code_baseline"), str) and HEX40_RE.match(data["code_baseline"])):
            errors.append(DocumentError("BAD_FIELD", "code_baseline 必须是 40 位小写十六进制 commit", line=base_line, field="code_baseline"))
        if not isinstance(data.get("worktree_dirty"), bool):
            errors.append(DocumentError("BAD_FIELD", "worktree_dirty 必须是布尔值", line=base_line, field="worktree_dirty"))
        if not isinstance(data.get("evidence_section"), str) or not data.get("evidence_section"):
            errors.append(DocumentError("BAD_FIELD", "evidence_section 必须是非空字符串", line=base_line, field="evidence_section"))

        group_ids: set[str] = set()
        group_orders: set[int] = set()
        groups = data.get("groups")
        if isinstance(groups, list) and groups:
            for g in groups:
                if not isinstance(g, dict) or not NAME_RE.match(str(g.get("id", ""))):
                    errors.append(DocumentError("BAD_GROUP", "group.id 必须匹配 [a-z][a-z0-9_-]*", line=base_line, field="groups"))
                    continue
                if g["id"] in group_ids:
                    errors.append(DocumentError("DUPLICATE_GROUP_ID", f"重复分组 id: {g['id']}", line=base_line, field="groups"))
                group_ids.add(g["id"])
                order = g.get("order")
                if not isinstance(order, int) or order in group_orders:
                    errors.append(DocumentError("BAD_GROUP_ORDER", f"group.order 必须唯一整数: {g['id']}", line=base_line, field="groups.order"))
                group_orders.add(order)
        else:
            errors.append(DocumentError("BAD_FIELD", "groups 必须是非空数组", line=base_line, field="groups"))
        data["_group_ids"] = sorted(group_ids)

        route_ids: set[str] = set()
        routes = data.get("routes")
        if isinstance(routes, list) and routes:
            for r in routes:
                if not isinstance(r, dict) or not NAME_RE.match(str(r.get("id", ""))):
                    errors.append(DocumentError("BAD_ROUTE", "route.id 必须匹配 [a-z][a-z0-9_-]*", line=base_line, field="routes"))
                    continue
                if r["id"] in route_ids:
                    errors.append(DocumentError("DUPLICATE_ROUTE_ID", f"重复路线 id: {r['id']}", line=base_line, field="routes"))
                route_ids.add(r["id"])
                mods = r.get("modules")
                if not isinstance(mods, list) or not mods or not all(isinstance(x, str) for x in mods):
                    errors.append(DocumentError("BAD_ROUTE", f"路线 {r['id']} 的 modules 必须是非空字符串数组", line=base_line, field="routes"))
        else:
            errors.append(DocumentError("BAD_FIELD", "routes 必须是非空数组", line=base_line, field="routes"))
        if data.get("default_route") not in route_ids:
            errors.append(DocumentError("DANGLING_DEFAULT_ROUTE", "default_route 必须指向 routes 中的 id", line=base_line, field="default_route"))

        profile_ids: set[str] = set()
        profiles = data.get("layout_profiles")
        if isinstance(profiles, list) and profiles:
            for p in profiles:
                if not isinstance(p, dict) or not NAME_RE.match(str(p.get("id", ""))):
                    errors.append(DocumentError("BAD_PROFILE", "layout_profiles.id 必须匹配 [a-z][a-z0-9_-]*", line=base_line, field="layout_profiles"))
                    continue
                if p["id"] in profile_ids:
                    errors.append(DocumentError("DUPLICATE_PROFILE_ID", f"重复布局 profile: {p['id']}", line=base_line, field="layout_profiles"))
                profile_ids.add(p["id"])
        else:
            errors.append(DocumentError("BAD_FIELD", "layout_profiles 必须是非空数组", line=base_line, field="layout_profiles"))
        data["_profile_ids"] = sorted(profile_ids)

        sample = data.get("sample_run")
        if sample is not None and not isinstance(sample, dict):
            errors.append(DocumentError("BAD_FIELD", "sample_run 必须是对象", line=base_line, field="sample_run"))
        return data

    # -- 模块元数据 -----------------------------------------------------------

    def _parse_modules(self, structure: dict[str, Any], errors: list[DocumentError]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for mod_id, mod in structure["modules"].items():
            fence = mod.get("fence")
            if fence is None:
                errors.append(
                    DocumentError("MISSING_MODULE_FENCE", f"模块 {mod_id} 缺少 `{MODULE_FENCE_INFO}` fence", line=mod.get("marker_line"))
                )
                continue
            data = self._load_json(fence["content"], fence["line"], f"module:{mod_id}", errors)
            if data is None or not isinstance(data, dict):
                if data is not None:
                    errors.append(DocumentError("MODULE_TYPE", f"模块 {mod_id} 元数据必须是 JSON 对象", line=fence["line"]))
                continue
            if data.get("id") != mod_id:
                errors.append(
                    DocumentError("MODULE_ID_MISMATCH",
                                  f"标记 id={mod_id} 与元数据 id={data.get('id')} 不一致", line=fence["line"], field="id")
                )
            self._check_keys(
                data,
                required=[
                    "id", "title", "order", "group", "lifecycle", "implementation", "summary",
                    "depends_on", "optional_depends_on", "inputs", "source_refs", "evidence",
                    "outputs", "view",
                ],
                optional=[],
                where=f"module:{mod_id}",
                errors=errors,
                line=fence["line"],
            )
            h2 = mod.get("h2")
            if h2 is None:
                errors.append(DocumentError("MISSING_H2", f"模块 {mod_id} 缺少二级标题", line=mod.get("marker_line")))
            else:
                m = H2_MODULE_RE.match(h2["text"])
                if not m:
                    errors.append(
                        DocumentError("BAD_H2", f"模块二级标题应为 '## 序号 · 标题',实际: {h2['text'][:40]}", line=h2["line"])
                    )
                elif m.group(2).strip() != data.get("title"):
                    errors.append(
                        DocumentError(
                            "TITLE_MISMATCH",
                            f"二级标题 '{m.group(2).strip()}' 与元数据 title '{data.get('title')}' 不一致",
                            line=h2["line"],
                            field="title",
                        )
                    )
            h3s = [h["text"].strip() for h in mod.get("h3", [])]
            if h3s != list(MODULE_SECTION_HEADINGS):
                errors.append(
                    DocumentError(
                        "SECTION_HEADINGS_MISMATCH",
                        f"六段三级标题必须依次为 {list(MODULE_SECTION_HEADINGS)},实际: {h3s}",
                        line=(mod.get("h3") or [{}])[0].get("line") if mod.get("h3") else mod.get("marker_line"),
                    )
                )
            # 按 h3 标题行号把正文块归入六段
            sections: dict[str, list[dict[str, Any]]] = {}
            blocks = mod.get("sections_content") or []
            heading_lines = [h["line"] for h in mod.get("h3", [])]
            names = list(MODULE_SECTION_HEADINGS)
            for block in blocks:
                bl = block.get("line")
                # 找到 <= bl 的最后一个标题
                seg = 0
                for hi, hl in enumerate(heading_lines):
                    if hl is not None and bl is not None and bl >= hl:
                        seg = hi
                name = names[seg] if names else "作用与判断"
                sections.setdefault(name, []).append(block)
            for name in MODULE_SECTION_HEADINGS:
                sections.setdefault(name, [])
            _validate_link_schemes(blocks, errors, mod.get("marker_line") or 0)

            out.append({"meta": data, "sections": sections, "line": fence["line"]})
        out.sort(key=lambda m: m["meta"].get("order", 10**9))
        return out

    # -- 交叉校验 -----------------------------------------------------------

    def _cross_validate(
        self,
        system: dict[str, Any],
        modules: list[dict[str, Any]],
        structure: dict[str, Any],
        errors: list[DocumentError],
    ) -> None:
        module_ids = [m["meta"].get("id") for m in modules]
        if len(set(module_ids)) != len(module_ids):
            seen: set[str] = set()
            for mid in module_ids:
                if mid in seen:
                    errors.append(DocumentError("DUPLICATE_MODULE_ID", f"重复模块 id: {mid}", field="id"))
                seen.add(mid)
        id_set = set(module_ids)
        orders: dict[int, str] = {}
        artifact_owner: dict[str, str] = {}

        for m in modules:
            meta = m["meta"]
            mid = meta.get("id")
            line = m["line"]
            if not NAME_RE.match(str(mid or "")):
                errors.append(DocumentError("BAD_MODULE_ID", f"模块 id 非法: {mid!r}", line=line, field="id"))
                continue
            order = meta.get("order")
            if not isinstance(order, int):
                errors.append(DocumentError("BAD_FIELD", "order 必须是整数", line=line, field="order"))
            elif order in orders:
                errors.append(DocumentError("DUPLICATE_ORDER", f"order {order} 同时属于 {orders[order]} 与 {mid}", line=line, field="order"))
            else:
                orders[order] = mid
            if meta.get("group") not in system.get("_group_ids", []):
                errors.append(DocumentError("DANGLING_GROUP", f"模块 {mid} 引用未知分组 {meta.get('group')}", line=line, field="group"))
            if meta.get("lifecycle") not in LIFECYCLES:
                errors.append(DocumentError("BAD_ENUM", f"lifecycle 非法: {meta.get('lifecycle')}", line=line, field="lifecycle"))
            if meta.get("implementation") not in IMPLEMENTATIONS:
                errors.append(DocumentError("BAD_ENUM", f"implementation 非法: {meta.get('implementation')}", line=line, field="implementation"))
            ev = meta.get("evidence")
            if not isinstance(ev, list) or not ev or not set(ev) <= EVIDENCE_KINDS:
                errors.append(DocumentError("BAD_EVIDENCE", "evidence 必须是 code/sample/design/unknown 的非空子集", line=line, field="evidence"))
            if not isinstance(meta.get("summary"), str) or not meta.get("summary"):
                errors.append(DocumentError("BAD_FIELD", "summary 必须是非空字符串", line=line, field="summary"))
            for f in ("depends_on", "optional_depends_on", "inputs"):
                v = meta.get(f)
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    errors.append(DocumentError("BAD_FIELD", f"{f} 必须是字符串数组", line=line, field=f))
            deps = set(meta.get("depends_on") or [])
            odeps = set(meta.get("optional_depends_on") or [])
            if mid in deps or mid in odeps:
                errors.append(DocumentError("SELF_DEPENDENCY", f"模块 {mid} 依赖自身", line=line, field="depends_on"))
            for dep in list(deps) + list(odeps):
                if dep not in id_set:
                    errors.append(DocumentError("DANGLING_DEPENDENCY", f"模块 {mid} 引用未知依赖 {dep}", line=line, field="depends_on"))

            outputs = meta.get("outputs")
            if not isinstance(outputs, list):
                errors.append(DocumentError("BAD_FIELD", "outputs 必须是数组", line=line, field="outputs"))
                outputs = []
            primary = None
            view = meta.get("view")
            if not isinstance(view, dict):
                errors.append(DocumentError("BAD_FIELD", "view 必须是对象", line=line, field="view"))
            else:
                self._check_keys(
                    view, required=["enabled", "renderer", "adapter", "filters", "primary_artifact"],
                    optional=[], where=f"module:{mid}.view", errors=errors, line=line,
                )
                if not isinstance(view.get("enabled"), bool):
                    errors.append(DocumentError("BAD_FIELD", "view.enabled 必须是布尔值", line=line, field="view.enabled"))
                if view.get("renderer") not in self.renderer_capabilities:
                    errors.append(
                        DocumentError(
                            "UNREGISTERED_RENDERER",
                            f"模块 {mid} 引用未注册 renderer: {view.get('renderer')}",
                            line=line,
                            field="view.renderer",
                        )
                    )
                if view.get("adapter") not in self.adapter_capabilities:
                    errors.append(
                        DocumentError(
                            "UNREGISTERED_ADAPTER",
                            f"模块 {mid} 引用未注册 adapter: {view.get('adapter')}",
                            line=line,
                            field="view.adapter",
                        )
                    )
                filters = view.get("filters")
                if not isinstance(filters, list) or not set(filters or []) <= MODULE_FILTER_KINDS:
                    errors.append(DocumentError("BAD_ENUM", f"view.filters 非法: {filters}", line=line, field="view.filters"))
                primary = view.get("primary_artifact")

            out_ids: list[str] = []
            for art in outputs:
                if not isinstance(art, dict):
                    errors.append(DocumentError("BAD_ARTIFACT", f"模块 {mid} 的产物必须是对象", line=line, field="outputs"))
                    continue
                self._check_keys(
                    art,
                    required=["id", "title", "kind", "paths", "record_key", "timebase", "schema", "required"],
                    optional=[],
                    where=f"module:{mid}.outputs",
                    errors=errors,
                    line=line,
                )
                aid = art.get("id")
                if not isinstance(aid, str) or not aid:
                    errors.append(DocumentError("BAD_ARTIFACT", "产物 id 必须是非空字符串", line=line, field="outputs.id"))
                    continue
                if aid in artifact_owner:
                    errors.append(
                        DocumentError(
                            "DUPLICATE_ARTIFACT_ID",
                            f"产物 {aid} 同时由 {artifact_owner[aid]} 与 {mid} 定义",
                            line=line,
                            field="outputs.id",
                        )
                    )
                else:
                    artifact_owner[aid] = mid
                out_ids.append(aid)
                if art.get("kind") not in ARTIFACT_KINDS:
                    errors.append(DocumentError("BAD_ENUM", f"产物 {aid} kind 非法: {art.get('kind')}", line=line, field="outputs.kind"))
                if art.get("timebase") not in TIMEBASES:
                    errors.append(DocumentError("BAD_ENUM", f"产物 {aid} timebase 非法: {art.get('timebase')}", line=line, field="outputs.timebase"))
                if not isinstance(art.get("required"), bool):
                    errors.append(DocumentError("BAD_FIELD", f"产物 {aid} required 必须是布尔值", line=line, field="outputs.required"))
                rk = art.get("record_key")
                if not isinstance(rk, list) or not all(isinstance(x, str) for x in rk):
                    errors.append(DocumentError("BAD_FIELD", f"产物 {aid} record_key 必须是字符串数组", line=line, field="outputs.record_key"))
                paths = art.get("paths")
                if not isinstance(paths, dict) or not paths:
                    errors.append(DocumentError("BAD_FIELD", f"产物 {aid} paths 必须是非空映射", line=line, field="outputs.paths"))
                else:
                    for profile, patterns in paths.items():
                        if profile not in system.get("_profile_ids", []):
                            errors.append(
                                DocumentError(
                                    "DANGLING_PROFILE",
                                    f"产物 {aid} 引用未知布局 profile: {profile}",
                                    line=line,
                                    field="outputs.paths",
                                )
                            )
                        if not isinstance(patterns, list) or not patterns or not all(isinstance(x, str) for x in patterns):
                            errors.append(
                                DocumentError("BAD_FIELD", f"产物 {aid} paths[{profile}] 必须是非空字符串数组", line=line, field="outputs.paths")
                            )
                            continue
                        for pattern in patterns:
                            if "**" in pattern:
                                errors.append(
                                    DocumentError("FORBIDDEN_GLOB",
                                                  f"产物 {aid} 路径禁止 ** 全区扫描: {pattern}", line=line, field="outputs.paths")
                                )
                            for ph in PLACEHOLDER_RE.findall(pattern):
                                if ph not in KNOWN_PATH_PLACEHOLDERS:
                                    errors.append(
                                        DocumentError(
                                            "UNKNOWN_PLACEHOLDER",
                                            f"产物 {aid} 使用未知占位符 {{{ph}}}",
                                            line=line,
                                            field="outputs.paths",
                                        )
                                    )
            if primary is not None and primary not in out_ids:
                errors.append(
                    DocumentError(
                        "DANGLING_PRIMARY_ARTIFACT",
                        f"模块 {mid} 的 primary_artifact {primary} 不在其 outputs 中",
                        line=line,
                        field="view.primary_artifact",
                    )
                )

        # inputs 必须可解析到产物,且生产者必须在 depends_on/optional_depends_on 中
        for m in modules:
            meta = m["meta"]
            mid = meta.get("id")
            line = m["line"]
            deps_all = set(meta.get("depends_on") or []) | set(meta.get("optional_depends_on") or [])
            for inp in meta.get("inputs") or []:
                if inp not in artifact_owner:
                    errors.append(
                        DocumentError("DANGLING_INPUT", f"模块 {mid} 的输入 {inp} 无法解析到任何产物", line=line, field="inputs")
                    )
                elif artifact_owner[inp] not in deps_all:
                    errors.append(
                        DocumentError(
                            "INPUT_DEPENDENCY_MISMATCH",
                            f"模块 {mid} 的输入 {inp} 的生产者 {artifact_owner[inp]} 不在 depends_on/optional_depends_on 中",
                            line=line,
                            field="inputs",
                        )
                    )

        # routes 引用的模块必须存在
        for r in system.get("routes") or []:
            for mid in r.get("modules") or []:
                if mid not in id_set:
                    errors.append(
                        DocumentError("DANGLING_ROUTE_MODULE", f"路线 {r.get('id')} 引用未知模块 {mid}", field="routes")
                    )

        # 依赖无环(depends_on + optional_depends_on)
        graph: dict[str, set[str]] = {}
        for m in modules:
            meta = m["meta"]
            graph[meta.get("id")] = (set(meta.get("depends_on") or []) | set(meta.get("optional_depends_on") or [])) & id_set
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {k: WHITE for k in graph}

        def visit(node: str, stack: list[str]) -> None:
            color[node] = GRAY
            for dep in sorted(graph.get(node, ())):
                if color.get(dep) == GRAY:
                    errors.append(
                        DocumentError(
                            "DEPENDENCY_CYCLE",
                            "依赖存在环: " + " -> ".join(stack + [node, dep]),
                            field="depends_on",
                        )
                    )
                elif color.get(dep) == WHITE:
                    visit(dep, stack + [node])
            color[node] = BLACK

        for node in sorted(graph):
            if color[node] == WHITE:
                visit(node, [])

    # -- 工具 -----------------------------------------------------------

    def _load_json(self, text: str, line: int, where: str, errors: list[DocumentError]) -> Any:
        try:
            return parse_strict_json(text)
        except _DuplicateKeyError as exc:
            errors.append(DocumentError("DUPLICATE_JSON_KEY", f"{where}: {exc}", line=line))
        except ValueError as exc:
            offset = 0
            msg = str(exc)
            m = re.search(r"line (\d+)", msg)
            if m:
                offset = int(m.group(1)) - 1
            errors.append(
                DocumentError(
                    "INVALID_JSON",
                    f"{where}: {msg}",
                    line=(line + 1 + offset) if line is not None else None,
                )
            )
        return None

    def _check_keys(
        self,
        data: dict[str, Any],
        required: list[str],
        optional: list[str],
        where: str,
        errors: list[DocumentError],
        line: int | None,
    ) -> None:
        for key in required:
            if key not in data:
                errors.append(DocumentError("MISSING_FIELD", f"{where} 缺少必填字段 {key}", line=line, field=key))
        allowed = set(required) | set(optional)
        for key in data:
            if key.startswith("_"):
                continue
            if key not in allowed:
                errors.append(DocumentError("UNKNOWN_FIELD", f"{where} 含未声明键 {key}", line=line, field=key))
