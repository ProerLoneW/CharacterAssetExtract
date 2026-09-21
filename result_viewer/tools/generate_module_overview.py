# -*- coding: utf-8 -*-
"""生成 docs/00_系统模块速览.md(简洁版 pipeline 说明)。

内容全部来自 docs/02_全流程Pipeline说明.md 的模块元数据(经 result_viewer
后端解析器严格校验后提取),不手工维护第二份事实;文档二更新后重新运行:

    python result_viewer/tools/generate_module_overview.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "result_viewer"))

from backend.adapters import REGISTERED_ADAPTER_IDS, RENDERER_CAPABILITIES  # noqa: E402
from backend.document_parser import DocumentParser  # noqa: E402

DOC_02 = REPO / "docs" / "02_全流程Pipeline说明.md"
DOC_00 = REPO / "docs" / "00_系统模块速览.md"

LIFECYCLE_TAGS = {
    "active": "",
    "optional": "（可选）",
    "legacy": "（历史兼容）",
    "external": "（外部）",
    "deprecated": "（已废弃）",
}


def main() -> int:
    parser = DocumentParser(set(REGISTERED_ADAPTER_IDS), RENDERER_CAPABILITIES)
    result = parser.compile(DOC_02.read_text(encoding="utf-8"))
    if not result.ok:
        print("文档二校验失败,不生成速览:")
        for e in result.errors[:8]:
            print(f"  line {e.line} [{e.code}] {e.message}")
        return 1

    spec = result.spec
    system = spec["system"]
    groups = sorted(system["groups"], key=lambda g: g["order"])
    modules = sorted(spec["modules"], key=lambda m: m["order"])
    by_group: dict[str, list[dict]] = {}
    for m in modules:
        by_group.setdefault(m["group"], []).append(m)

    lines: list[str] = []
    lines.append("# 系统模块速览")
    lines.append("")
    lines.append(f"生成日期：{date.today().isoformat()}；来源：[02_全流程Pipeline说明.md](02_全流程Pipeline说明.md)快照 `{result.content_sha256[:12]}`。")
    lines.append("")
    lines.append(
        "本文只回答三件事：系统包含哪些模块、每个模块做什么、代码在哪里。"
        "模块的输入输出契约、路径布局、异常语义以文档二为准；网站中的模块 ID 与本文一致。"
        "文档二更新后运行 `python result_viewer/tools/generate_module_overview.py` 重新生成。"
    )
    lines.append("")
    lines.append(f"共 {len(modules)} 个模块，按业务分组如下（顺序为阅读顺序，不代表执行顺序；依赖关系见文档二与网站流程图）。")
    lines.append("")

    for g in groups:
        mods = by_group.get(g["id"], [])
        if not mods:
            continue
        lines.append(f"## {g['title']}({len(mods)} 个模块)")
        lines.append("")
        lines.append("| 模块 | 功能 | 代码位置 |")
        lines.append("|---|---|---|")
        for m in mods:
            name = f"**{m['title']}**{LIFECYCLE_TAGS.get(m['lifecycle'], '')}"
            mid = f"`{m['id']}`"
            summary = m["summary"].rstrip("。") + "。"
            refs = "<br>".join(f"`{r}`" for r in m.get("source_refs") or []) or "—"
            lines.append(f"| {name}<br>{mid} | {summary} | {refs} |")
        lines.append("")

    lines.append("## 模块 ID 说明")
    lines.append("")
    lines.append("- 模块 ID 是文档二与结果展示网站 URL 中的稳定标识(如 `?module=person_align`)；中文标题可改，ID 不变。")
    lines.append("- 标注（历史兼容）的模块属于旧 CHR 链或兼容支路，默认不在主线讲解路线中；标注（外部）的模块由仓库外工具产出结果。")
    lines.append("- `lifecycle=optional` 的模块按运行配置决定是否执行，不代表系统必然产出对应产物。")
    lines.append("- 「结果文件与服务」上表按文档二基线列出旧展示器 `asset_viewer/`；其新实现位于 `result_viewer/`(后端 `backend/` + 前端 `frontend/`)，设计依据见 [03_结果展示网站设计.md](03_结果展示网站设计.md)。")
    lines.append("")

    DOC_00.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"已生成 {DOC_00.relative_to(REPO)}({len(modules)} 个模块, {len(groups)} 个分组)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
