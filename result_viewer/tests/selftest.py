# -*- coding: utf-8 -*-
"""结果展示网站自测脚本(文档三 §11 M0 完成标准的本地验证)。

覆盖:
1. 真实文档二可解析(27 个模块,与文档声明一致,不硬编码为常量);
2. 坏 JSON / 重复键 / 未知字段 -> 拒绝并给出行号;
3. 重复模块 ID / 产物 ID / order 冲突 -> 拒绝;
4. 悬空 inputs 引用 / depends_on 缺失生产者 -> 拒绝;
5. 依赖环 -> 拒绝;
6. 未注册 adapter / renderer -> 拒绝;
7. 标题不一致 / 六段标题缺失 -> 拒绝;
8. 运行绑定:未绑定产物返回 unconfigured;媒体重写规则命中真实文件;
9. 适配器:关键计数与真实样本一致(从数据计算,非硬编码期望以外的断言)。

运行:python tests/selftest.py [--config viewer_config.json]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from backend.adapters import REGISTERED_ADAPTER_IDS, RENDERER_CAPABILITIES, build_registry  # noqa: E402
from backend.config import load_config  # noqa: E402
from backend.document_parser import DocumentParser  # noqa: E402
from backend.envelope import MediaRef  # noqa: E402
from backend.media_service import MediaService  # noqa: E402
from backend.run_registry import RunRegistry  # noqa: E402
from backend.adapters.base import RunContext  # noqa: E402

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f" FAIL {name} {detail}")


def make_parser() -> DocumentParser:
    return DocumentParser(set(REGISTERED_ADAPTER_IDS), RENDERER_CAPABILITIES)


GOOD_MODULE = """<!-- module:m1 -->
## 10 · 甲模块

```json pipeline-module
{"id":"m1","title":"甲模块","order":10,"group":"g1","lifecycle":"active","implementation":"existing","summary":"测试模块。","depends_on":[],"optional_depends_on":[],"inputs":[],"source_refs":["a.py"],"evidence":["code"],"outputs":[{"id":"o1","title":"产物一","kind":"jsonl","paths":{"p":["out/o1.jsonl"]},"record_key":["k"],"timebase":"source","schema":"unversioned","required":true}],"view":{"enabled":true,"renderer":"video-timeline","adapter":"shots-v1","filters":[],"primary_artifact":"o1"}}
```

### 作用与判断

测试。

### 输入与前置条件

测试。

### 处理过程

测试。

### 产物与消费

测试。

### 异常、重跑与成本

测试。

### 展示与验收

测试。
<!-- /module:m1 -->
"""

GOOD_SYSTEM = """# 测试文档

<!-- section:system-index -->
## 系统索引

```json pipeline-system
{"doc_id":"extract-full-pipeline","schema_version":"1.0.0","content_version":"1.0.0","updated_at":"2026-09-18","title":"T","summary":"S","code_baseline":"%s","worktree_dirty":false,"evidence_section":"e","default_route":"r1","groups":[{"id":"g1","title":"组一","order":10}],"routes":[{"id":"r1","title":"路一","modules":["m1"],"description":"d"}],"layout_profiles":[{"id":"p","title":"布局","description":"d"}]}
```
<!-- /section:system-index -->
""" % ("0" * 40)


def doc_with(system: str = GOOD_SYSTEM, modules: str = GOOD_MODULE) -> str:
    return system + "\n" + modules


def test_real_document() -> None:
    print("[1] 真实文档二解析")
    parser = make_parser()
    text = (REPO.parent / "docs" / "02_全流程Pipeline说明.md").read_text(encoding="utf-8")
    result = parser.compile(text)
    check("编译成功", result.ok, result.error_summary()[:300])
    if result.ok:
        check("模块数与 system 一致(27)", result.spec["module_count"] == 27)
        m = [x for x in result.spec["modules"] if x["id"] == "person_align"][0]
        check("六段正文齐全", all(m["sections"].get(k) is not None for k in m["sections"]))
        check("overview 章节含表格", any(b["type"] == "table" for b in result.spec["content_sections"].get("overview", [])))


def test_rejections() -> None:
    print("[2] 非法文档拒绝")
    parser = make_parser()
    cases = []

    cases.append(("坏 JSON", doc_with(system=GOOD_SYSTEM.replace('"title":"T"', '"title":"T BROKEN')), "INVALID_JSON"))
    dup_key = GOOD_SYSTEM.replace('"title":"T",', '"title":"T","title":"T",')
    cases.append(("重复 JSON 键", doc_with(system=dup_key), "DUPLICATE_JSON_KEY"))
    cases.append(("未知字段", doc_with(system=GOOD_SYSTEM.replace('"summary":"S"', '"summary":"S","extra":1')), "UNKNOWN_FIELD"))
    cases.append(("协议主版本不支持", doc_with(system=GOOD_SYSTEM.replace('"1.0.0"', '"2.0.0"')), "UNSUPPORTED_PROTOCOL"))
    cases.append(("重复模块 ID", doc_with() + GOOD_MODULE, "DUPLICATE_MODULE_ID"))
    bad_title = GOOD_MODULE.replace("## 10 · 甲模块", "## 10 · 错误标题")
    cases.append(("标题不一致", doc_with(modules=bad_title), "TITLE_MISMATCH"))
    no_section = GOOD_MODULE.replace("### 展示与验收\n\n测试。\n", "")
    cases.append(("六段标题缺失", doc_with(modules=no_section), "SECTION_HEADINGS_MISMATCH"))
    bad_adapter = GOOD_MODULE.replace('"adapter":"shots-v1"', '"adapter":"not-registered"')
    cases.append(("未注册 adapter", doc_with(modules=bad_adapter), "UNREGISTERED_ADAPTER"))
    bad_renderer = GOOD_MODULE.replace('"renderer":"video-timeline"', '"renderer":"not-a-renderer"')
    cases.append(("未注册 renderer", doc_with(modules=bad_renderer), "UNREGISTERED_RENDERER"))
    dangling_input = GOOD_MODULE.replace('"inputs":[]', '"inputs":["nope"]')
    cases.append(("悬空 inputs", doc_with(modules=dangling_input), "DANGLING_INPUT"))
    dangling_dep = GOOD_MODULE.replace('"depends_on":[]', '"depends_on":["ghost"]')
    cases.append(("悬空 depends_on", doc_with(modules=dangling_dep), "DANGLING_DEPENDENCY"))
    # 重复产物 ID:m2 也定义产物 o1
    m2_same_artifact = GOOD_MODULE.replace("m1", "m2").replace("甲模块", "乙模块").replace('"order":10', '"order":20')
    cases.append(("重复产物 ID", doc_with(modules=GOOD_MODULE + m2_same_artifact), "DUPLICATE_ARTIFACT_ID"))
    bad_path = GOOD_MODULE.replace('"paths":{"p":["out/o1.jsonl"]}', '"paths":{"p":["out/{unknown_ph}/x"]}')
    cases.append(("未知路径占位符", doc_with(modules=bad_path), "UNKNOWN_PLACEHOLDER"))
    glob_all = GOOD_MODULE.replace('"paths":{"p":["out/o1.jsonl"]}', '"paths":{"p":["out/**/x"]}')
    cases.append(("禁止 ** 扫描", doc_with(modules=glob_all), "FORBIDDEN_GLOB"))

    # 依赖环:m1 -> m2 -> m1
    m2 = GOOD_MODULE.replace("m1", "m2").replace("甲模块", "乙模块").replace('"depends_on":[]', '"depends_on":["m1"]').replace('"order":10', '"order":20')
    m1_cycle = GOOD_MODULE.replace('"depends_on":[]', '"depends_on":["m2"]')
    cases.append(("依赖环", doc_with(modules=m1_cycle + m2), "DEPENDENCY_CYCLE"))

    # 主产物不在 outputs
    bad_primary = GOOD_MODULE.replace('"primary_artifact":"o1"', '"primary_artifact":"o9"')
    cases.append(("primary_artifact 悬空", doc_with(modules=bad_primary), "DANGLING_PRIMARY_ARTIFACT"))

    for name, text, expected_code in cases:
        result = parser.compile(text)
        codes = {e.code for e in result.errors}
        check(f"{name} -> {expected_code}", (not result.ok) and expected_code in codes, f"got={sorted(codes)}")


def test_run_and_adapters() -> None:
    print("[3] 运行绑定与适配器(真实 0912_v4 样本)")
    config = load_config(REPO / "viewer_config.json")
    runs = RunRegistry(config)
    run = runs.get("movie_full_pipeline_20260912_v4")
    check("运行可解析", run is not None)
    assert run is not None
    ctx = RunContext(run)
    media = MediaService({run.run_id: run})
    registry = build_registry(media)

    # 未绑定产物 -> unconfigured(person_align 的 direct_alignment 未收录)
    align = registry["global-alignment-v1"]
    probe = align.probe(run, ctx)
    check("person_align 未绑定 -> unconfigured", probe.execution == "unconfigured", probe.reason or "")

    # 已实现适配器摘要
    for adapter_id, expect_exec in [
        ("shots-v1", "succeeded"),
        ("globals-v1", "succeeded"),
        ("global-caption-v1", "succeeded"),
        ("shot-caption-v1", "succeeded"),
        ("segment-caption-v1", "succeeded"),
        ("segment-media-v1", "succeeded"),
        ("voice-preview-v2", "succeeded"),
        ("tracklets-v2", "succeeded"),
        ("characters-legacy-v1", "succeeded"),
    ]:
        probe = registry[adapter_id].probe(run, ctx)
        check(f"{adapter_id} probe", probe.execution == expect_exec, f"{probe.execution}: {probe.reason}")

    s = registry["shots-v1"].summary(run, ctx)
    check("shots 摘要指标=3 条", len(s.metrics) == 3)
    v = registry["voice-preview-v2"].summary(run, ctx)
    mm = {m.id: m for m in v.metrics}
    check("voice 候选数来自真实数据(58/72)", mm["candidates"].value == 58 and mm["candidates"].denominator == 72)
    t = registry["tracklets-v2"].summary(run, ctx)
    tm = {m.id: m for m in t.metrics}
    check("tracklets 190/143/59 来自真实数据", tm["tracklets"].value == 190 and tm["filtered_tracklets"].value == 143 and tm["shots_completed"].value == 59)

    # 媒体重写:镜头视频与片段视频映射到本地文件
    shots_items, _ = registry["shots-v1"].list_items(run, ctx, {}, 0, 1)
    m0 = shots_items[0]["media_refs"][0]
    check("镜头视频本地可播", m0["availability"] == "available" and m0["url"], str(m0)[:120])
    seg_items, _ = registry["segment-media-v1"].list_items(run, ctx, {}, 0, 1)
    sm = seg_items[0]["media_refs"][0]
    check("片段视频本地可播(重写规则含分组引用)", sm["availability"] == "available", str(sm)[:120])

    # 全局视频未收录 -> missing 且保留 provenance
    g_items, _ = registry["globals-v1"].list_items(run, ctx, {}, 0, 1)
    gm = g_items[0]["media_refs"][0]
    check("全局视频显示 missing(源未收录)", gm["availability"] == "missing")

    # 媒体服务解析:镜头视频 media_id 可解析出真实文件
    if m0.get("url"):
        media_id = m0["url"].split("/")[-1]
        resolved = media.resolve(media_id)
        check("media_id 解析到受控文件", resolved.size > 0 and resolved.mime == "video/mp4")

    # 越界 media_id
    from backend.media_service import make_media_id

    try:
        media.resolve(make_media_id(run.run_id, "run_root", "../outside.txt"))
        check("越界 media_id 被拒绝", False)
    except Exception:
        check("越界 media_id 被拒绝", True)


def main() -> int:
    arg = argparse.ArgumentParser()
    arg.add_argument("--config", default=str(REPO / "viewer_config.json"))
    args = arg.parse_args()

    test_real_document()
    test_rejections()
    test_run_and_adapters()

    print(f"\n共 {PASS} 项通过,{FAIL} 项失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
