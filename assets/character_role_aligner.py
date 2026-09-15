#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CHR → caption 角色（S-ID）对齐器：统一视觉 / 音频资产的 ID 体系。

问题背景
--------
* caption 链（mimo_*_captioner*）由 global caption 建立人物注册表
  ``subject_registry``，角色 ID 为 ``S1``/``S2``/...，作用域为单个
  global clip（记作 ``global_0040:S1``）。音频简洁链
  （mimo_global_captioner_speaker_v2 → export_all_asr_audio_review）
  直接把 ASR 归属到这套 S-ID。
* 视觉链 Stage3（visual_character_registry_builder）按人脸/身体 embedding
  聚类，产出自己的视觉身份 ``CHR001``/``CHR002``/...，与 S-ID 不打通。

本模块做的事
------------
Stage1.5（semantic_role_tracklet_refiner）已经把每条 tracklet 对齐到
``semantic_scope_id:subject_id``；Stage3 的 registry.json 记录了每个 CHR
由哪些 tracklet 组成。因此对每个 CHR 取其成员 tracklet 的 S-ID 加权投票，
即可把视觉身份映射回 caption 角色：

    weight(tracklet) = confidence × decision_factor
    decision_factor: keep_primary=1.0, keep_supporting=0.85, keep_context=0.5
    （reject_*/review/unknown/non_person 不计票）

投票先按 ``(scope, shot_id)`` 局部聚合——同一 shot 内同一 subject 只取最大
单票（shot_id 为 tracklet_id 去掉 ``_T\d+`` 后缀；轨迹碎片不会让同一段画面
重复计票），再跨 shot 求和得到 scope 总票。

判定（按 scope 独立进行，同一 CHR 可跨多个 global 映射到不同 S-ID）：
  * confirmed   : 第一名票数占比 ≥ --min-share 且总权重 ≥ --min-weight
  * contested   : 前两名占比差 < --contest-margin
  * unresolved  : 无任何有效证据

**只有 confirmed 会进入 role_claims 并物化**；contested/weak/unresolved
仅保留在报告的 ``chr_to_roles`` 中，不参与 winner 竞争、不产出资产目录。

产物（--output 下）：
  role_alignment.json           完整报告（投票明细、冲突、未对齐项）
  characters/<scope>/<S_ID>/    物化的统一角色资产目录（可选）
      visual/                   复制自 --asset-bank 的 CHR 画廊（face/body/...）
      voice/                    复制自 --voice-export/voice_asset_manifest.jsonl 中
                                review_status=approved 且 voice_link.status=matched
                                行的 consolidated 净语音 reference_audio；无 manifest
                                或无获批行则不复制任何语音（preview 不会自动晋升），
                                并记 voice_skipped_reason；复制前清空旧 voice/ 防残留
      role_character.json       该角色的汇总清单

冲突处理：同一 (scope, S-ID) 被多个 CHR 认领时，权重最高者物化到
``<S_ID>/``，其余记入报告的 ``id_conflicts``，画廊仍保留在 Stage4 的
CHR 目录中，不静默合并。

可选开关（均默认关闭；不开时产物与旧版完全一致）：
  * --multi-chr-gallery-review（P0-3）：对 id_conflicts 的落选 CHR 用
    Stage3 tracklet_features.json 的人脸 embedding 做画廊复核，结论写入
    id_conflicts[].gallery_review；判定 compatible 的落选者登记为同一角色
    的附加独立画廊引用（role_gallery_refs / role_character.json 的
    related_visual_characters），不复制/合并其资产、不改 winner 语义。
  * --allow-low-frequency-admission（P1-3）：weak 但票重达到
    --low-freq-min-weight、无竞争且有可靠视觉资产的 CHR 允许物化并进入
    role_to_chr，role_character.json 记 admission_tier=low_frequency。

用法::

    python assets/character_role_aligner.py \
        --identity   <Stage3 输出目录> \
        --stage3-view <Stage1.5 输出>/stage3_inputs/primary \
        --asset-bank <Stage4 输出目录> \
        --voice-export <export_all_asr_audio_review 输出目录> \
        --output <对齐输出目录>

    # 可选 I2I 桥接导出（I2I_ASSET_OPTIMIZATION_PLAN.md B0-1/B0-2/B0-3）；
    # 独立运行时不触发上面的角色对齐流程，只读 workspace 既有产物并写新文件：
    python assets/character_role_aligner.py \
        --export-i2i-bundles <workspace 根目录（含 assets/、captions/ 等）> \
        [--i2i-output-root <导出根目录，默认同 workspace>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("character_role_aligner")

SCHEMA_VERSION = "character-role-alignment-v2"

DECISION_FACTOR = {
    "keep_primary": 1.0,
    "keep_supporting": 0.85,
    "keep_context": 0.5,
}

# Tracklet ids end in a per-shot fragment suffix (``..._T01``); the shot id is
# what remains after stripping it.
TRACKLET_SHOT_SUFFIX = re.compile(r"_T\d+$")


# ---------------------------------------------------------------------------
# 输入加载
# ---------------------------------------------------------------------------

def load_tracklet_alignments(view_dir: Path) -> Dict[str, dict]:
    """从 Stage1.5 的 stage3 视图签名中收集 tracklet_id → semantic_alignment。"""
    out: Dict[str, dict] = {}
    for sig_path in sorted(view_dir.glob("shots/*/tracklet_signatures.json")):
        payload = json.loads(sig_path.read_text(encoding="utf-8"))
        for item in payload.get("tracklet_signatures", []):
            tid = str(item.get("tracklet_id"))
            align = item.get("semantic_alignment")
            if tid and align:
                out[tid] = align
    return out


def load_identity(identity_dir: Path) -> Tuple[List[dict], List[dict], dict]:
    registry_path = identity_dir / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    characters = registry.get("characters", [])
    pending = registry.get("pending_candidates", [])
    return characters, pending, registry


# ---------------------------------------------------------------------------
# 投票
# ---------------------------------------------------------------------------

def vote_chr(tracklet_ids: List[str], alignments: Dict[str, dict],
             min_share: float, min_weight: float,
             contest_margin: float) -> Dict[str, dict]:
    """对单个 CHR 按 scope 投票。返回 {scope: {...}}。

    票重先按 (scope, shot) 局部聚合：同一 shot 内同一 subject 只计最大单票
    （轨迹碎片不会让同一段画面重复计票），scope 总票为各 shot 票之和。
    """
    shot_best: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    detail_rows: List[dict] = []
    for tid in tracklet_ids:
        align = alignments.get(tid)
        shot_id = TRACKLET_SHOT_SUFFIX.sub("", tid)
        if not align:
            detail_rows.append({"tracklet_id": tid, "shot_id": shot_id, "counted": False,
                                "reason": "no_semantic_alignment"})
            continue
        subject_id = str(align.get("subject_id") or "unknown")
        decision = str(align.get("decision") or "")
        entity_type = str(align.get("entity_type") or "")
        scope = str(align.get("semantic_scope_id") or "unscoped")
        factor = DECISION_FACTOR.get(decision, 0.0)
        try:
            confidence = float(align.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        weight = round(confidence * factor, 6)
        counted = (
            entity_type == "person"
            and subject_id not in ("", "unknown")
            and weight > 0.0
        )
        if counted:
            current = shot_best[(scope, shot_id)].get(subject_id, 0.0)
            if weight > current:
                shot_best[(scope, shot_id)][subject_id] = weight
        detail_rows.append({
            "tracklet_id": tid, "shot_id": shot_id, "counted": counted, "scope": scope,
            "subject_id": subject_id, "decision": decision,
            "entity_type": entity_type, "confidence": confidence,
            "weight": weight,
        })

    votes: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    shot_rows: List[dict] = []
    for (scope, shot_id), table in sorted(shot_best.items()):
        for subject_id, shot_weight in table.items():
            votes[scope][subject_id] += shot_weight
        shot_rows.append({"scope": scope, "shot_id": shot_id,
                          "votes": {subject: round(value, 6)
                                    for subject, value in
                                    sorted(table.items(), key=lambda kv: -kv[1])}})

    result: Dict[str, dict] = {}
    for scope, table in votes.items():
        total = sum(table.values())
        ranked = sorted(table.items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_w = ranked[0]
        share = top_w / total if total > 0 else 0.0
        runner_share = ranked[1][1] / total if len(ranked) > 1 and total > 0 else 0.0
        if len(ranked) > 1 and (share - runner_share) < contest_margin:
            status = "contested"
        elif share >= min_share and total >= min_weight:
            status = "confirmed"
        else:
            status = "weak"
        result[scope] = {
            "subject_id": top_id, "status": status,
            "vote_weight": round(top_w, 6), "vote_share": round(share, 4),
            "runner_up": {"subject_id": ranked[1][0],
                          "vote_share": round(runner_share, 4)} if len(ranked) > 1 else None,
            "all_votes": {k: round(v, 6) for k, v in ranked},
        }
    for row in detail_rows:
        scope = row.get("scope") or "unscoped"
        result.setdefault(scope, {
            "subject_id": None, "status": "unresolved",
            "vote_weight": 0.0, "vote_share": 0.0,
            "runner_up": None, "all_votes": {},
        })
    if not result:
        result["unscoped"] = {
            "subject_id": None, "status": "unresolved",
            "vote_weight": 0.0, "vote_share": 0.0,
            "runner_up": None, "all_votes": {},
        }
    return {"by_scope": result, "tracklet_votes": detail_rows, "shot_votes": shot_rows}


# ---------------------------------------------------------------------------
# 物化统一角色目录
# ---------------------------------------------------------------------------

def materialize_role(out_dir: Path, scope: str, subject_id: str, chr_id: str,
                     asset_bank: Optional[Path], voice_export: Optional[Path],
                     vote_info: dict,
                     admission_tier: Optional[str] = None,
                     related_visual_characters: Optional[list] = None) -> dict:
    role_dir = out_dir / "characters" / scope / subject_id
    role_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "semantic_scope_id": scope,
        "subject_id": subject_id,
        "visual_character_id": chr_id,
        "alignment": vote_info,
        "visual_assets": [],
        "voice_assets": [],
    }
    # P1-3 低频准入层级 / P0-3 兼容的附加独立画廊引用：均为可选标注，
    # 为 None 时不写键，保持默认路径产物不变。
    if admission_tier is not None:
        summary["admission_tier"] = admission_tier
    if related_visual_characters is not None:
        summary["related_visual_characters"] = related_visual_characters
    if asset_bank:
        src = asset_bank / "characters" / chr_id
        if src.is_dir():
            dst = role_dir / "visual"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            summary["visual_assets"] = sorted(
                str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()
            )
        else:
            LOGGER.warning("asset bank 中不存在 %s，跳过视觉物化", src)
    if voice_export:
        dst = role_dir / "voice"
        # 防重跑残留：无论本轮能否找到获批语音，先清空旧产物。
        shutil.rmtree(dst, ignore_errors=True)
        manifest = voice_export / "voice_asset_manifest.jsonl"
        copied: List[str] = []
        if manifest.is_file():
            seen: set = set()
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                link = row.get("voice_link") or {}
                # preview 不会自动晋升为正式语音资产：必须人工复核 approved、与
                # consolidated registry 的连接为 matched，且 scope/S-ID 都对得上。
                if row.get("review_status") != "approved":
                    continue
                if link.get("status") != "matched":
                    continue
                if f"global_{int(row.get('global_id') or 0):04d}" != scope:
                    continue
                if str(row.get("asr_speaker_id")) != subject_id:
                    continue
                for source_path in link.get("reference_audio") or []:
                    if source_path in seen:
                        continue
                    seen.add(source_path)
                    src_file = Path(source_path)
                    if not src_file.is_file():
                        LOGGER.warning("获批语音资产不存在，跳过: %s", src_file)
                        continue
                    dst.mkdir(parents=True, exist_ok=True)
                    target = dst / f"{row.get('voice_speaker_id') or link.get('speaker_id') or 'voice'}_{src_file.name}"
                    shutil.copy2(src_file, target)
                    copied.append(target.name)
            if not copied:
                summary["voice_skipped_reason"] = "no_approved_matched_rows"
        else:
            summary["voice_skipped_reason"] = "voice_asset_manifest_missing"
        summary["voice_assets"] = sorted(copied)
    (role_dir / "role_character.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# P0-3 多 CHR 认领的视觉证据仲裁（可选；--multi-chr-gallery-review 启用）
# ---------------------------------------------------------------------------
#
# 同一 (scope, S-ID) 被多个 CHR 认领时，票重仲裁只决定物化 winner，落选者
# 与 winner 的视觉关系此前没有任何记录。本功能用 Stage3 已算好的人脸
# embedding（<identity>/tracklet_features.json）对「winner × 落选者」的
# tracklet 全配对做画廊相似度复核，把结论写回 id_conflicts 条目；判定为
# compatible 的落选 CHR 作为同一角色的附加独立画廊引用登记
# （role_gallery_refs / role_character.json 的 related_visual_characters），
# 不复制/合并其资产，不改变 winner 与 role_to_chr 语义。

# 已注册语义角色 ID（S1/S2/...）；unknown/new_subject_candidate 不匹配。
REGISTERED_SUBJECT_ID = re.compile(r"^S\d+$")

_VISUAL_PERSON_DIR = Path(__file__).resolve().parent / "visual_person"


def _load_visual_builder():
    """懒加载 visual_character_registry_builder（其顶层 import cv2/numpy）。

    只在开启 --multi-chr-gallery-review 时调用，保持基础路径（含 I2I 导出
    模式）不引入 cv2/numpy 依赖。
    """
    import importlib
    import sys
    sys.path.insert(0, str(_VISUAL_PERSON_DIR))
    return importlib.import_module("visual_character_registry_builder")


def _load_identity_tracklets(identity_dir: Path) -> Dict[str, dict]:
    """从 <identity_dir>/tracklet_features.json 重建 tracklet_id → 轻量
    tracklet（只含 face_features，元素为 vcrb.FaceFeature，供
    gallery_similarity 复用）；文件缺失或条目无特征时按无特征处理。"""
    from types import SimpleNamespace
    vcrb = _load_visual_builder()
    features_path = identity_dir / "tracklet_features.json"
    tracklets: Dict[str, Any] = {}
    if not features_path.is_file():
        LOGGER.warning("未找到 %s，画廊复核涉及的 tracklet 均视为无特征", features_path)
        return tracklets
    payload = json.loads(features_path.read_text(encoding="utf-8"))
    for item in payload.get("tracklets", []):
        tid = str(item.get("tracklet_id"))
        if not tid:
            continue
        face_features = []
        for raw in item.get("face_features") or []:
            embedding = raw.get("embedding")
            if embedding is None:
                continue
            face_features.append(vcrb.FaceFeature(
                asset_path=str(raw.get("asset_path") or ""),
                observation_id=str(raw.get("observation_id") or ""),
                det_score=float(raw.get("det_score") or 0.0),
                face_short_side=int(raw.get("face_short_side") or 0),
                sharpness=float(raw.get("sharpness") or 0.0),
                quality_score=float(raw.get("quality_score") or 0.0),
                quality_tier=str(raw.get("quality_tier") or "rejected"),
                yaw=raw.get("yaw"),
                embedding=list(embedding),
            ))
        tracklets[tid] = SimpleNamespace(face_features=face_features)
    return tracklets


def review_conflict_galleries(id_conflict_claims: Dict[Tuple[str, str], List[str]],
                              characters: List[dict], identity_dir: Path,
                              contradiction_floor: float,
                              compat_min_mean: float) -> Dict[Tuple[str, str], dict]:
    """对每个多 CHR 认领冲突做人脸画廊复核。

    输入 {(scope, sid): [按票重降序的认领 chr_id]}；对 winner（第一位）与
    每个落选 CHR 做 tracklet 全配对 gallery_similarity（selector 为
    _usable_features，即排除 rejected 的人脸），取所有非 None topk_mean 的
    最大值 best_topk_mean，判定：
      * 没有任何可比配对（全 None）      → no_comparable_faces
      * best < contradiction_floor       → contradicted（存在人脸反证，
        保持分离，建议回 Stage1.5 复核身份标签）
      * floor <= best < compat_min_mean  → inconclusive（保持分离）
      * best >= compat_min_mean          → compatible（无反证且达到兼容线，
        记录为同一角色的附加独立画廊引用）

    返回 {(scope, sid): {"winner": chr, "losers": {chr: {...}},
    "compatible": [chr, ...]}}。
    """
    vcrb = _load_visual_builder()
    tracklets = _load_identity_tracklets(identity_dir)
    chr_tracklets = {
        str(record.get("character_id")): [str(t) for t in record.get("tracklet_ids", [])]
        for record in characters
    }

    def _tracklet(tid: str):
        # tracklet_features.json 里没有的 tracklet 视为无特征。
        cached = tracklets.get(tid)
        if cached is not None:
            return cached
        from types import SimpleNamespace
        return SimpleNamespace(face_features=[])

    result: Dict[Tuple[str, str], dict] = {}
    for (scope, sid), claimants in sorted(id_conflict_claims.items()):
        if not claimants:
            continue
        winner = claimants[0]
        losers: Dict[str, dict] = {}
        compatible: List[str] = []
        for loser in claimants[1:]:
            best: Optional[float] = None
            for left_tid in chr_tracklets.get(winner, []):
                for right_tid in chr_tracklets.get(loser, []):
                    metrics = vcrb.gallery_similarity(
                        _tracklet(left_tid), _tracklet(right_tid),
                        feature_selector=vcrb._usable_features)
                    topk_mean = metrics.get("topk_mean")
                    if topk_mean is not None and (best is None or topk_mean > best):
                        best = float(topk_mean)
            if best is None:
                verdict = "no_comparable_faces"
            elif best < contradiction_floor:
                verdict = "contradicted"
            elif best < compat_min_mean:
                verdict = "inconclusive"
            else:
                verdict = "compatible"
                compatible.append(loser)
            losers[loser] = {"best_topk_mean": best, "verdict": verdict}
        result[(scope, sid)] = {"winner": winner, "losers": losers,
                                "compatible": compatible}
    return result


# ---------------------------------------------------------------------------
# P1-3 低频角色分层准入（可选；--allow-low-frequency-admission 启用）
# ---------------------------------------------------------------------------
#
# 票重不足 --min-weight 的 CHR 默认只落得 weak 状态、不进物化竞争，低频
# 配角因此整角色丢失。准入条件刻意从严（全部满足才放行）：weak 且票数
# 达到独立的低门槛、S-ID 为已注册语义角色、CHR 内无身份竞争、无 confirmed
# 认领者也无跨 CHR 认领竞争、且 CHR 有非空 face_gallery 作为可靠视觉资产。
# review/unknown 决策的票 factor=0 不会形成有效 vote_weight，天然进不来。

def _has_reliable_visual_assets(record: Optional[dict]) -> bool:
    """Stage3 registry 人物记录的 face_gallery（slot → 文件列表 dict）中
    存在至少一个非空文件列表。"""
    gallery = (record or {}).get("face_gallery") or {}
    if not isinstance(gallery, dict):
        return False
    return any(bool(files) for files in gallery.values())


def select_low_frequency_roles(chr_votes: Dict[str, dict],
                               role_winner_claims: Dict[Tuple[str, str], str],
                               characters_by_id: Dict[str, dict],
                               min_weight: float) -> Dict[Tuple[str, str], str]:
    """从 weak 投票中筛选可低频准入的 (scope, sid) → chr_id。

    候选条件（全部满足）：
      1. info["status"] == "weak" 且 vote_weight >= min_weight
         （confirmed/contested/unresolved 一律不走此分支）；
      2. subject_id 匹配 ^S\\d+$（已注册语义角色）；
      3. info["runner_up"] is None（CHR 内无身份竞争）；
      4. 该 (scope, sid) 不在 role_winner_claims（无 confirmed 认领者），
         且没有任何其他 CHR 的 by_scope[scope].subject_id == sid 且状态
         非 unresolved（无跨 CHR 认领竞争）；
      5. characters_by_id[chr_id] 有可靠视觉资产（face_gallery 非空）。
    """
    # 每个 (scope, sid) 下「状态非 unresolved 的」CHR 集合，用于跨 CHR 竞争检查。
    claims_by_sid: Dict[Tuple[str, str], set] = defaultdict(set)
    for chr_id, vote in chr_votes.items():
        for scope, info in (vote.get("by_scope") or {}).items():
            sid = info.get("subject_id")
            if sid and info.get("status") != "unresolved":
                claims_by_sid[(scope, sid)].add(chr_id)

    selected: Dict[Tuple[str, str], str] = {}
    for chr_id, vote in sorted(chr_votes.items()):
        for scope, info in sorted((vote.get("by_scope") or {}).items()):
            if info.get("status") != "weak":
                continue
            try:
                weight = float(info.get("vote_weight") or 0.0)
            except (TypeError, ValueError):
                continue
            if weight < min_weight:
                continue
            sid = info.get("subject_id")
            if not sid or not REGISTERED_SUBJECT_ID.match(str(sid)):
                continue
            if info.get("runner_up") is not None:
                continue
            key = (scope, str(sid))
            if key in role_winner_claims:
                continue
            if claims_by_sid.get(key, set()) - {chr_id}:
                continue
            if not _has_reliable_visual_assets(characters_by_id.get(chr_id)):
                continue
            selected[key] = chr_id
    return selected


# ---------------------------------------------------------------------------
# I2I 桥接导出（可选；--export-i2i-bundles 启用，默认关闭）
# ---------------------------------------------------------------------------
#
# 规划来源：extract_full_pipeline/I2I_ASSET_OPTIMIZATION_PLAN.md 第三节
# （B0-1 逐参考图 scene–segment–shot–主体索引、B0-2 有序多图输入包、
# B0-3 按目标时刻组装文本）。只读 scene / key_object / segment caption /
# shots / global / role_alignment 的既有产物，写出的全部是新文件：
#
#   <output_root>/assets/scene/scene_segment_index.json   逐 anchor 双向索引
#   <output_root>/assets/i2i_bundles/scene/*.json         场景任务 bundle
#   <output_root>/assets/i2i_bundles/object/*.json        物体任务 bundle
#
# 不修改、不覆盖任何上游产物；不加 flag 时本模块行为与之前完全一致。

I2I_BUNDLE_SCHEMA = "i2i-reference-bundle-v1"
SCENE_SEGMENT_INDEX_SCHEMA = "scene-segment-index-v1"
SCENE_PROMPT_TEMPLATE_VERSION = "scene-v1"
OBJECT_PROMPT_TEMPLATE_VERSION = "object-v1"
# segment 边界判定容差：±半帧（秒值 = 0.5 / fps）。
I2I_FRAME_TOLERANCE = 0.5
# aux anchor 人物面积占比超过该阈值视为携带人物布局信息（purpose=layout）。
I2I_LAYOUT_PERSON_AREA = 0.005


def _i2i_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _i2i_read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _i2i_read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        LOGGER.warning("I2I 导出输入缺失: %s", path)
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _i2i_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _i2i_peek_schema(path: Path) -> Optional[str]:
    try:
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        return json.loads(line).get("schema_version")
            return None
        return _i2i_read_json(path).get("schema_version")
    except (OSError, json.JSONDecodeError):
        return None


def _i2i_image_size(path: Path) -> Optional[Dict[str, int]]:
    """纯标准库读取 JPEG/PNG 尺寸（JPEG SOF / PNG IHDR）。"""
    try:
        with path.open("rb") as fh:
            head = fh.read(2)
            if head == b"\xff\xd8":  # JPEG
                while True:
                    prefix = fh.read(1)
                    if not prefix:
                        return None
                    if prefix != b"\xff":
                        continue
                    marker = fh.read(1)
                    while marker == b"\xff":
                        marker = fh.read(1)
                    if not marker or marker in (b"\xd8", b"\xd9"):
                        continue
                    len_bytes = fh.read(2)
                    if len(len_bytes) < 2:
                        return None
                    seg_len = int.from_bytes(len_bytes, "big")
                    if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        data = fh.read(5)
                        if len(data) < 5:
                            return None
                        return {"height": int.from_bytes(data[1:3], "big"),
                                "width": int.from_bytes(data[3:5], "big")}
                    fh.seek(seg_len - 2, 1)
            if head == b"\x89P":
                rest = fh.read(22)
                if len(rest) == 22 and rest[2:6] == b"\r\n\x1a\n" and rest[10:14] == b"IHDR":
                    return {"width": int.from_bytes(rest[14:18], "big"),
                            "height": int.from_bytes(rest[18:22], "big")}
    except OSError:
        return None
    return None


def _i2i_rel_asset(workspace: Path, asset: Optional[str]) -> Dict[str, Any]:
    """产物绝对路径 → 相对 workspace 的路径 + 存在性校验。"""
    if not asset:
        return {"asset_path": None, "exists": False}
    p = Path(asset)
    exists = p.is_file()
    try:
        rel: Optional[str] = str(p.relative_to(workspace))
    except ValueError:
        rel = str(p)
    return {"asset_path": rel, "exists": exists}


def _i2i_file_descriptor(workspace: Path, path: Optional[Path]) -> dict:
    if path is None:
        return {"path": None, "exists": False}
    desc: Dict[str, Any] = {"exists": path.is_file()}
    try:
        desc["path"] = str(path.relative_to(workspace))
    except ValueError:
        desc["path"] = str(path)
    if desc["exists"]:
        desc["sha256_16"] = _i2i_sha256(path)
        schema = _i2i_peek_schema(path)
        if schema:
            desc["schema_version"] = schema
    return desc


def _i2i_provenance(inputs: Dict[str, Any], paths: List[Optional[Path]],
                    sort_strategy: Optional[str] = None,
                    note: Optional[str] = None) -> dict:
    seen: Dict[str, Path] = {}
    for p in paths:
        if p is not None:
            seen[str(p)] = p
    return {
        "generator": "extract_full_pipeline/assets/character_role_aligner.py --export-i2i-bundles",
        "generated_at": _i2i_now(),
        "sort_strategy": sort_strategy,
        "note": note,
        "inputs": [_i2i_file_descriptor(inputs["workspace"], p) for p in seen.values()],
    }


def _i2i_composite_key(source_video: Optional[str], global_id: int, segment_id: int) -> str:
    """组合键：(source_video, global_id, segment_id)——segment_id 是 clip 局部编号，
    禁止脱离 source_video/global 单独使用。"""
    return f"{source_video or 'unknown_source'}::global_{global_id:04d}::segment_{segment_id}"


def _i2i_shot_num(shot_id: Any) -> Optional[int]:
    if shot_id is None:
        return None
    m = re.search(r"(\d+)\s*$", str(shot_id))
    return int(m.group(1)) if m else None


def _i2i_match_segment(segments: List[dict], t: float,
                       tol: float) -> Tuple[Optional[dict], str]:
    """clip 本地秒 → segment。strict：start <= t < end（末段含端点）；
    仅浮点越界时进入 ±tol 兜底并如实标注 frame_tolerance。"""
    for i, seg in enumerate(segments):
        rng = seg.get("time_range_sec") or [None, None]
        s, e = rng[0], rng[1]
        if s is None or e is None:
            continue
        if s <= t < e or (i == len(segments) - 1 and s <= t <= e + tol):
            return seg, "strict"
    best: Optional[Tuple[float, dict]] = None
    for seg in segments:
        rng = seg.get("time_range_sec") or [None, None]
        s, e = rng[0], rng[1]
        if s is None or e is None:
            continue
        if s - tol <= t < e + tol:
            dist = 0.0 if s <= t <= e else min(abs(t - s), abs(t - e))
            if best is None or dist < best[0]:
                best = (dist, seg)
    if best is not None:
        return best[1], "frame_tolerance"
    return None, "unmatched"


def _i2i_normalize_ost_window(rng: Optional[list], seg_s: float, seg_e: float,
                              tol: float) -> Tuple[Optional[list], str]:
    """on_screen_text 时间窗在产物中坐标系不统一（clip 本地 / segment 本地混用），
    按取值范围推断并归一到 clip 本地秒；无法归一时如实标注，不进 prompt。"""
    if not rng or len(rng) < 2 or rng[0] is None or rng[1] is None:
        return None, "missing"
    s, e = float(rng[0]), float(rng[1])
    if seg_s - tol <= s <= seg_e + tol and e > s:
        return [s, e], "clip_local"
    if -tol <= s and e <= (seg_e - seg_s) + tol:
        return [seg_s + s, seg_s + e], "segment_local_assumed"
    return [s, e], "unresolved"


def load_i2i_inputs(workspace: Path) -> Dict[str, Any]:
    """装载桥接所需的全部上游产物（只读）。"""
    inputs: Dict[str, Any] = {"workspace": workspace, "paths": {}}

    shots: Dict[int, dict] = {}
    shot_files = []
    for rel in ("shots/shots.jsonl", "middle/shots.jsonl"):
        path = workspace / rel
        if path.is_file():
            shot_files.append(path)
            for row in _i2i_read_jsonl(path):
                sid = _i2i_shot_num(row.get("shot_id"))
                if sid is not None:
                    shots.setdefault(sid, row)
    inputs["shots"] = shots
    inputs["paths"]["shots_jsonl"] = shot_files

    clips_path = workspace / "global/global_clips.jsonl"
    clips = {}
    for row in _i2i_read_jsonl(clips_path):
        if row.get("global_id") is not None:
            clips[int(row["global_id"])] = row
    inputs["global_clips"] = clips
    inputs["paths"]["global_clips"] = clips_path

    seg_path = workspace / "captions/segment/segment_captions.jsonl"
    segment_blocks: List[dict] = []
    for row in _i2i_read_jsonl(seg_path):
        gid = int(row["global_id"])
        segs = (row.get("segment_caption") or {}).get("segments") or []
        segs = sorted(segs, key=lambda s: (s.get("time_range_sec") or [0.0])[0])
        segment_blocks.append({
            "global_id": gid,
            "source_video": row.get("source_video"),
            "clip_path": row.get("clip_path"),
            "shot_ids": row.get("shot_ids") or [],
            "segments": segs,
        })
    inputs["segment_blocks"] = segment_blocks
    inputs["paths"]["segment_captions"] = seg_path

    scene_dirs: Dict[int, dict] = {}
    scene_index_paths: Dict[int, Path] = {}
    scene_meta_paths: Dict[Tuple[int, str], Path] = {}
    for gdir in sorted((workspace / "assets/scene").glob("*_global_*")):
        if not gdir.is_dir():
            continue
        m = re.search(r"global_(\d+)$", gdir.name)
        if not m:
            continue
        gid = int(m.group(1))
        index_path = gdir / "scene_index.json"
        index = _i2i_read_json(index_path) if index_path.is_file() else {}
        if index_path.is_file():
            scene_index_paths[gid] = index_path
        scenes: Dict[str, dict] = {}
        for meta_path in sorted(gdir.glob("scene_*/scene_meta.json")):
            scenes[meta_path.parent.name] = {
                "meta": _i2i_read_json(meta_path),
                "dir": meta_path.parent,
            }
            scene_meta_paths[(gid, meta_path.parent.name)] = meta_path
        scene_dirs[gid] = {"dir": gdir, "index": index, "scenes": scenes}
    inputs["scene"] = scene_dirs
    inputs["paths"]["scene_index"] = scene_index_paths
    inputs["paths"]["scene_meta"] = scene_meta_paths

    queue_path = workspace / "assets/key_object/grounding_queue.jsonl"
    inputs["object_queue"] = _i2i_read_jsonl(queue_path)
    inputs["paths"]["grounding_queue"] = queue_path

    obs_path = workspace / "assets/key_object/grounding/object_observations.jsonl"
    obs_by_cand: Dict[str, List[dict]] = defaultdict(list)
    for row in _i2i_read_jsonl(obs_path):
        obs_by_cand[str(row.get("candidate_id"))].append(row)
    for rows in obs_by_cand.values():
        rows.sort(key=lambda r: (r.get("frame_index") is None,
                                 r.get("frame_index") or 0))
    inputs["object_observations"] = obs_by_cand
    inputs["paths"]["object_observations"] = obs_path

    ra_path = workspace / "assets/role_alignment.json"
    inputs["role_alignment"] = _i2i_read_json(ra_path) if ra_path.is_file() else {}
    inputs["paths"]["role_alignment"] = ra_path
    return inputs


def build_scene_segment_index(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """B0-1：每张 anchor 独立关联 segment / shot_ids / key_subject_ids，
    输出 scene→segments 与 segment→scenes 双向映射，标注未覆盖 segment 与
    跨 segment 的 scene。"""
    workspace: Path = inputs["workspace"]
    globals_out: List[dict] = []
    seg_to_scenes: Dict[str, List[str]] = {}
    scene_to_segments: Dict[str, List[str]] = {}
    uncovered_total: List[dict] = []
    cross_total: List[dict] = []

    for gblock in inputs["segment_blocks"]:
        gid = gblock["global_id"]
        scope = f"global_{gid:04d}"
        clip = inputs["global_clips"].get(gid) or {}
        source_video = gblock.get("source_video") or clip.get("source_video")
        scene_block = inputs["scene"].get(gid) or {"dir": None, "index": {}, "scenes": {}}
        video_info = scene_block["index"].get("video") or {}
        fps = video_info.get("fps")
        if not fps:  # 兜底：该 global 任一 shot 的 fps
            for shot_id in gblock.get("shot_ids") or []:
                shot = inputs["shots"].get(int(shot_id))
                if shot and shot.get("fps"):
                    fps = shot["fps"]
                    break
        tol = I2I_FRAME_TOLERANCE / float(fps) if fps else 0.021
        segments = gblock["segments"]

        scene_entries: List[dict] = []
        for scene_name in sorted(scene_block["scenes"]):
            sdir: Path = scene_block["scenes"][scene_name]["dir"]
            meta = scene_block["scenes"][scene_name]["meta"]
            anchors: List[dict] = []
            seg_ids: List[int] = []
            for sel in sorted(meta.get("selected") or [], key=lambda r: r.get("rank") or 0):
                t = float(sel.get("timestamp_sec"))
                seg, how = _i2i_match_segment(segments, t, tol)
                asset = _i2i_rel_asset(workspace, str(sdir / str(sel.get("asset"))))
                if seg is not None:
                    sid = int(seg.get("segment_id"))
                    if sid not in seg_ids:
                        seg_ids.append(sid)
                    seg_rng = seg.get("time_range_sec")
                    shot_ids = seg.get("shot_ids") or []
                    subjects = seg.get("key_subject_ids") or []
                    key = _i2i_composite_key(source_video, gid, sid)
                else:
                    sid, seg_rng, shot_ids, subjects, key = None, None, [], [], None
                anchors.append({
                    "rank": sel.get("rank"),
                    "asset": sel.get("asset"),
                    "asset_path": asset["asset_path"],
                    "asset_exists": asset["exists"],
                    "timestamp_sec": t,
                    "time_coordinate": "global-clip 本地秒",
                    "frame_index": sel.get("frame_index"),
                    "segment_id": sid,
                    "segment_match": how,
                    "segment_composite_key": key,
                    "segment_time_range_sec": seg_rng,
                    "shot_ids": shot_ids,
                    "key_subject_ids": subjects,
                    "evidence": {
                        "dino_similarity_to_main": sel.get("dino_similarity_to_main"),
                        "person_area_ratio": sel.get("person_area_ratio"),
                        "total_score": sel.get("total_score"),
                    },
                })
            scene_entries.append({
                "scene_id": scene_name,
                "scene_dir": str(sdir.relative_to(workspace)),
                "segment_ids": sorted(seg_ids),
                "spans_multiple_segments": len(seg_ids) > 1,
                "anchors": anchors,
            })

        seg_entries: List[dict] = []
        for seg in segments:
            sid = int(seg.get("segment_id"))
            key = _i2i_composite_key(source_video, gid, sid)
            covering = [e["scene_id"] for e in scene_entries if sid in e["segment_ids"]]
            anchor_count = sum(1 for e in scene_entries for a in e["anchors"]
                               if a["segment_id"] == sid)
            entry: Dict[str, Any] = {
                "composite_key": key,
                "source_video": source_video,
                "global_id": gid,
                "segment_id": sid,
                "time_range_sec": seg.get("time_range_sec"),
                "source_time_range_sec": seg.get("source_time_range_sec"),
                "shot_ids": seg.get("shot_ids") or [],
                "key_subject_ids": seg.get("key_subject_ids") or [],
                "covered": bool(covering),
                "anchor_count": anchor_count,
                "covering_scenes": covering,
            }
            if not covering:
                entry["uncovered_reason"] = (
                    "该 segment 时间窗内没有任何 scene anchor（scene 提取未覆盖）；"
                    "需补采/补选 scene 后才能生成有效场景 bundle")
                uncovered_total.append({
                    "composite_key": key, "global_id": gid, "segment_id": sid,
                    "time_range_sec": seg.get("time_range_sec"),
                    "reason": entry["uncovered_reason"],
                })
            seg_to_scenes[key] = covering
            seg_entries.append(entry)

        for e in scene_entries:
            skey = f"{scope}/{e['scene_id']}"
            keys = [_i2i_composite_key(source_video, gid, s) for s in e["segment_ids"]]
            scene_to_segments[skey] = keys
            if e["spans_multiple_segments"]:
                cross_total.append({
                    "scene": skey, "segment_ids": e["segment_ids"],
                    "segment_composite_keys": keys,
                    "note": "anchors 落在多个 segment；跨 segment 借用仅复用环境/布局信息",
                })

        globals_out.append({
            "global_id": gid,
            "scope": scope,
            "video_id": scene_block["dir"].name if scene_block["dir"] is not None else None,
            "source_video": source_video,
            "global_clip": clip.get("clip_path"),
            "global_time_range_sec": clip.get("time_range_sec"),
            "fps": fps,
            "frame_tolerance_sec": round(tol, 6),
            "video": {k: video_info.get(k) for k in ("width", "height")},
            "segments": seg_entries,
            "scenes": scene_entries,
        })

    prov_paths: List[Optional[Path]] = [
        inputs["paths"].get("segment_captions"),
        inputs["paths"].get("global_clips"),
        inputs["paths"].get("role_alignment"),
    ]
    prov_paths += inputs["paths"].get("shots_jsonl") or []
    prov_paths += list((inputs["paths"].get("scene_index") or {}).values())
    prov_paths += list((inputs["paths"].get("scene_meta") or {}).values())
    return {
        "schema_version": SCENE_SEGMENT_INDEX_SCHEMA,
        "generator": "character_role_aligner.py --export-i2i-bundles",
        "generated_at": _i2i_now(),
        "workspace": str(workspace),
        "coordinate_system": "global-clip 本地秒（与 scene_meta.selected[].timestamp_sec "
                             "及 segment.time_range_sec 同系）",
        "composite_key_format": "{source_video}::global_{NNNN}::segment_{N}",
        "globals": globals_out,
        "segment_to_scenes": seg_to_scenes,
        "scene_to_segments": scene_to_segments,
        "uncovered_segments": uncovered_total,
        "cross_segment_scenes": cross_total,
        "summary": {
            "global_count": len(globals_out),
            "segment_count": sum(len(g["segments"]) for g in globals_out),
            "covered_segment_count": sum(1 for g in globals_out
                                         for s in g["segments"] if s["covered"]),
            "uncovered_segment_count": len(uncovered_total),
            "scene_count": sum(len(g["scenes"]) for g in globals_out),
            "cross_segment_scene_count": len(cross_total),
        },
        "provenance": _i2i_provenance(inputs, prov_paths),
    }


def _i2i_subject_details(seg: dict, scope: str, role_to_chr: Dict[str, str]) -> List[dict]:
    """segment 级外观线索 + source_shots 级 screen_position（经时间索引确认），
    S-ID→CHR 仅查本 scope，映射不到如实标 missing。"""
    obs_by_sid = {str(o.get("subject_id")): o
                  for o in seg.get("semantic_role_observations") or []}
    positions: Dict[str, set] = defaultdict(set)
    for shot in seg.get("source_shots") or []:
        caption = shot.get("shot_caption") or {}
        for o in caption.get("semantic_role_observations") or []:
            if o.get("screen_position") and o.get("subject_id"):
                positions[str(o["subject_id"])].add(str(o["screen_position"]))
    out = []
    for sid in seg.get("key_subject_ids") or []:
        sid = str(sid)
        scoped = f"{scope}:{sid}"
        chr_id = role_to_chr.get(scoped)
        obs = obs_by_sid.get(sid) or {}
        out.append({
            "subject_id": sid,
            "scope": scope,
            "scoped_id": scoped,
            "chr_id": chr_id,
            "mapping_status": "mapped" if chr_id else "missing",
            "visual_identity_cues": obs.get("visual_identity_cues") or [],
            "screen_positions": sorted(positions.get(sid, [])),
            "appearance_source": "segment semantic_role_observations + source_shots",
        })
    return out


def _i2i_segment_queue_objects(seg: dict, inputs: Dict[str, Any],
                               clip_start: float, tol: float) -> List[dict]:
    """segment.key_object_candidates 为空时，从 grounding_queue 按时间窗补接
    （shot 所属 global 的 clip 本地坐标），并注明来源。"""
    rng = seg.get("time_range_sec") or [None, None]
    if rng[0] is None:
        return []
    seg_s, seg_e = float(rng[0]), float(rng[1])
    shot_set = {int(s) for s in seg.get("shot_ids") or []}
    out = []
    for cand in inputs["object_queue"]:
        shot_num = _i2i_shot_num(cand.get("shot_id"))
        if shot_num is None or shot_num not in shot_set:
            continue
        shot = inputs["shots"].get(shot_num)
        if not shot:
            continue
        shot_local_start = float(shot["time_range_sec"][0]) - clip_start
        vr = cand.get("visible_time_range_sec") or [0.0, 0.0]
        win = [shot_local_start + float(vr[0]), shot_local_start + float(vr[1])]
        if win[1] < seg_s - tol or win[0] > seg_e + tol:
            continue
        out.append({
            "candidate_id": cand.get("candidate_id"),
            "canonical_name": cand.get("canonical_name"),
            "observed_state": cand.get("observed_state"),
            "possible_states": cand.get("possible_states") or [],
            "shot_id": cand.get("shot_id"),
            "clip_local_visible_time_range_sec": [round(win[0], 6), round(win[1], 6)],
            "verification_status": cand.get("verification_status"),
            "source": "grounding_queue",
        })
    out.sort(key=lambda c: (c["clip_local_visible_time_range_sec"][0],
                            str(c["candidate_id"])))
    return out


def render_scene_prompt(scene_description: Optional[str], description: Optional[str],
                        subjects: List[dict], objects: List[dict],
                        ost_in_target: List[dict], dialogue: List[dict],
                        n_refs: int) -> str:
    """template_version=scene-v1；缺字段写“未知”，不编造。"""
    head = "还原目标时刻的场景，遵循参考图 1 的构图与比例。"
    if n_refs > 1:
        head += f"参考图 2～{n_refs} 分别补充所标注的环境、人物布局或身份信息。"
    else:
        head += "本任务仅 1 张参考图。"
    subj_parts = []
    for s in subjects:
        chr_txt = s["chr_id"] or "未映射"
        cues = "、".join(s["visual_identity_cues"]) if s["visual_identity_cues"] else "外观未知"
        pos = "、".join(s["screen_positions"]) if s["screen_positions"] else "位置未知"
        subj_parts.append(f"{s['subject_id']}→{chr_txt}（{cues}；位置：{pos}）")
    subj_line = "；".join(subj_parts) if subj_parts else "无（该 segment 未标注关键主体）"
    obj_parts = [f"{o.get('canonical_name') or '未知物体'}"
                 f"（状态：{o.get('observed_state') or '未知'}）" for o in objects]
    obj_line = ("；".join(obj_parts) if obj_parts
                else "无（grounding_queue 中无落在本时间窗的候选）")
    ost_parts = [f"「{e.get('text')}」（{e.get('position') or '位置未知'}）"
                 for e in ost_in_target]
    ost_line = "；".join(ost_parts) if ost_parts else "无"
    dlg_parts = [f"[{d.get('speaker_id') or 'unknown'}] {d.get('text')}"
                 for d in dialogue if d.get("text")]
    dlg_line = "；".join(dlg_parts) if dlg_parts else "无"
    return "\n".join([
        head,
        f"环境：{scene_description or '未知'}",
        f"此刻主体及位置：{subj_line}",
        f"此刻物体及状态：{obj_line}",
        f"屏幕文字：{ost_line}",
        f"剧情背景：{description or '未知'}",
        f"对话背景：{dlg_line}",
    ])


def render_object_prompt(canonical_name: Optional[str], visual_attributes: dict,
                         observed_state: Optional[str], surface_text: Optional[str],
                         n_refs: int) -> str:
    """template_version=object-v1；缺字段写“未知”，无可见文字写“无”。"""
    name = canonical_name or "未知物体"
    if n_refs > 1:
        head = f"还原 {name}，以参考图 1 的目标视角为准，其余参考补充已标注的形状和细节。"
    else:
        head = f"还原 {name}，以参考图 1 为准（本任务仅 1 张参考图）。"
    attr_line = ("，".join(f"{k}={v}" for k, v in visual_attributes.items())
                 if visual_attributes else "未知")
    return "\n".join([
        head,
        f"外观属性：{attr_line}",
        f"目标状态：{observed_state or '未知'}",
        f"文字或标识：{surface_text or '无'}",
    ])


def build_scene_bundle(inputs: Dict[str, Any], index: Dict[str, Any],
                       global_id: int, segment_id: int) -> dict:
    workspace: Path = inputs["workspace"]
    gidx = next(g for g in index["globals"] if g["global_id"] == global_id)
    seg_entry = next(s for s in gidx["segments"] if s["segment_id"] == segment_id)
    gblock = next(b for b in inputs["segment_blocks"] if b["global_id"] == global_id)
    seg = next(s for s in gblock["segments"] if int(s.get("segment_id")) == segment_id)
    scope = gidx["scope"]
    tol = gidx["frame_tolerance_sec"]
    seg_rng = seg.get("time_range_sec") or [None, None]
    seg_s, seg_e = float(seg_rng[0]), float(seg_rng[1])
    clip = inputs["global_clips"].get(global_id) or {}
    clip_start = float((clip.get("time_range_sec") or [0.0])[0])
    role_to_chr = inputs["role_alignment"].get("role_to_chr") or {}

    covering = seg_entry["covering_scenes"]
    scene_lookup = {e["scene_id"]: e for e in gidx["scenes"]}
    ordered_scenes = sorted(
        covering,
        key=lambda n: min(a["timestamp_sec"] for a in scene_lookup[n]["anchors"]))

    refs: List[dict] = []
    for sname in ordered_scenes:
        for a in scene_lookup[sname]["anchors"]:
            cross = a["segment_id"] not in (None, segment_id)
            if a["rank"] == 1:
                purpose = "environment+layout"
                purpose_note = "主参考：构图与比例以其为准"
            elif (a["evidence"].get("person_area_ratio") or 0.0) > I2I_LAYOUT_PERSON_AREA:
                purpose = "layout"
                purpose_note = "含可见人物，补充人物布局信息"
            else:
                purpose = "environment"
                purpose_note = "补充环境/视角信息"
            if cross:
                purpose_note += "；该 anchor 时间落在相邻 segment，仅复用其环境/布局信息"
            size = None
            if a["asset_exists"] and a["asset_path"]:
                size = _i2i_image_size(workspace / a["asset_path"])
            refs.append({
                "ref_id": None, "rank": None,  # 排序后统一编号，与 prompt 图号一致
                "asset_path": a["asset_path"],
                "asset_exists": a["asset_exists"],
                "purpose": purpose,
                "purpose_note": purpose_note,
                "source_frame": a["frame_index"],
                "source_time": a["timestamp_sec"],
                "time_coordinate": "global-clip 本地秒",
                "scene_id": sname,
                "scene_rank": a["rank"],
                "geometry": {**(size or {"width": None, "height": None}),
                             "coordinate_system": "图像像素"},
                "evidence": {**a["evidence"],
                             "anchor_segment_id": a["segment_id"],
                             "cross_segment": cross,
                             "segment_match": a["segment_match"]},
            })
    for i, r in enumerate(refs, 1):
        r["ref_id"] = f"ref_{i}"
        r["rank"] = i

    moments = sorted(r["source_time"] for r in refs if not r["evidence"]["cross_segment"])

    ost_entries: List[dict] = []
    for item in seg.get("on_screen_text") or []:
        norm, coord = _i2i_normalize_ost_window(item.get("time_range_sec"),
                                                seg_s, seg_e, tol)
        in_target = bool(norm and moments
                         and norm[1] >= moments[0] - tol
                         and norm[0] <= moments[-1] + tol)
        ost_entries.append({
            "text": item.get("text"),
            "position": item.get("position"),
            "language": item.get("language"),
            "time_range_sec_raw": item.get("time_range_sec"),
            "time_range_sec_clip_local": norm if coord != "unresolved" else None,
            "coordinate_system": coord,
            "in_target_window": in_target,
        })
    ost_in_target = [e for e in ost_entries if e["in_target_window"]]

    dialogue = [{"speaker_id": d.get("speaker_id"), "text": d.get("text"),
                 "time_range_sec": d.get("time_range_sec"),
                 "is_off_screen": d.get("is_off_screen")}
                for d in seg.get("dialogue") or []]
    subjects = _i2i_subject_details(seg, scope, role_to_chr)

    queue_objects = _i2i_segment_queue_objects(seg, inputs, clip_start, tol)
    if seg.get("key_object_candidates"):
        object_entries = [dict(o, source="segment") if isinstance(o, dict)
                          else {"name": o, "source": "segment"}
                          for o in seg["key_object_candidates"]]
        obj_source = "segment"
    else:
        object_entries = queue_objects
        obj_source = "grounding_queue" if queue_objects else None

    review_assets: List[dict] = []
    for sname in ordered_scenes:
        sdir = workspace / scene_lookup[sname]["scene_dir"]
        for fname, note in (
            ("topk_preview.jpg", "候选帧拼图预览，含分数标注，仅供人工审阅，禁止作为 I2I 参考"),
            ("person_mask_main.png", "主帧人物区域检测框 union 代理 mask（非 SAM 单实例 mask），仅供审阅"),
        ):
            p = sdir / fname
            if p.is_file():
                review_assets.append({"asset_path": str(p.relative_to(workspace)),
                                      "scene_id": sname, "kind": fname, "note": note})

    if refs:
        rendered = render_scene_prompt(
            seg.get("scene_description"), seg.get("description"),
            subjects, object_entries, ost_in_target, dialogue, len(refs))
        prompt = {
            "template_version": SCENE_PROMPT_TEMPLATE_VERSION,
            "rendered_text": rendered,
            "reference_bindings": [
                {"ref_id": r["ref_id"], "figure_no": r["rank"],
                 "asset_path": r["asset_path"], "purpose": r["purpose"],
                 "scene_id": r["scene_id"]} for r in refs],
        }
    else:
        prompt = {
            "template_version": SCENE_PROMPT_TEMPLATE_VERSION,
            "rendered_text": None,
            "skipped_reason": "无参考图：segment 未被 scene 覆盖，补齐 scene 后重渲染",
            "reference_bindings": [],
        }

    missing_assets = [r["asset_path"] for r in refs if not r["asset_exists"]]
    missing_fields = [name for name, val in
                      (("description", seg.get("description")),
                       ("scene_description", seg.get("scene_description"))) if not val]
    missing_entities = [s["scoped_id"] for s in subjects
                        if s["mapping_status"] == "missing"]
    reasons: List[str] = []
    if not covering:
        reasons.append("segment 未被任何 scene 覆盖"
                       "（见 scene_segment_index.json 的 uncovered_segments），"
                       "不以重复图凑数")
    if missing_assets:
        reasons.append(f"{len(missing_assets)} 个 anchor 文件缺失")
    ready = bool(refs) and not missing_assets
    completeness = {
        "required_image_count": len(refs) if refs else 1,
        "required_policy": "所需 = 覆盖本 segment 的 scene 的全部 anchor（至少 1 张主参考）",
        "actual_image_count": len(refs) - len(missing_assets),
        "missing_assets": missing_assets,
        "missing_fields": missing_fields,
        "missing_entities": missing_entities,
        "ready": ready,
        "reasons": reasons,
    }

    prov_paths: List[Optional[Path]] = [
        inputs["paths"].get("segment_captions"),
        inputs["paths"].get("global_clips"),
        inputs["paths"].get("role_alignment"),
        inputs["paths"].get("grounding_queue"),
        (inputs["paths"].get("scene_index") or {}).get(global_id),
    ]
    prov_paths += [(inputs["paths"].get("scene_meta") or {}).get((global_id, s))
                   for s in ordered_scenes]
    prov_paths += inputs["paths"].get("shots_jsonl") or []

    return {
        "schema_version": I2I_BUNDLE_SCHEMA,
        "task_id": f"scene_global_{global_id:04d}_segment_{segment_id:03d}",
        "task_type": "scene",
        "target": {
            "source_video": gidx["source_video"],
            "global_id": global_id,
            "scope": scope,
            "segment_id": segment_id,
            "segment_composite_key": seg_entry["composite_key"],
            "segment_time_range_sec": seg.get("time_range_sec"),
            "time_coordinate": "global-clip 本地秒",
            "source_time_range_sec": seg.get("source_time_range_sec"),
            "source_time_coordinate": "电影全局秒",
            "target_moments_sec": moments,
            "shot_ids": seg.get("shot_ids") or [],
            "output_canvas": gidx.get("video"),
        },
        "references": refs,
        "review_assets": review_assets,
        "text_context": {
            "description": seg.get("description"),
            "scene_description": seg.get("scene_description"),
            "segment_time_range_sec": seg.get("time_range_sec"),
            "source_time_range_sec": seg.get("source_time_range_sec"),
            "on_screen_text": ost_entries,
            "on_screen_text_note": "坐标系由导出侧按取值范围推断；"
                                   "仅 in_target_window=true 的条目进入 prompt",
            "dialogue": dialogue,
            "dialogue_note": "仅作上下文，不自动渲染为字幕",
            "key_subject_ids": seg.get("key_subject_ids") or [],
            "key_object_candidates": object_entries,
            "key_object_candidates_source": obj_source,
            "segment_key_object_candidates_raw": seg.get("key_object_candidates") or [],
            "fields_origin": "captions/segment/segment_captions.jsonl"
                             "（segment 级）；queue 补接自 assets/key_object/grounding_queue.jsonl",
        },
        "prompt": prompt,
        "entity_index": {
            "subjects": subjects,
            "objects": object_entries,
            "role_mapping_table": "assets/role_alignment.json::role_to_chr",
            "role_mapping_scope_rule": "仅查本 scope（global_NNNN:S-ID），不跨 scope 借用",
        },
        "completeness": completeness,
        "provenance": _i2i_provenance(
            inputs, prov_paths,
            sort_strategy="覆盖 scene 按主 anchor 时间升序，scene 内按 rank 升序；"
                          "ref 编号与 prompt 图号一致"),
    }


def build_object_bundle(inputs: Dict[str, Any], cand: dict) -> dict:
    workspace: Path = inputs["workspace"]
    cid = str(cand.get("candidate_id"))
    obs_rows = inputs["object_observations"].get(cid, [])
    shot_num = _i2i_shot_num(cand.get("shot_id"))
    shot = inputs["shots"].get(shot_num) if shot_num is not None else None
    gid = None
    if shot_num is not None:
        for g, row in inputs["global_clips"].items():
            if shot_num in (row.get("shot_ids") or []):
                gid = g
                break
    clip = inputs["global_clips"].get(gid) if gid is not None else {}
    clip_start = float((clip.get("time_range_sec") or [0.0])[0]) if clip else 0.0
    scope = f"global_{gid:04d}" if gid is not None else None
    role_to_chr = inputs["role_alignment"].get("role_to_chr") or {}

    vr = cand.get("visible_time_range_sec") or [None, None]
    clip_window = None
    if shot and vr[0] is not None:
        shot_local_start = float(shot["time_range_sec"][0]) - clip_start
        clip_window = [round(shot_local_start + float(vr[0]), 6),
                       round(shot_local_start + float(vr[1]), 6)]

    seg_assoc: List[dict] = []
    nearby_ost: List[dict] = []
    if gid is not None:
        gblock = next((b for b in inputs["segment_blocks"]
                       if b["global_id"] == gid), None)
        if gblock:
            source_video = gblock.get("source_video") or clip.get("source_video")
            for seg in gblock["segments"]:
                if shot_num in (seg.get("shot_ids") or []):
                    seg_assoc.append({
                        "segment_id": seg.get("segment_id"),
                        "composite_key": _i2i_composite_key(
                            source_video, gid, int(seg.get("segment_id"))),
                        "segment_time_range_sec": seg.get("time_range_sec"),
                    })
                    if clip_window:
                        rng = seg.get("time_range_sec") or [None, None]
                        for item in seg.get("on_screen_text") or []:
                            norm, coord = _i2i_normalize_ost_window(
                                item.get("time_range_sec"),
                                float(rng[0]), float(rng[1]), 0.021)
                            if norm and norm[1] >= clip_window[0] \
                                    and norm[0] <= clip_window[1]:
                                nearby_ost.append({
                                    "text": item.get("text"),
                                    "position": item.get("position"),
                                    "time_range_sec_clip_local": norm,
                                    "coordinate_system": coord,
                                })

    refs: List[dict] = []
    review_assets: List[dict] = []
    missing_assets: List[dict] = []
    for i, ob in enumerate(obs_rows, 1):
        assets = ob.get("assets") or {}
        crop = _i2i_rel_asset(workspace, assets.get("crop_path"))
        rgba = _i2i_rel_asset(workspace, assets.get("rgba_path"))
        frame = _i2i_rel_asset(workspace, assets.get("frame_path"))
        mask = _i2i_rel_asset(workspace, assets.get("mask_path"))
        overlay = _i2i_rel_asset(workspace, assets.get("overlay_path"))
        size = (_i2i_image_size(workspace / crop["asset_path"])
                if crop["exists"] and crop["asset_path"] else None)
        bbox = ob.get("mask_bbox_xyxy")
        sel_bbox = ob.get("selected_bbox_xyxy")
        crop_origin = None
        crop_matches_bbox = None
        if size and sel_bbox:
            bw, bh = int(sel_bbox[2] - sel_bbox[0]), int(sel_bbox[3] - sel_bbox[1])
            crop_matches_bbox = (size["width"] == bw and size["height"] == bh)
            if crop_matches_bbox:
                crop_origin = [int(sel_bbox[0]), int(sel_bbox[1])]
        additional = [
            {"asset_path": rgba["asset_path"], "exists": rgba["exists"],
             "kind": "rgba",
             "alpha_semantics": "alpha=255 为 SAM 目标实例区域、alpha=0 为背景；"
                                "作为参考图使用前需按 alpha 抠取或合成到底色，"
                                "直接当 RGB 使用会污染背景"},
            {"asset_path": frame["asset_path"], "exists": frame["exists"],
             "kind": "full_frame"},
            {"asset_path": mask["asset_path"], "exists": mask["exists"],
             "kind": "mask",
             "coordinate_system": "全帧像素，与 frame_path 对齐"},
        ]
        for label, info in (("crop", crop), ("rgba", rgba)):
            if not info["exists"]:
                missing_assets.append({"ref_id": f"ref_{i}", "kind": label,
                                       "asset_path": info["asset_path"]})
        refs.append({
            "ref_id": f"ref_{i}",
            "rank": i,
            "asset_path": crop["asset_path"],
            "asset_kind": "crop",
            "asset_exists": crop["exists"],
            "purpose": "object",
            "source_frame": ob.get("frame_index"),
            "source_time": ob.get("timestamp_sec"),
            "time_coordinate": "shot-clip 本地秒",
            "candidate_id": cid,
            "geometry": {**(size or {"width": None, "height": None}),
                         "crop_bbox_xyxy_frame_pixels": sel_bbox,
                         "mask_bbox_xyxy_frame_pixels": bbox,
                         "crop_origin_xy_frame_pixels": crop_origin,
                         "crop_matches_crop_bbox": crop_matches_bbox,
                         "coordinate_system": "crop_bbox(selected_bbox)/mask_bbox 为全帧像素"
                                              " xyxy；crop 按 crop_bbox 从全帧裁出"},
            "additional_assets": additional,
            "evidence": {
                "mask_area_ratio": ob.get("mask_area_ratio"),
                "sam_score": ob.get("sam_score"),
                "ambiguous_instance": ob.get("ambiguous_instance"),
                "selected_grounding_score": ob.get("selected_grounding_score"),
                "selected_phrase_vote_count": ob.get("selected_phrase_vote_count"),
                "status": ob.get("status"),
            },
        })
        if overlay["exists"]:
            review_assets.append({
                "asset_path": overlay["asset_path"], "kind": "overlay",
                "ref_id": f"ref_{i}",
                "note": "可视化调试图（叠加框/mask），仅供人工审阅，禁止作为 I2I 参考",
            })

    surface_text = (cand.get("object_surface_text") or cand.get("visible_text")
                    or cand.get("text_on_object"))
    attrs = cand.get("visual_attributes") or {}
    n_amb = sum(1 for o in obs_rows if o.get("ambiguous_instance"))
    notes = ["observations 未携带 visual_attributes/observed_state 等字段，"
             "已从 grounding_queue join 取值"]
    if n_amb:
        notes.append(f"{n_amb}/{len(obs_rows)} 帧 ambiguous_instance=true，"
                     "存在实例歧义，使用时以各帧 evidence 为准")
    if not surface_text and nearby_ost:
        notes.append("可见时间窗内存在 on_screen_text 条目，但未确认其位于物体表面，"
                     "未用作「文字或标识」")
    missing_fields = []
    if not attrs:
        missing_fields.append("visual_attributes")
    if not cand.get("observed_state"):
        missing_fields.append("observed_state")

    not_ok = [o.get("frame_index") for o in obs_rows if o.get("status") != "ok"]
    ready = bool(obs_rows) and not missing_assets and not not_ok
    reasons: List[str] = []
    if not obs_rows:
        reasons.append("object_observations.jsonl 中无该候选的观测帧")
    if missing_assets:
        reasons.append(f"{len(missing_assets)} 个参考文件缺失")
    if not_ok:
        reasons.append(f"观测帧 status!=ok: frame_index={not_ok}")
    completeness = {
        "required_image_count": len(obs_rows),
        "required_policy": "所需 = 上游为该候选产出的全部观测帧（本 workspace 均为 5 帧）",
        "actual_image_count": len(refs) - sum(1 for m in missing_assets
                                              if m["kind"] == "crop"),
        "missing_assets": missing_assets,
        "missing_fields": missing_fields,
        "notes": notes,
        "ready": ready,
        "reasons": reasons,
    }

    related = []
    for sid in cand.get("related_semantic_subject_ids") or []:
        sid = str(sid)
        if re.fullmatch(r"S\d+", sid) and scope:
            scoped = f"{scope}:{sid}"
            chr_id = role_to_chr.get(scoped)
            related.append({"subject_id": sid, "scope": scope, "scoped_id": scoped,
                            "chr_id": chr_id,
                            "mapping_status": "mapped" if chr_id else "missing"})
        else:
            related.append({"subject_id": sid, "scope": scope, "scoped_id": None,
                            "chr_id": None,
                            "mapping_status": "unresolvable_non_registry_id"})

    rendered = render_object_prompt(cand.get("canonical_name"), attrs,
                                    cand.get("observed_state"), surface_text,
                                    len(refs))
    prov_paths: List[Optional[Path]] = [
        inputs["paths"].get("grounding_queue"),
        inputs["paths"].get("object_observations"),
        inputs["paths"].get("segment_captions"),
        inputs["paths"].get("role_alignment"),
        inputs["paths"].get("global_clips"),
    ]
    prov_paths += inputs["paths"].get("shots_jsonl") or []
    return {
        "schema_version": I2I_BUNDLE_SCHEMA,
        "task_id": f"object_{cid}",
        "task_type": "object",
        "target": {
            "candidate_id": cid,
            "canonical_name": cand.get("canonical_name"),
            "shot_id": cand.get("shot_id"),
            "global_id": gid,
            "scope": scope,
            "segment_association": seg_assoc,
            "observed_state": cand.get("observed_state"),
            "possible_states": cand.get("possible_states") or [],
            "state_note": "possible_states 为备选状态，不要求同一输出同时呈现",
            "visible_time_range_sec_shot_local": vr,
            "clip_local_visible_time_range_sec": clip_window,
        },
        "references": refs,
        "review_assets": review_assets,
        "text_context": {
            "canonical_name": cand.get("canonical_name"),
            "grounding_phrases": cand.get("grounding_phrases") or [],
            "grounding_phrases_note": "用于提取不重复可见属性，保留原文供追溯；"
                                      "不机械追加为指令",
            "visual_attributes": attrs,
            "possible_states": cand.get("possible_states") or [],
            "observed_state": cand.get("observed_state"),
            "priority": cand.get("priority"),
            "proposal_status": cand.get("proposal_status"),
            "verification_status": cand.get("verification_status"),
            "nearby_on_screen_text": nearby_ost,
            "fields_origin": "assets/key_object/grounding_queue.jsonl"
                             "（候选字段）；assets/key_object/grounding/object_observations.jsonl"
                             "（逐帧 evidence）",
        },
        "prompt": {
            "template_version": OBJECT_PROMPT_TEMPLATE_VERSION,
            "rendered_text": rendered,
            "reference_bindings": [
                {"ref_id": r["ref_id"], "figure_no": r["rank"],
                 "asset_path": r["asset_path"], "purpose": r["purpose"],
                 "candidate_id": cid} for r in refs],
        },
        "entity_index": {
            "candidate": {
                "candidate_id": cid,
                "canonical_name": cand.get("canonical_name"),
                "semantic_scope_id_as_recorded": cand.get("semantic_scope_id"),
                "resolved_scope": scope,
                "scope_resolution": "shot_id → global_clips.jsonl 的 shot_ids",
            },
            "related_subjects": related,
            "segment_association": seg_assoc,
            "role_mapping_table": "assets/role_alignment.json::role_to_chr",
        },
        "completeness": completeness,
        "provenance": _i2i_provenance(
            inputs, prov_paths,
            sort_strategy="observations 按 frame_index 升序；ref_1 为首帧 crop，"
                          "与 prompt 图号一致"),
    }


def export_i2i_bundles(workspace: Path, output_root: Optional[Path] = None) -> dict:
    """桥接导出入口：写 scene_segment_index.json + assets/i2i_bundles/。
    只新增文件，不修改任何上游产物；返回汇总指标。"""
    workspace = workspace.resolve()
    output_root = (output_root or workspace).resolve()
    inputs = load_i2i_inputs(workspace)
    if not inputs["segment_blocks"]:
        raise SystemExit(f"未找到 segment caption：{workspace}/captions/segment/")

    overwritten: List[str] = []

    def _write(path: Path, payload: dict) -> None:
        if path.exists():
            overwritten.append(str(path))
            LOGGER.warning("覆盖本导出工具的旧产物: %s", path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    index = build_scene_segment_index(inputs)
    index_path = output_root / "assets" / "scene" / "scene_segment_index.json"
    _write(index_path, index)

    scene_dir = output_root / "assets" / "i2i_bundles" / "scene"
    object_dir = output_root / "assets" / "i2i_bundles" / "object"
    stats = {"scene_bundles": 0, "scene_ready": 0,
             "object_bundles": 0, "object_ready": 0,
             "assets_checked": 0, "assets_missing": 0}

    def _count_assets(bundle: dict) -> None:
        for ref in bundle.get("references") or []:
            flags = [ref.get("asset_exists")]
            flags += [a.get("exists") for a in ref.get("additional_assets") or []]
            for ok in flags:
                stats["assets_checked"] += 1
                if not ok:
                    stats["assets_missing"] += 1
        stats["assets_checked"] += len(bundle.get("review_assets") or [])

    for g in index["globals"]:
        for s in g["segments"]:
            bundle = build_scene_bundle(inputs, index, g["global_id"], s["segment_id"])
            _write(scene_dir / f"{bundle['task_id']}.json", bundle)
            stats["scene_bundles"] += 1
            stats["scene_ready"] += 1 if bundle["completeness"]["ready"] else 0
            _count_assets(bundle)

    for cand in inputs["object_queue"]:
        bundle = build_object_bundle(inputs, cand)
        _write(object_dir / f"{bundle['task_id']}.json", bundle)
        stats["object_bundles"] += 1
        stats["object_ready"] += 1 if bundle["completeness"]["ready"] else 0
        _count_assets(bundle)

    checked = stats["assets_checked"]
    coverage = {
        g["scope"]: {
            f"segment_{s['segment_id']}": {
                "covered": s["covered"], "anchor_count": s["anchor_count"],
                "covering_scenes": s["covering_scenes"],
            } for s in g["segments"]
        } for g in index["globals"]
    }
    summary = {
        "scene_segment_index": str(index_path),
        "segment_coverage": coverage,
        "uncovered_segments": [u["composite_key"]
                               for u in index["uncovered_segments"]],
        "cross_segment_scenes": [c["scene"]
                                 for c in index["cross_segment_scenes"]],
        "bundles": {
            "scene": {"count": stats["scene_bundles"],
                      "ready": stats["scene_ready"], "dir": str(scene_dir)},
            "object": {"count": stats["object_bundles"],
                       "ready": stats["object_ready"], "dir": str(object_dir)},
        },
        "asset_validation": {
            "checked": checked,
            "missing": stats["assets_missing"],
            "pass_rate": round((checked - stats["assets_missing"]) / checked, 6)
            if checked else None,
        },
        "overwritten_existing_outputs": overwritten,
    }
    return summary


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity", type=Path, default=None,
                   help="Stage3 输出目录（含 registry.json）；"
                        "独立使用 --export-i2i-bundles 时可省略")
    p.add_argument("--stage3-view", type=Path, default=None,
                   help="Stage1.5 输出的 stage3_inputs/<policy>/ 目录"
                        "（须与 Stage3 实际消费的视图一致）")
    p.add_argument("--asset-bank", type=Path, default=None,
                   help="Stage4 输出目录（含 characters/CHRnnn/）；不给则不物化视觉资产")
    p.add_argument("--voice-export", type=Path, default=None,
                   help="export_all_asr_audio_review 输出目录（含 valid/<scope>/<S_ID>/）；"
                        "不给则不合并音频")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--export-i2i-bundles", type=Path, default=None, metavar="WORKSPACE",
                   help="可选 I2I 桥接导出：读取 WORKSPACE 下 scene/key_object/segment/"
                        "role 既有产物，写出 assets/scene/scene_segment_index.json 与 "
                        "assets/i2i_bundles/{scene,object}/（只新增文件）；"
                        "不给 --identity 等参数时独立运行")
    p.add_argument("--i2i-output-root", type=Path, default=None,
                   help="I2I 导出根目录，默认与 --export-i2i-bundles 的 WORKSPACE 相同")
    p.add_argument("--min-share", type=float, default=0.6)
    p.add_argument("--min-weight", type=float, default=1.0)
    p.add_argument("--contest-margin", type=float, default=0.15)
    p.add_argument("--multi-chr-gallery-review", action="store_true",
                   help="P0-3 多 CHR 认领的视觉证据仲裁（默认关闭）：对 id_conflicts 的"
                        "落选 CHR 用 tracklet_features.json 人脸 embedding 做画廊复核，"
                        "结论写入 id_conflicts[].gallery_review；compatible 落选者登记为"
                        "附加独立画廊引用（不复制/合并资产，不改 winner 语义）")
    p.add_argument("--chr-compat-contradiction-floor", type=float, default=0.25,
                   help="P0-3：画廊复核 topk_mean 低于该值视为存在人脸反证（contradicted）")
    p.add_argument("--chr-compat-min-mean", type=float, default=0.40,
                   help="P0-3：画廊复核 topk_mean 达到该值且无反证视为 compatible")
    p.add_argument("--allow-low-frequency-admission", action="store_true",
                   help="P1-3 低频角色分层准入（默认关闭）：weak 但票重达标、无竞争、"
                        "有可靠视觉资产的 CHR 允许物化并进入 role_to_chr")
    p.add_argument("--low-freq-min-weight", type=float, default=0.9,
                   help="P1-3：低频准入的独立票重门槛（独立于 --min-weight）")
    p.add_argument("--no-materialize", action="store_true",
                   help="只产出 role_alignment.json，不复制资产文件")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)s %(message)s")

    if args.i2i_output_root is not None and args.export_i2i_bundles is None:
        build_parser().error("--i2i-output-root 需与 --export-i2i-bundles 联用")

    # 独立 I2I 导出模式：不触发角色对齐流程。
    if args.export_i2i_bundles is not None and args.identity is None:
        summary = export_i2i_bundles(args.export_i2i_bundles, args.i2i_output_root)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    missing = [flag for flag, val in (("--identity", args.identity),
                                      ("--stage3-view", args.stage3_view),
                                      ("--output", args.output)) if val is None]
    if missing:
        build_parser().error(
            "the following arguments are required: %s" % ", ".join(missing))

    alignments = load_tracklet_alignments(args.stage3_view)
    LOGGER.info("loaded %d tracklet alignments from %s",
                len(alignments), args.stage3_view)
    characters, pending, registry = load_identity(args.identity)
    LOGGER.info("identity: %d promoted CHR, %d pending candidates",
                len(characters), len(pending))

    # 1) 逐 CHR 投票
    chr_votes: Dict[str, dict] = {}
    for record in characters:
        chr_id = str(record.get("character_id"))
        tids = [str(t) for t in record.get("tracklet_ids", [])]
        chr_votes[chr_id] = vote_chr(tids, alignments, args.min_share,
                                     args.min_weight, args.contest_margin)
        chr_votes[chr_id]["tracklet_count"] = len(tids)
        chr_votes[chr_id]["shots"] = record.get("shots", [])

    # 2) 反向索引 (scope, S-ID) → 认领它的 CHR 列表，检冲突。
    #    只有 confirmed 进入物化竞争；contested/weak 仅留在 chr_to_roles 报告中。
    role_claims: Dict[Tuple[str, str], List[Tuple[str, float]]] = defaultdict(list)
    for chr_id, vote in chr_votes.items():
        for scope, info in vote["by_scope"].items():
            sid = info.get("subject_id")
            if sid and info.get("status") == "confirmed":
                role_claims[(scope, sid)].append((chr_id, info.get("vote_weight", 0.0)))

    id_conflicts: List[dict] = []
    role_winner: Dict[Tuple[str, str], str] = {}
    for (scope, sid), claimants in sorted(role_claims.items()):
        claimants.sort(key=lambda kv: kv[1], reverse=True)
        role_winner[(scope, sid)] = claimants[0][0]
        if len(claimants) > 1:
            id_conflicts.append({
                "semantic_scope_id": scope, "subject_id": sid,
                "claimed_by": [{"character_id": c, "vote_weight": w}
                               for c, w in claimants],
                "materialized": claimants[0][0],
                "note": "多个视觉身份映射到同一角色；仅物化权重最高者，"
                        "其余保留在 Stage4 的 CHR 目录",
            })

    # P0-3（--multi-chr-gallery-review，默认关闭）：对多 CHR 认领冲突做人脸
    # 画廊复核。只往 id_conflicts 条目写 gallery_review、登记 compatible
    # 落选者的附加独立画廊引用；不复制/合并落选资产，不改票重与 winner 语义。
    gallery_review_map: Dict[Tuple[str, str], dict] = {}
    role_gallery_refs: Dict[str, dict] = {}
    if args.multi_chr_gallery_review and id_conflicts:
        conflict_claims = {
            (entry["semantic_scope_id"], entry["subject_id"]):
                [c["character_id"] for c in entry["claimed_by"]]
            for entry in id_conflicts
        }
        gallery_review_map = review_conflict_galleries(
            conflict_claims, characters, args.identity,
            args.chr_compat_contradiction_floor, args.chr_compat_min_mean)
        for entry in id_conflicts:
            key = (entry["semantic_scope_id"], entry["subject_id"])
            review = gallery_review_map.get(key)
            if review is None:
                continue
            entry["gallery_review"] = {
                "winner": review["winner"],
                "losers": review["losers"],
                "note": "落选 CHR 与 winner 人脸画廊（tracklet_features.json）"
                        "topk_mean 复核；contradicted 建议回 Stage1.5 复核身份标签",
            }
            if review["compatible"]:
                scoped = f"{entry['semantic_scope_id']}:{entry['subject_id']}"
                role_gallery_refs[scoped] = {
                    "primary": review["winner"],
                    "additional": list(review["compatible"]),
                }

    # P1-3（--allow-low-frequency-admission，默认关闭）：低频角色分层准入。
    # 准入的 (scope, sid) 本来就只有该 CHR 一个 winner，并入 role_winner
    # 即自然进入 role_to_chr；物化时打 admission_tier=low_frequency。
    low_freq_admitted: Dict[Tuple[str, str], str] = {}
    if args.allow_low_frequency_admission:
        characters_by_id = {str(record.get("character_id")): record
                            for record in characters}
        low_freq_admitted = select_low_frequency_roles(
            chr_votes, role_winner, characters_by_id, args.low_freq_min_weight)
        role_winner.update(low_freq_admitted)
        for (scope, sid), chr_id in sorted(low_freq_admitted.items()):
            LOGGER.info("low-frequency admitted %s/%s <- %s (weight=%s)",
                        scope, sid, chr_id,
                        chr_votes[chr_id]["by_scope"][scope].get("vote_weight"))

    # 3) 物化 + 汇总
    materialized: List[dict] = []
    if not args.no_materialize:
        for (scope, sid), chr_id in sorted(role_winner.items()):
            info = chr_votes[chr_id]["by_scope"][scope]
            admission_tier = ("low_frequency"
                              if (scope, sid) in low_freq_admitted else None)
            review = gallery_review_map.get((scope, sid))
            related = (
                [{"character_id": c,
                  "best_topk_mean": review["losers"][c]["best_topk_mean"],
                  "relation": "compatible_additional_gallery"}
                 for c in review["compatible"]]
                if review is not None and review["compatible"] else None
            )
            materialized.append(materialize_role(
                args.output, scope, sid, chr_id,
                args.asset_bank, args.voice_export, info,
                admission_tier=admission_tier,
                related_visual_characters=related,
            ))
            LOGGER.info("materialized %s/%s <- %s (%s)",
                        scope, sid, chr_id, info.get("status"))

    unassigned = [
        {"character_id": chr_id,
         "statuses": {s: i.get("status") for s, i in vote["by_scope"].items()},
         "tracklet_count": vote["tracklet_count"]}
        for chr_id, vote in chr_votes.items()
        if all(i.get("subject_id") is None for i in vote["by_scope"].values())
    ]

    cross_scope_aliases = {
        chr_id: aliases
        for chr_id, vote in chr_votes.items()
        if len(aliases := [
            {"semantic_scope_id": scope, "subject_id": info.get("subject_id"),
             "status": info.get("status")}
            for scope, info in sorted(vote["by_scope"].items())
            if info.get("subject_id")
        ]) > 1
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "identity": str(args.identity),
            "stage3_view": str(args.stage3_view),
            "asset_bank": str(args.asset_bank) if args.asset_bank else None,
            "voice_export": str(args.voice_export) if args.voice_export else None,
        },
        "params": {"min_share": args.min_share, "min_weight": args.min_weight,
                   "contest_margin": args.contest_margin},
        "character_count": len(characters),
        "pending_candidate_count": len(pending),
        "chr_to_roles": chr_votes,
        "role_to_chr": {f"{scope}:{sid}": chr_id
                        for (scope, sid), chr_id in sorted(role_winner.items())},
        "id_conflicts": id_conflicts,
        # 同一 CHR 在多个 scope 下的 S-ID 别名：仅展示、不合并（S-ID 本就是
        # clip 局部编号，跨 scope 不同 S-ID 不代表冲突）。
        "cross_scope_aliases": cross_scope_aliases,
        "unassigned_visual_identities": unassigned,
        "materialized_roles": materialized,
        "stage3_registry_stats": {
            k: registry.get(k) for k in
            ("character_count", "cannot_link_violation_count") if k in registry
        },
    }
    # 以下键仅在对应开关开启时追加；默认关闭时报告保持原有结构。
    if args.multi_chr_gallery_review:
        report["params"].update({
            "multi_chr_gallery_review": True,
            "chr_compat_contradiction_floor": args.chr_compat_contradiction_floor,
            "chr_compat_min_mean": args.chr_compat_min_mean,
        })
        if role_gallery_refs:
            report["role_gallery_refs"] = role_gallery_refs
    if args.allow_low_frequency_admission:
        report["params"].update({
            "allow_low_frequency_admission": True,
            "low_freq_min_weight": args.low_freq_min_weight,
        })
        report["low_frequency_admissions"] = [
            {"semantic_scope_id": scope, "subject_id": sid, "character_id": chr_id,
             "vote_weight": chr_votes[chr_id]["by_scope"][scope].get("vote_weight")}
            for (scope, sid), chr_id in sorted(low_freq_admitted.items())
        ]
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "role_alignment.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    LOGGER.info("wrote %s", report_path)

    # 组合模式：对齐流程之后追加 I2I 桥接导出（读取 workspace 既有产物）。
    if args.export_i2i_bundles is not None:
        export_summary = export_i2i_bundles(args.export_i2i_bundles,
                                            args.i2i_output_root)
        LOGGER.info("i2i export done: %s", json.dumps(export_summary["bundles"],
                                                      ensure_ascii=False))

    print(json.dumps({
        "characters": len(characters),
        "roles_materialized": len(materialized),
        "id_conflicts": len(id_conflicts),
        "unassigned_chr": len(unassigned),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
