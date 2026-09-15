# -*- coding: utf-8 -*-
"""三级标注 Schema 定义（冻结契约 v1.0）。

clip（镜头级）→ segment（场景级）→ global（全片级）
所有记录以 JSONL 落盘，每行一条记录，字段缺失时用默认值，
读取时容忍未知字段（向前兼容）。

注意：本文件是全部模块的契约，任何字段改动必须同步：
  1. SCHEMA_VERSION 升版本号
  2. docs/00_schema_design.md 文档
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _filter_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    """from_dict 时只保留 dataclass 已声明的字段，容忍未知字段。"""
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


def now_ts() -> float:
    return time.time()


# clip 状态机（唯一权威顺序；qc/m03/m05 的续跑跳过都基于它）
STATUS_ORDER = ("created", "qc_done", "asr_done", "recut_done",
                "speaker_done", "captioned")


def status_at_least(rec, name: str) -> bool:
    """rec.status 是否已达到（或越过）状态链中的 name。未知状态按未达到处理。"""
    try:
        return STATUS_ORDER.index(rec.status) >= STATUS_ORDER.index(name)
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# 子结构：连续性标记（邻接表节点指针）
# ---------------------------------------------------------------------------

@dataclass
class AdjacentLink:
    """前/后相邻 clip 指针 + 连续性标记。

    对应框架图 ②④ 的规则：
      - 第一个 clip 的 prev.clip_id=None, continuous=False
      - 最后一个 clip 的 next.clip_id=None, continuous=False
      - ④ 再切割产生的相邻子 clip（002-1/002-2）互标 continuous=True
    """
    clip_id: Optional[str] = None
    continuous: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AdjacentLink":
        return cls(**_filter_kwargs(cls, d or {}))


# ---------------------------------------------------------------------------
# 子结构：ASR
# ---------------------------------------------------------------------------

@dataclass
class WordTS:
    """词级时间戳（供 ④ 精确切刀）。"""
    word: str = ""
    start: float = 0.0   # clip 内相对秒
    end: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WordTS":
        return cls(**_filter_kwargs(cls, d or {}))


@dataclass
class ASRSegment:
    """一句/一段说话，带说话边界与词级时间戳。"""
    start: float = 0.0          # clip 内相对秒
    end: float = 0.0
    text: str = ""
    words: List[WordTS] = field(default_factory=list)
    confidence: float = 0.0
    speaker_local: Optional[str] = None   # ASR/diarization 给的局部说话人标签
    lang: str = ""                        # 语种标签

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ASRSegment":
        d = dict(d or {})
        d["words"] = [WordTS.from_dict(w) for w in d.get("words", [])]
        return cls(**_filter_kwargs(cls, d))


@dataclass
class AudioEvent:
    """非人声音频事件（BGM/音效/环境声），音视频模型的训练信号之一。"""
    start: float = 0.0
    end: float = 0.0
    label: str = ""          # bgm / sfx / ambient / silence ...
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AudioEvent":
        return cls(**_filter_kwargs(cls, d or {}))


@dataclass
class ASRResult:
    """clip 级 ASR 打标结果（模块③产出）。"""
    engine: str = ""                       # openmoss / sensevoice / whisperx ...
    segments: List[ASRSegment] = field(default_factory=list)
    audio_events: List[AudioEvent] = field(default_factory=list)
    main_lang: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "segments": [s.to_dict() for s in self.segments],
            "audio_events": [asdict(e) for e in self.audio_events],
            "main_lang": self.main_lang,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ASRResult":
        d = dict(d or {})
        d["segments"] = [ASRSegment.from_dict(s) for s in d.get("segments", [])]
        d["audio_events"] = [AudioEvent.from_dict(e) for e in d.get("audio_events", [])]
        return cls(**_filter_kwargs(cls, d))


# ---------------------------------------------------------------------------
# 子结构：质检打分卡（视频/音频/同步/时长 → 等级 A/B/C）
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """单项质检：0~1 分数 + 硬剔除标志。"""
    score: float = 1.0
    hard_reject: bool = False
    detail: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QualityScore":
        return cls(**_filter_kwargs(cls, d or {}))


@dataclass
class QualityCard:
    """clip 质检打分卡（质检大框产出）。

    视频侧: aesthetic / subtitle / watermark / blur / optical_flow
    音频侧: snr / silence / dnsmos
    同步  : av_sync / lip_sync
    时长  : duration
    """
    # 视频侧
    aesthetic: QualityScore = field(default_factory=QualityScore)
    subtitle: QualityScore = field(default_factory=QualityScore)   # OCR 命中字幕区
    watermark: QualityScore = field(default_factory=QualityScore)
    blur: QualityScore = field(default_factory=QualityScore)
    optical_flow: QualityScore = field(default_factory=QualityScore)
    # 音频侧
    snr: QualityScore = field(default_factory=QualityScore)
    silence: QualityScore = field(default_factory=QualityScore)
    dnsmos: QualityScore = field(default_factory=QualityScore)
    # 同步与时长
    av_sync: QualityScore = field(default_factory=QualityScore)
    lip_sync: QualityScore = field(default_factory=QualityScore)
    duration: QualityScore = field(default_factory=QualityScore)
    # 汇总
    grade: str = "A"            # A / B / C（⑦ 聚合按等级加权；C 级视为不合格）
    rejected: bool = False      # 任一硬剔除为 True 则整 clip 剔除
    extra: Dict[str, Any] = field(default_factory=dict)  # 原始测量值（of_score、lse_d 等）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QualityCard":
        d = dict(d or {})
        for name in ("aesthetic", "subtitle", "watermark", "blur", "optical_flow",
                     "snr", "silence", "dnsmos", "av_sync", "lip_sync", "duration"):
            if isinstance(d.get(name), dict):
                d[name] = QualityScore.from_dict(d[name])
        return cls(**_filter_kwargs(cls, d))


# ---------------------------------------------------------------------------
# 子结构：说话人对应（模块⑤产出）
# ---------------------------------------------------------------------------

@dataclass
class SpeakerMapEntry:
    """clip 内一条 face track ↔ 全局人物 ID 的映射。"""
    track_id: str = ""                    # clip 内 face track id
    global_id: Optional[str] = None       # Sub1..SubN；未知人物为 None
    is_speaking: bool = False
    speak_intervals: List[List[float]] = field(default_factory=list)  # [[s,e],...]
    evidence: str = ""                    # asd / lipsync / voiceprint / vlm / manual
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeakerMapEntry":
        return cls(**_filter_kwargs(cls, d or {}))


# ---------------------------------------------------------------------------
# 子结构：clip caption（模块⑥产出，固定 JSON 字段，供 ⑦ 聚合）
# ---------------------------------------------------------------------------

@dataclass
class ClipCaption:
    """结构化镜头描述。人物一律用全局 ID 指代，禁止起名。"""
    scene: str = ""                 # 场景/地点
    characters: List[str] = field(default_factory=list)   # 画面人物全局 ID 列表
    action: str = ""                # 动作与事件
    dialogue: str = ""              # 对话摘要（谁说给谁听，用全局 ID）
    mood: str = ""                  # 情绪氛围
    shot_type: str = ""             # 景别：特写/中景/全景...
    camera_movement: str = ""       # 运镜：推/拉/摇/移/固定
    audio_desc: str = ""            # 音频描述（人声/BGM/音效）
    raw_text: str = ""              # 模型原始输出（兜底）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClipCaption":
        return cls(**_filter_kwargs(cls, d or {}))


# ---------------------------------------------------------------------------
# 一级：Clip 记录（镜头级）
# ---------------------------------------------------------------------------

@dataclass
class ClipRecord:
    """镜头级记录。④ 冻结后的"最终 clip 宇宙"成员。"""
    schema_version: str = SCHEMA_VERSION
    clip_id: str = ""               # 形如 002 或再切割后的 002-1
    video_id: str = ""
    # 时间/帧区间（相对源片）
    start_frame: int = 0
    end_frame: int = 0
    fps: float = 25.0
    start_time: float = 0.0         # 秒
    end_time: float = 0.0
    duration: float = 0.0           # 秒
    # 邻接表（首尾 clip 的对外指针为 None + continuous=False）
    prev: AdjacentLink = field(default_factory=AdjacentLink)
    next: AdjacentLink = field(default_factory=AdjacentLink)
    parent_clip_id: Optional[str] = None   # ④ 再切割前的原始 clip id（未切割则 None）
    # 产物路径
    video_path: str = ""
    audio_path: str = ""            # 48kHz wav
    thumbnail: str = ""
    # 各模块标注
    quality: QualityCard = field(default_factory=QualityCard)
    asr: ASRResult = field(default_factory=ASRResult)
    speaker_map: List[SpeakerMapEntry] = field(default_factory=list)
    caption: ClipCaption = field(default_factory=ClipCaption)
    # 状态机：created -> qc_done -> asr_done -> recut_done -> speaker_done -> captioned
    status: str = "created"
    updated_at: float = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["prev"] = self.prev.to_dict()
        d["next"] = self.next.to_dict()
        d["quality"] = self.quality.to_dict()
        d["asr"] = self.asr.to_dict()
        d["speaker_map"] = [asdict(e) for e in self.speaker_map]
        d["caption"] = self.caption.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClipRecord":
        d = dict(d or {})
        d["prev"] = AdjacentLink.from_dict(d.get("prev"))
        d["next"] = AdjacentLink.from_dict(d.get("next"))
        d["quality"] = QualityCard.from_dict(d.get("quality"))
        d["asr"] = ASRResult.from_dict(d.get("asr"))
        d["speaker_map"] = [SpeakerMapEntry.from_dict(e) for e in d.get("speaker_map", [])]
        d["caption"] = ClipCaption.from_dict(d.get("caption"))
        return cls(**_filter_kwargs(cls, d))


# ---------------------------------------------------------------------------
# 二级：Segment 记录（场景级）
# ---------------------------------------------------------------------------

@dataclass
class SegmentRecord:
    """场景级记录（模块⑦产出）。"""
    schema_version: str = SCHEMA_VERSION
    segment_id: str = ""
    video_id: str = ""
    clip_ids: List[str] = field(default_factory=list)   # 窗口内 clip 序列（有序）
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    unqualified_ratio: float = 0.0       # 不合格 clip 时长占比（须 < 0.2）
    dinov3_score: float = 0.0            # 窗内 clip 两两 DINOv3 相似度均值
    continuity_ok: bool = True           # 小模型判定 scene 连续性
    segment_prompt: str = ""             # clip caption 聚合成的场景级 prompt
    grade_weight: float = 1.0            # 按 clip 等级加权后的训练权重
    status: str = "created"
    updated_at: float = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SegmentRecord":
        return cls(**_filter_kwargs(cls, d or {}))


# ---------------------------------------------------------------------------
# 三级：Global 记录（全片级）
# ---------------------------------------------------------------------------

@dataclass
class SubjectEntry:
    """全局人物花名册条目（模块①产出）。"""
    sub_id: str = ""                     # Sub1..SubN
    centroid_path: str = ""              # centroid embedding 文件（npy）路径
    rep_faces: List[str] = field(default_factory=list)  # 代表人脸图集（top-k 图路径）
    appear_clips: List[str] = field(default_factory=list)   # 出现的 clip 列表
    cluster_size: int = 0
    description: str = ""                # VLM 对代表脸图集生成的人物表征（⑤⑥ 对号入座用）

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubjectEntry":
        return cls(**_filter_kwargs(cls, d or {}))


@dataclass
class GlobalRecord:
    """全片级记录（① 建花名册，⑦ 写 global_prompt 后闭环）。"""
    schema_version: str = SCHEMA_VERSION
    video_id: str = ""
    source_path: str = ""                # 源片路径
    duration: float = 0.0
    roster: List[SubjectEntry] = field(default_factory=list)   # Sub1..SubN
    segment_ids: List[str] = field(default_factory=list)       # 全片 segment 序列
    global_prompt: str = ""              # 全片人物/主线/风格
    status: str = "created"
    updated_at: float = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["roster"] = [asdict(s) for s in self.roster]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GlobalRecord":
        d = dict(d or {})
        d["roster"] = [SubjectEntry.from_dict(s) for s in d.get("roster", [])]
        return cls(**_filter_kwargs(cls, d))
