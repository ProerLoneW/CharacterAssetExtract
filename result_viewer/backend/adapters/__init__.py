# -*- coding: utf-8 -*-
"""适配器注册表。

文档三 §5.1 要求:文档二声明的 view.adapter 必须能解析到已注册能力,否则候选快照被拒绝。
因此这里登记文档二全部 27 个 adapter ID:
- 有数据实现的(本运行可绑定真实产物)直接实例化;
- 其余用 RegisteredOnlyAdapter 登记 —— 绑定后返回明确"未实现数据解析"状态,
  不假装成功,也不静默空列表。
"""

from __future__ import annotations

from ..envelope import ApiError, UNREGISTERED_ADAPTER
from ..media_service import MediaService
from .base import ModuleAdapter, RegisteredOnlyAdapter
from .person_assets import LegacyCharactersAdapter, TrackletsAdapter
from .semantic_captions import GlobalCaptionAdapter, SegmentCaptionAdapter, ShotCaptionAdapter
from .video_structure import GlobalsAdapter, SegmentMediaAdapter, ShotsAdapter
from .mainline_assets import (GlobalAlignmentAdapter, CharactersGlobalAdapter,
                             ObjectQueueAdapter, ObjectObservationsAdapter, SceneIndexAdapter,
                             SegmentPlanAdapter, AsrTimelineAdapter, I2ITestAdapter,
                             TTSTestAdapter)

# adapter_id -> (module_id, 说明) —— 文档二中声明的全部能力
REGISTERED_ADAPTER_IDS: dict[str, tuple[str, str, str]] = {
    "source-v1": ("source_prepare", "source_video", "source_video"),
    "shots-v1": ("shot_split", "镜头清单", "shots"),
    "globals-v1": ("global_group", "全局片段清单", "globals"),
    "segment-plan-v1": ("segment_plan", "上游固定分段清单", "segment_plan"),
    "asr-v1": ("asr_timeline", "权威 ASR 输入", "authoritative_asr"),
    "global-caption-v1": ("global_describe", "全局描述与说话归属", "global_captions"),
    "shot-caption-v1": ("shot_describe", "镜头描述清单", "shot_captions"),
    "segment-caption-v1": ("segment_describe", "片段描述清单", "segment_captions"),
    "segment-media-v1": ("segment_export", "导出片段清单", "segment_media"),
    "tracklets-v2": ("person_tracks", "人物轨迹证据目录", "tracklets"),
    "global-alignment-v1": ("person_align", "全局人物归属清单", "direct_alignment"),
    "semantic-tracklets-v3": ("legacy_refine", "语义筛选与策略视图", "semantic_tracklet_views"),
    "visual-identity-v2": ("legacy_identity", "视觉身份图与 CHR 注册表", "visual_identity"),
    "chr-gallery-v1": ("legacy_gallery", "CHR 资产画廊", "chr_gallery"),
    "speaker-registry-v1": ("voice_model", "声音建模产物", "speaker_registry"),
    "voice-preview-v2": ("voice_preview", "ASR 声音预览清单", "voice_previews"),
    "characters-global-v1": ("character_pack", "新链角色素材目录", "characters"),
    "characters-legacy-v1": ("legacy_bind", "旧链角色绑定报告", "role_binding"),
    "object-queue-v2": ("object_queue", "物体定位队列", "object_candidates"),
    "object-observations-v3-v4": ("object_ground", "物体观测清单", "object_observations"),
    "scene-index-v3": ("scene_select", "场景参考与评分", "scene_assets"),
    "i2i-bundle-v1": ("generation_bundle", "I2I 任务包", "i2i_bundles"),
    "i2i-test-v1": ("i2i_review", "i2i_results", "i2i_results"),
    "tts-test-v1": ("tts_review", "TTS 测试结果", "tts_results"),
    "clip-records-v1": ("compat_structure", "ClipRecord 兼容记录", "compat_records"),
    "demo-assets-v1": ("asset_demo", "演示资产输出", "demo_assets"),
    "workspace-artifacts-v1": ("result_browse", "可浏览产物目录", "workspace_artifacts"),
}

# 前端已实现的受控 renderer 类型(文档三 §4 表格)
RENDERER_CAPABILITIES = {
    "source-overview",
    "video-timeline",
    "transcript",
    "caption-detail",
    "tracklet-gallery",
    "alignment-evidence",
    "identity-graph",
    "character-gallery",
    "audio-evidence",
    "record-table",
    "object-evidence",
    "scene-gallery",
    "generation-bundle",
    "image-comparison",
    "audio-comparison",
    "artifact-browser",
}

def build_registry(media: MediaService) -> dict[str, ModuleAdapter]:
    from .voice_assets import VoicePreviewAdapter

    registry: dict[str, ModuleAdapter] = {}

    for cls in (
        ShotsAdapter,
        GlobalsAdapter,
        SegmentMediaAdapter,
        GlobalCaptionAdapter,
        ShotCaptionAdapter,
        SegmentCaptionAdapter,
        VoicePreviewAdapter,
        TrackletsAdapter,
        LegacyCharactersAdapter,
        GlobalAlignmentAdapter,
        CharactersGlobalAdapter,
        ObjectQueueAdapter,
        ObjectObservationsAdapter,
        SceneIndexAdapter,
        SegmentPlanAdapter,
        AsrTimelineAdapter,
        I2ITestAdapter,
        TTSTestAdapter,
    ):
        adapter = cls(media)
        registry[adapter.adapter_id] = adapter

    for adapter_id, (module_id, note, primary_artifact) in REGISTERED_ADAPTER_IDS.items():
        if adapter_id not in registry:
            only = RegisteredOnlyAdapter(adapter_id, module_id, note, media)
            only.required_artifacts = (primary_artifact,)
            registry[adapter_id] = only

    return registry


def get_adapter(registry: dict[str, ModuleAdapter], adapter_id: str) -> ModuleAdapter:
    adapter = registry.get(adapter_id)
    if adapter is None:
        raise ApiError(UNREGISTERED_ADAPTER, f"adapter 未注册: {adapter_id}", status_code=500)
    return adapter
