# -*- coding: utf-8 -*-
"""兼容性 shim：让 `import build_object_grounding_queue_from_shot_captions_v1`
能够找到原 v1 模块。

原始 ``build_object_grounding_queue_v2`` 用旧文件名 import v1；
为保持 v2 源码"原封不动"，这里提供一个同名 shim 转发到
``assets.key_object.build_object_grounding_queue_v1``（真正的实现）。
"""

from extract_full_pipeline.assets.key_object.build_object_grounding_queue_v1 import (  # noqa: F401
    build_queue,
    read_jsonl,
)

__all__ = ["build_queue", "read_jsonl"]