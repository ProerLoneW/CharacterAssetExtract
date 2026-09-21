# -*- coding: utf-8 -*-
"""结果展示网站后端包。

分层(对应 docs/03_结果展示网站设计.md §2/§10):
- document_parser / document_store : 文档二(PipelineDoc 1.0)解析、校验、快照与热加载
- run_registry                     : 运行注册表、产物绑定、路径边界与重写
- adapters/*                       : 原始 schema -> 统一 ResultEnvelope 的数据适配器
- media_service                    : 受控 media_id 的媒体字节与 Range 响应
- module_service                   : 按模块聚合状态/指标/实体的读取服务
- app                              : FastAPI 组装(静态前端 + API + 媒体同源)
"""

__version__ = "0.1.0"
