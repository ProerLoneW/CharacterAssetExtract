# -*- coding: utf-8 -*-
"""FastAPI 应用组装:静态前端 + 统一 API + 媒体,同一端口同源(文档三 §2/§7/§9)。

启动方式(在 result_viewer 目录):
    python -m backend.app --config viewer_config.json
开发模式(前端未构建时):--dev-frontend 指向 Vite dev server 的 origin,页面自动跳转。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .adapters import RENDERER_CAPABILITIES, REGISTERED_ADAPTER_IDS, build_registry
from .config import ConfigError, RuntimeConfig, load_config
from .document_parser import DocumentParser
from .document_store import DocumentStore
from .envelope import ApiError
from .media_service import MediaService, iter_file_chunks
from .module_service import ModuleService
from .run_registry import RunRegistry


class PrefixCompatMiddleware:
    """接受带平台代理前缀和不带前缀的两种请求路径。

    路由统一注册在 base_path 前缀下；平台代理若是"剥离前缀转发"(或本地直连),
    请求路径不含前缀,这里补上前缀再交给路由,使两种代理行为都可用。
    """

    def __init__(self, app, prefix: str) -> None:
        self.app = app
        self.prefix = (prefix or "").rstrip("/")

    async def __call__(self, scope, receive, send):
        if self.prefix and scope.get("type") == "http":
            path = scope.get("path", "")
            if path != self.prefix and not path.startswith(self.prefix + "/"):
                scope["path"] = self.prefix + path
        await self.app(scope, receive, send)


def create_app(config: RuntimeConfig) -> FastAPI:
    adapter_registry = build_registry(MediaService({}))
    parser = DocumentParser(
        adapter_capabilities=set(REGISTERED_ADAPTER_IDS),
        renderer_capabilities=RENDERER_CAPABILITIES,
    )
    store = DocumentStore(
        config.doc_path,
        parser,
        poll_interval_sec=config.poll_interval_sec,
        debounce_ms=config.debounce_ms,
    )
    store.initial_compile()
    store.start_polling()

    media_service = MediaService.__new__(MediaService)  # 先占位,运行注册表就绪后重建
    runs = RunRegistry(config)
    media_service = MediaService({r.run_id: r for r in runs.all_runs()})
    # 重建适配器(共享最终 media 服务)
    adapter_registry = build_registry(media_service)
    service = ModuleService(config, store, runs, media_service, adapter_registry)

    base = config.base_path or ""
    app = FastAPI(title="结果展示网站", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.result_viewer = {
        "config": config,
        "store": store,
        "service": service,
        "media": media_service,
    }

    # 开发期 Vite 直连需要跨域;生产同源无影响
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET"],
        allow_headers=["*"],
        expose_headers=["ETag", "Content-Range", "Accept-Ranges"],
    )

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        return JSONResponse(status_code=exc.status_code, content=exc.to_payload(store.revision()))

    # ------------------------------------------------------------------
    # API(相对 BASE_PATH;前端用相对路径 fetch,天然匹配代理前缀)
    # ------------------------------------------------------------------

    @app.get(base + "/api/v1/version")
    async def version():
        snap = store.current
        return {
            "api_version": "1.0",
            "doc_revision": store.revision(),
            "doc_compiled_at": snap.compiled_at if snap else None,
            "doc_ok": bool(snap and snap.ok),
            "last_document_errors": store.last_error[:10],
            "adapter_capabilities": sorted(REGISTERED_ADAPTER_IDS),
            "renderer_capabilities": sorted(RENDERER_CAPABILITIES),
            "ui_build_id": app.state.result_viewer.get("ui_build_id"),
            "service": "result_viewer",
        }

    @app.get(base + "/api/v1/spec")
    async def spec(request: Request, doc_revision: str | None = None):
        etag = f'"{store.revision()}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        payload = service.spec_payload(doc_revision)
        return JSONResponse(payload, headers={"ETag": etag, "Cache-Control": "no-cache"})

    @app.get(base + "/api/v1/runs")
    async def list_runs():
        return service.runs_payload()

    @app.get(base + "/api/v1/runs/{run_id}/modules")
    async def run_modules(run_id: str, doc_revision: str | None = None):
        return service.modules_payload(run_id, doc_revision)

    @app.get(base + "/api/v1/runs/{run_id}/modules/{module_id}/items")
    async def module_items(
        run_id: str,
        module_id: str,
        request: Request,
        doc_revision: str | None = None,
        scope: str | None = None,
        subject: str | None = None,
        shot: str | None = None,
        status: str | None = None,
        variant: str | None = None,
        segment: str | None = None,
        cursor: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
    ):
        filters = {"scope": scope, "subject": subject, "shot": shot, "status": status, "variant": variant, "segment": segment}
        return service.items_payload(run_id, module_id, doc_revision, filters, cursor, limit)

    @app.get(base + "/api/v1/runs/{run_id}/modules/{module_id}/items/{entity_id}")
    async def module_item(run_id: str, module_id: str, entity_id: str, doc_revision: str | None = None):
        return service.item_payload(run_id, module_id, entity_id, doc_revision)

    @app.get(base + "/api/v1/runs/{run_id}/artifacts/{artifact_id}/preview")
    async def artifact_preview(run_id: str, artifact_id: str, offset: int = Query(default=0, ge=0), limit: int = Query(default=50, ge=1, le=200)):
        return service.artifact_preview_payload(run_id, artifact_id, offset, limit)

    @app.get(base + "/api/v1/runs/{run_id}/lineage/{entity_id}")
    async def lineage(run_id: str, entity_id: str, doc_revision: str | None = None):
        return service.lineage_payload(run_id, entity_id, doc_revision)

    @app.get(base + "/healthz")
    async def healthz():
        snap = store.current
        doc_ok = bool(snap and snap.ok)
        roots_ok = {}
        for run in runs.all_runs():
            roots_ok[run.run_id] = {rid: root.path.is_dir() for rid, root in run.roots.items()}
        return {
            "status": "ok" if doc_ok else "degraded",
            "document": {
                "ok": doc_ok,
                "revision": store.revision(),
                "source": config.doc_path.name,
                "last_errors": store.last_error[:3],
            },
            "runs": roots_ok,
            "base_path": base,
        }

    # ------------------------------------------------------------------
    # 媒体:GET/HEAD + Range(文档三 §8)
    # ------------------------------------------------------------------

    def _media_response(media_id: str, request: Request, method: str) -> Response:
        resolved = media_service.resolve(media_id)
        size = resolved.size
        range_header = request.headers.get("range")
        common = {
            "Accept-Ranges": "bytes",
            "ETag": resolved.etag,
            "Cache-Control": "private, max-age=300",
        }
        if method == "HEAD":
            return Response(status_code=200, headers={**common, "Content-Length": str(size), "Content-Type": resolved.mime})
        if range_header:
            span = MediaService.parse_range(range_header, size)
            if span is None:
                return Response(
                    status_code=416,
                    headers={**common, "Content-Range": f"bytes */{size}"},
                )
            start, end = span
            from fastapi.responses import StreamingResponse

            return StreamingResponse(
                iter_file_chunks(resolved.file_path, start, end),
                status_code=206,
                headers={
                    **common,
                    "Content-Range": f"bytes {start}-{end}/{size}",
                    "Content-Length": str(end - start + 1),
                    "Content-Type": resolved.mime,
                },
            )
        return FileResponse(
            resolved.file_path,
            headers={**common, "Content-Type": resolved.mime},
        )

    @app.head(base + "/media/{media_id}")
    async def media_head(media_id: str, request: Request):
        return _media_response(media_id, request, "HEAD")

    @app.get(base + "/media/{media_id}")
    async def media_get(media_id: str, request: Request):
        return _media_response(media_id, request, "GET")

    # ------------------------------------------------------------------
    # 静态前端(构建产物;query 路由刷新仍回到同一入口页)
    # ------------------------------------------------------------------

    dist = config.frontend_dist
    if dist is not None and dist.is_dir():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount(base + "/assets", StaticFiles(directory=assets), name="assets")

        @app.get(base + "/", include_in_schema=False)
        async def index():
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

        @app.get(base + "/index.html", include_in_schema=False)
        async def index_html():
            return RedirectResponse(base + "/")

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    arg = argparse.ArgumentParser(description="结果展示网站服务")
    arg.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "viewer_config.json"))
    arg.add_argument("--host", default=None)
    arg.add_argument("--port", type=int, default=None)
    arg.add_argument("--root-path", default=None, help="平台代理保留/剥离前缀时的 FastAPI root_path")
    args = arg.parse_args(argv)

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        raise SystemExit(f"配置错误: {exc}")

    host = args.host or config.listen_host
    port = args.port or config.service_port
    root_path = args.root_path if args.root_path is not None else ""

    app = create_app(config)
    app.state.result_viewer["ui_build_id"] = _ui_build_id(config)
    print(f"[result_viewer] doc={'ok' if app.state.result_viewer['store'].current and app.state.result_viewer['store'].current.ok else 'ERROR'} "
          f"runs={config.allowed_runs} base_path='{config.base_path}'")
    uvicorn.run(PrefixCompatMiddleware(app, config.base_path), host=host, port=port,
                root_path=root_path, log_level="info")


def _ui_build_id(config: RuntimeConfig) -> str | None:
    dist = config.frontend_dist
    if dist is None:
        return None
    marker = dist / "build_id.json"
    if marker.is_file():
        try:
            return json.loads(marker.read_text(encoding="utf-8")).get("build_id")
        except Exception:
            return None
    return None


if __name__ == "__main__":
    main()
