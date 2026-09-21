# -*- coding: utf-8 -*-
"""result_viewer 一键启动/部署脚本(Windows 本机与 Linux 服务器通用)。

用法(在仓库根目录):
    python result_viewer/serve.py                          # 0.0.0.0:<配置端口>,同网段可访问
    python result_viewer/serve.py --host 127.0.0.1         # 仅本机访问
    python result_viewer/serve.py --port 8930
    python result_viewer/serve.py --root-path /gemini_web/result_viewer/<代理段>/
                                    # 平台代理"保留前缀转发"时设置;剥离型代理不用设

等价环境变量: RESULT_VIEWER_HOST / RESULT_VIEWER_PORT / RESULT_VIEWER_ROOT_PATH(命令行优先)。
旧 asset_viewer 的等价命令是 python -m extract_full_pipeline.asset_viewer.server --host 0.0.0.0 --port 8910。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "result_viewer" / "viewer_config.json"


def _stdout_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _load_defaults() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _preflight_backend() -> None:
    missing = []
    for mod in ("fastapi", "uvicorn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"缺少后端依赖: {', '.join(missing)}。先执行: {sys.executable} -m pip install {' '.join(missing)}"
        )


def _ensure_frontend(auto_build: bool) -> bool:
    dist = REPO_ROOT / "result_viewer" / "frontend_dist"
    if (dist / "index.html").is_file():
        return True
    if not auto_build:
        return False
    npm = "npm.cmd" if os.name == "nt" else "npm"
    frontend = REPO_ROOT / "result_viewer" / "frontend"
    if not (frontend / "package.json").is_file() or subprocess.run([npm, "-v"], capture_output=True).returncode != 0:
        return False
    print("[serve] frontend_dist 缺失,尝试构建前端(首次约 1-2 分钟)…", flush=True)
    if subprocess.run([npm, "install"], cwd=frontend, shell=os.name == "nt").returncode != 0:
        return False
    return subprocess.run([npm, "run", "build"], cwd=frontend, shell=os.name == "nt").returncode == 0


def _lan_ips() -> list[str]:
    ips: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, socket.AF_INET):
            addr = info[4][0]
            if addr not in ips and not addr.startswith("127."):
                ips.append(addr)
    except Exception:
        pass
    return ips


def main() -> None:
    _stdout_utf8()
    arg = argparse.ArgumentParser(description="结果展示网站一键启动")
    arg.add_argument("--host", default=os.environ.get("RESULT_VIEWER_HOST") or "0.0.0.0",
                     help="默认 0.0.0.0(暴露给可访问本机/服务器的人);仅本机用 127.0.0.1")
    arg.add_argument("--port", type=int, default=int(os.environ.get("RESULT_VIEWER_PORT") or 0),
                     help="默认取 viewer_config.json 的 service_port")
    arg.add_argument("--root-path", default=os.environ.get("RESULT_VIEWER_ROOT_PATH") or "",
                     help="平台代理保留前缀转发时设置,如 /gemini_web/result_viewer/xxx/")
    arg.add_argument("--config", default=str(CONFIG_PATH))
    arg.add_argument("--skip-build", action="store_true", help="前端产物缺失时不自动构建")
    args = arg.parse_args()

    cfg = _load_defaults()
    port = args.port or int(cfg.get("service_port") or 8930)

    _preflight_backend()
    if not _ensure_frontend(not args.skip_build):
        print("[serve] 警告: frontend_dist 缺失且未构建,页面将不可用(API 仍可访问)。", flush=True)
        print("[serve] 手动构建: cd result_viewer/frontend && npm install && npm run build", flush=True)

    prefix = args.root_path.rstrip("/")
    url_path = f"{prefix}/" if prefix else "/"
    print(f"[serve] 文档: {cfg.get('doc_path')}  运行绑定: {cfg.get('allowed_runs')}", flush=True)
    print(f"[serve] http://127.0.0.1:{port}{url_path}", flush=True)
    for ip in _lan_ips():
        if args.host == "0.0.0.0":
            print(f"[serve] http://{ip}:{port}{url_path}", flush=True)
    if prefix:
        print("[serve] root_path 已设置:平台代理须'保留前缀转发';若代理会剥离前缀则不要传 --root-path", flush=True)
    print("[serve] Ctrl+C 停止", flush=True)

    cmd = [sys.executable, "-m", "result_viewer.backend.app",
           "--config", args.config, "--host", args.host, "--port", str(port)]
    if prefix:
        cmd += ["--root-path", prefix]
    sys.exit(subprocess.call(cmd, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
