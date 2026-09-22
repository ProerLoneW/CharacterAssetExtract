#!/usr/bin/env bash
# 使 viewer 常驻:配置改动后自动重载后端,配合前端轮询实现新实验热更新。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
config="$REPO_ROOT/result_viewer/viewer_config.json"
host="${RESULT_VIEWER_HOST:-}"
port="${RESULT_VIEWER_PORT:-0}"
root_path="${RESULT_VIEWER_ROOT_PATH:-}"
interval=3
python_bin="$(command -v python3 || command -v python)"

usage() {
  cat <<'EOF'
用法:
  scripts/serve_result_viewer_hot.sh [--host HOST] [--port PORT] [--config FILE]
                                      [--root-path PATH] [--interval SEC] [--python FILE]

端口和 host 未显式指定时从 viewer_config.json 读取。Ctrl+C 停止。
EOF
}

while (($#)); do
  case "$1" in
    --config) config="$2"; shift 2 ;;
    --host) host="$2"; shift 2 ;;
    --port) port="$2"; shift 2 ;;
    --root-path) root_path="$2"; shift 2 ;;
    --interval) interval="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] 未知参数: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$config" ]] || { echo "[error] viewer 配置不存在: $config" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "[error] Python 不可执行: $python_bin" >&2; exit 2; }
[[ -f "$REPO_ROOT/result_viewer/frontend_dist/index.html" ]] || {
  echo "[error] frontend_dist 缺失;请先执行: cd result_viewer/frontend && npm install && npm run build" >&2
  exit 2
}

cmd=("$python_bin" -u "$REPO_ROOT/result_viewer/supervise.py"
     --config "$config" --port "$port" --interval "$interval")
[[ -n "$host" ]] && cmd+=(--host "$host")
[[ -n "$root_path" ]] && cmd+=(--root-path "$root_path")

printf '[viewer] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
