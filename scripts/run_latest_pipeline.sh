#!/usr/bin/env bash
# 从视频目录运行当前最新全流程,在 workspace/<result-name> 独立落盘,
# 终端实时显示输出并同步保存 console.log;成功后默认注册到 result_viewer。
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEFAULT_WORKSPACE_ROOT="$(cd -- "$REPO_ROOT/.." && pwd)/workspace"
DEFAULT_VIEWER_CONFIG="$REPO_ROOT/result_viewer/viewer_config.json"
DEFAULT_PIPELINE_PYTHON="/gemini/platform/public/aigc/lss/Interns/wyn/general_video_annot/bin/python"

usage() {
  cat <<'EOF'
用法:
  scripts/run_latest_pipeline.sh --input-dir <视频目录> --result-name <workspace 下的新目录名> [选项]

必填:
  --input-dir DIR       视频目录
  --result-name NAME    结果目录名;不能包含 /,同名时按 pipeline_state 断点续跑

常用选项:
  --asr-dir DIR         ASR 目录;省略时先尝试 <input-dir>/asr,再尝试视频旁的同名 JSON/JSONL
  --workspace-root DIR  结果父目录(默认为项目的 ../workspace)
  --viewer-config FILE  viewer 配置(默认 result_viewer/viewer_config.json)
  --python FILE         pipeline Python(默认 general_video_annot/bin/python)
  --no-viewer-register  成功后不注册到网页
  -h, --help            显示帮助

在 -- 之后可透传 run_full_pipeline.py 参数,例如:
  -- --caption-workers 3 --align-concurrency 2
  -- --force
EOF
}

input_dir=""
result_name=""
asr_dir=""
workspace_root="$DEFAULT_WORKSPACE_ROOT"
viewer_config="$DEFAULT_VIEWER_CONFIG"
python_bin="$DEFAULT_PIPELINE_PYTHON"
register_viewer=1
pipeline_args=()

while (($#)); do
  case "$1" in
    --input-dir)
      [[ $# -ge 2 ]] || { echo "[error] --input-dir 缺少参数" >&2; exit 2; }
      input_dir="$2"; shift 2 ;;
    --result-name|--run-name)
      [[ $# -ge 2 ]] || { echo "[error] $1 缺少参数" >&2; exit 2; }
      result_name="$2"; shift 2 ;;
    --asr-dir)
      [[ $# -ge 2 ]] || { echo "[error] --asr-dir 缺少参数" >&2; exit 2; }
      asr_dir="$2"; shift 2 ;;
    --workspace-root)
      [[ $# -ge 2 ]] || { echo "[error] --workspace-root 缺少参数" >&2; exit 2; }
      workspace_root="$2"; shift 2 ;;
    --viewer-config)
      [[ $# -ge 2 ]] || { echo "[error] --viewer-config 缺少参数" >&2; exit 2; }
      viewer_config="$2"; shift 2 ;;
    --python)
      [[ $# -ge 2 ]] || { echo "[error] --python 缺少参数" >&2; exit 2; }
      python_bin="$2"; shift 2 ;;
    --no-viewer-register)
      register_viewer=0; shift ;;
    -h|--help)
      usage; exit 0 ;;
    --)
      shift; pipeline_args=("$@"); break ;;
    *)
      echo "[error] 未知参数: $1(需透传给 pipeline 的参数请放在 -- 之后)" >&2
      usage >&2
      exit 2 ;;
  esac
done

[[ -n "$input_dir" ]] || { echo "[error] 必须传 --input-dir" >&2; usage >&2; exit 2; }
[[ -n "$result_name" ]] || { echo "[error] 必须传 --result-name" >&2; usage >&2; exit 2; }
[[ "$result_name" != "." && "$result_name" != ".." && "$result_name" != */* ]] || {
  echo "[error] --result-name 只能是 workspace 下的单个目录名,不能包含 /: $result_name" >&2
  exit 2
}
[[ -d "$input_dir" ]] || { echo "[error] 视频目录不存在: $input_dir" >&2; exit 2; }
[[ -x "$python_bin" ]] || python_bin="$(command -v python3 || command -v python)"
[[ -n "$python_bin" && -x "$python_bin" ]] || { echo "[error] 找不到可用 Python" >&2; exit 2; }

input_dir="$(cd -- "$input_dir" && pwd)"
mkdir -p -- "$workspace_root"
workspace_root="$(cd -- "$workspace_root" && pwd)"
output_root="$workspace_root/$result_name"
mkdir -p -- "$output_root"

asr_args=()
if [[ -n "$asr_dir" ]]; then
  [[ -d "$asr_dir" ]] || { echo "[error] ASR 目录不存在: $asr_dir" >&2; exit 2; }
  asr_args=(--asr-dir "$(cd -- "$asr_dir" && pwd)")
elif [[ -d "$input_dir/asr" ]]; then
  asr_args=(--asr-dir "$input_dir/asr")
fi

viewer_args=()
if ((register_viewer)); then
  [[ -f "$viewer_config" ]] || { echo "[error] viewer 配置不存在: $viewer_config" >&2; exit 2; }
  viewer_config="$(cd -- "$(dirname -- "$viewer_config")" && pwd)/$(basename -- "$viewer_config")"
  viewer_args=(--update-viewer-config "$viewer_config")
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://tcp30000-1058.default.svc.cluster.local:30000/v1}"
export MODEL_NAME="${MODEL_NAME:-MiMo-V2.5}"
sam3_src="$REPO_ROOT/../third_party/sam3_src"
if [[ -d "$sam3_src" ]]; then
  export PYTHONPATH="$sam3_src${PYTHONPATH:+:$PYTHONPATH}"
fi

log_file="$output_root/console.log"
exec > >(tee -a "$log_file") 2>&1

echo "[launcher] input_dir=$input_dir"
echo "[launcher] output_root=$output_root"
if ((${#asr_args[@]})); then
  echo "[launcher] asr_dir=${asr_args[1]}"
else
  echo "[launcher] asr_dir=未显式指定,将查找视频旁的同名 JSON/JSONL"
fi
echo "[launcher] console_log=$log_file"
if ((register_viewer)); then
  echo "[launcher] viewer_config=$viewer_config"
fi

cmd=("$python_bin" -u "$REPO_ROOT/run_full_pipeline.py"
     --input-dir "$input_dir"
     --output-root "$output_root"
     "${asr_args[@]}"
     "${viewer_args[@]}"
     "${pipeline_args[@]}")
printf '[launcher] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if "${cmd[@]}"; then
  status=0
  echo "[launcher] SUCCESS: $output_root"
  if ((register_viewer)); then
    echo "[launcher] viewer 已注册;supervise.py 守护时约 3-6 秒后出现在网页运行下拉框。"
  fi
else
  status=$?
  echo "[launcher] FAILED(exit=$status): 请查看 $log_file"
fi
exit "$status"
