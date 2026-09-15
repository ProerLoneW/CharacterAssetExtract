#!/bin/bash
# 模块② 包装脚本：调 teleAI-T2V 的 **镜头切片** 流水线。
#
# **只** 跑切镜（TransNetV2）。不跑 audio / base_filter_v2 / caption / dwpose 等。
#
# 实现：本仓库 m02_shots/teleai_src/ 下的 teleai_cut_only.py 调本地的
# main_split_fast（单次 TransNetV2 切镜 + 直接保留全部区间；**跳过**上游
# main_split_only 对每个 clip 的二次 TransNet 精修，valid_range 取全片），
# 不依赖仓库里的 video_filter_and_pack_new.py 那一坨 audio/caption imports。
# 仓库原文件**未修改**——本地副本 video_split.py 唯一改动是把
# save_cut_with_timeout 里的硬编码 /root/anaconda3/bin/python 替换为
# sys.executable（本机 conda python），保证 ffmpeg cut 子进程能跑通。
#
# 本地模型（transnetv2 权重）通过 config_test.json 里的 **绝对路径** 访问
# teleAI-T2V 仓库下的模型目录（用户偏好：保留上游模型所有权不动）：
#   video_split_model: /gemini/platform/public/aigc/lss/Interns/wyn/teleAI-T2V/local_models/transnetv2/transnetv2-pytorch-weights.pth
#
# 想跑全 pipeline（audio/base_filter/caption）：加 --full flag，
# 等价于切回原 teleAI 仓库路径跑 cut_v2 + audio + base_filter_v2。
#
# 用法：
#   bash run_teleai_pipeline.sh <teleAI-T2V根目录> <数据包目录> <原始视频目录> [配置文件名]
# 例：
#   bash run_teleai_pipeline.sh /gemini/platform/public/aigc/lss/Interns/wyn/teleAI-T2V \
#        /tmp/mov_001/teleai_pkg \
#        /tmp/mov_001/processed config_test.json
set -euo pipefail

TELEAI_ROOT="${1:?需要 teleAI-T2V 根目录}"
PKG_PATH="${2:?需要数据包目录}"
RAW_VIDEO_DIR="${3:?需要原始视频目录}"
CONFIG_NAME="${4:-config_test.json}"

# 默认仅切镜；--full 切回原全 pipeline
USE_LOCAL=1
for arg in "$@"; do
    case "$arg" in
        --full) USE_LOCAL=0 ;;
    esac
done

# 用 conda 环境（系统 python 通常没装 omegaconf）；允许 PYTHON 覆盖
CONDA_PY="/gemini/platform/public/aigc/lss/Interns/wyn/general_video_annot/bin/python"
if [ -x "$CONDA_PY" ]; then
    PYTHON="$CONDA_PY"
else
    PYTHON="${PYTHON:-python}"
fi

if [ ! -f "$TELEAI_ROOT/new_data_ppl_v1.0/$CONFIG_NAME" ]; then
    echo "❌ 配置文件 $TELEAI_ROOT/new_data_ppl_v1.0/$CONFIG_NAME 不存在"
    exit 1
fi
CONFIG_SRC="$TELEAI_ROOT/new_data_ppl_v1.0/$CONFIG_NAME"

mkdir -p "${PKG_PATH}"
if [ "$USE_LOCAL" -eq 1 ]; then
    # 本地模式：复制 config 进数据包后再改 target_path，**不改动** teleAI-T2V 仓库原文件
    CONFIG_PATH="${PKG_PATH}/config_cut_only.json"
    cp "$CONFIG_SRC" "$CONFIG_PATH"
else
    # 全流水线模式维持原行为：直接改仓库里的 config
    CONFIG_PATH="$CONFIG_SRC"
fi
# 把 config 里的 target_path 指向本数据包（保持与上游兼容）
sed -i -E "s|\"target_path\"[[:space:]]*:[[:space:]]*\".*\"|\"target_path\": \"$PKG_PATH\"|g" "$CONFIG_PATH"
# 注意：test_cut.json 由 teleai_cut_only.py 内部生成（它按 RAW_VIDEO_DIR 扫描）

mkdir -p "${PKG_PATH}/v2.0.0"
echo '{"cut_v2": false, "cut_only": false, "audio": false, "base_filter_v2": false}' > "${PKG_PATH}/v2.0.0/process_status.json"

if [ "$USE_LOCAL" -eq 1 ]; then
    # 走本仓库副本（推荐：只跑切镜，不引入仓库 audio 等大依赖链）
    LOCAL_SCRIPT="$(cd "$(dirname "$0")" && pwd)/teleai_src/teleai_cut_only.py"
    # SplitMinLength（默认75帧≈3秒）会过滤掉短镜头；设为 10 帧（≈0.4s）
    # 以保证所有切镜产物都进入 results。等价于 teleAI 上游同样的参数覆盖。
    export SplitMinLength="${SplitMinLength:-10}"
    "$PYTHON" "$LOCAL_SCRIPT" \
        --config "$CONFIG_PATH" \
        --input-dir "$RAW_VIDEO_DIR" \
        --output-dir "$PKG_PATH"
else
    # 走原仓库路径（run everything：cut_v2 + audio + base_filter_v2）
    cd "$TELEAI_ROOT/new_data_ppl_v1.0"
    "$PYTHON" gen_testjson.py "$RAW_VIDEO_DIR" "$PKG_PATH"
    sed -i -E "s|\"json_path\"[[:space:]]*:[[:space:]]*\".*\"|\"json_path\": \"$PKG_PATH/test_cut.json\"|g" "$CONFIG_PATH"
    "$PYTHON" video_filter_and_pack_new.py --config "$CONFIG_PATH" --owner wzh --commit_id xx \
        --mode "cut_v2,audio,base_filter_v2"
fi

echo "✅ teleAI-T2V 切镜完成：$PKG_PATH/v2.0.0/"
echo "下一步：用 m02_shots.importer 把 $PKG_PATH/v2.0.0/video/data/ 的镜头导入三级 Schema"