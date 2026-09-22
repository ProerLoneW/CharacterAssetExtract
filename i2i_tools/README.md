# 多图资产高清还原（UniAPI）

这组工具把人物、场景、关键物品的低质量提取图作为多张参考图，通过 UniAPI 的
Gemini 原生多模态接口还原为一张高质量参考资产。请求在同一个 `contents.parts` 中按
顺序传入多张 `inlineData` 图片和一段文本 prompt，并显式请求 `IMAGE` 输出。

## 文件

- `restore_character_reference.py`：人物身份、脸、体型和服装一致性优先，默认竖图。
- `restore_character_three_view_and_costume.py`：从 Stage4 角色视觉资产库读入每个
  appearance state 的脸+身体参考图，调用 Gemini 同时还原（1）三视图（3:4，
  front/3-4/side/back 的 turnaround）和（2）服饰参考（1:1，把整套服饰平铺在中性背景
  上）。输出 `reference/<CHR>/<APP>/<APP>_three_view.png` 和
  `<APP>_costume.png`，并附带 `.json` 溯源。
- `extract_character_clothing.py`：输入一张或多张角色参考图，分两次（或一次性两张）调
  用 Gemini 提取角色的上衣（top）和裤子（pants），每张输出都是无背景（alpha = 0）PNG，
  可直接做下游合成。详见下文「角色服装拆分（透明背景）」。
- `restore_scene_reference.py`：空间结构、陈设、视角一致性与去人补全优先，默认横图。
- `restore_key_object_reference.py`：物体几何、材质和剧情标记一致性优先，默认方图。
- `uniapi_gemini_image_client.py`：默认客户端；Gemini 原生多图+文本请求、重试、响应解析和元数据。
- `uniapi_image_client.py`：保留的 OpenAI Images `/v1/images/edits` 兼容客户端。

三个任务脚本默认调用：

```text
POST /gemini/v1beta/models/gemini-3-pro-image:generateContent
```

默认模型 `gemini-3-pro-image` 面向高质量资产生成，支持图文输入和图像输出；默认输出
`2K`，需要最终高分辨率资产时可使用 `--image-size 4K`。单次最多传 14 张参考图，但参考
图不是越多越好：人物通常选 3–5 张互补且身份清晰的图，关键物品通常选不超过 6 张最
可信的视角，避免低质量或错误帧稀释身份/结构约束。

## 模型选择

下面四个模型使用相同的多图+prompt 输入代码，只需更换 `--model`。默认
`--image-size auto` 会自动选择该模型支持的输出尺寸。

| 模型 ID | 定位 | 自动尺寸 | 建议用途 |
| --- | --- | --- | --- |
| `gemini-3-pro-image` | 最高质量、复杂指令、身份/细节保持 | 2K，可设 4K | 最终人物、场景和关键物品母版 |
| `gemini-3.1-flash-image` | 质量、速度和成本平衡 | 2K，可设 4K | 批量生产，推荐作为常规默认候选 |
| `gemini-3.1-flash-lite-image` | 最低延迟和成本 | 1K，仅支持 1K | 大批量预览、候选筛选 |
| `gemini-2.5-flash-image` | 旧版快速模型 | 固定 1K，不发送尺寸字段 | 兼容旧任务，参考图最好不超过 3 张 |

例如只切换模型，其他输入完全不变：

```bash
python i2i_tools/restore_character_reference.py \
  --images face.jpg body.jpg side.jpg \
  --prompt '保持人物身份与服装一致，恢复清晰完整参考图' \
  --model gemini-3.1-flash-image \
  --output outputs/CHR001_reference.png
```

也可以通过环境变量统一切换整个流水线：

```bash
export UNIAPI_GEMINI_IMAGE_MODEL='gemini-3.1-flash-image'
```

## 环境变量

```bash
export UNIAPI_API_KEY='你的 token'
# 可选：
export UNIAPI_GEMINI_BASE_URL='https://api.uniapi.io/gemini/v1beta'
export UNIAPI_GEMINI_IMAGE_MODEL='gemini-3-pro-image'
```

API key 只从环境变量读取，不会进入命令历史或元数据。

## 当前全流程接入

`run_full_pipeline.py` 默认在人物物化、物体参考集和场景选帧完成后调用
`run_pipeline_assets.py`。这个批处理入口直接适配当前主线布局，不依赖旧的
`role_to_chr` / `i2i_bundles`：

```bash
python i2i_tools/run_pipeline_assets.py \
  --workspace ../workspace/<run>/<video> \
  --dry-run
```

默认按人物轨迹数、物体有效参考数和场景重要度各取 3 项，输出到
`<workspace>/i2i_test/<model>/`，并写 `i2i_test/manifest.json`。各上限设为 `0`
才会全量生成；全量会显著增加调用数量和成本。统一编排可用 `--no-i2i`
关闭。批处理兼容本地旧变量名 `UNIAPI_KEY`，但新配置仍推荐
`UNIAPI_API_KEY`；key 不写入结果。

物体多参考请求如果返回空 `candidates[]`，批处理会自动用原始 crop 中文件最大的
一张做一次有界重试。走到该分支时，manifest 会记录 `fallback_used`、
`fallback_inputs` 和首次请求错误，便于复核小物体或严重遮挡物体的降级结果。

## 用法

人物（建议把最可靠的正脸/全身图放第一张）：

```bash
python i2i_tools/restore_character_reference.py \
  --images face.jpg body_front.jpg body_side.jpg \
  --asset-description '同一女性角色，黑色短发，深蓝夹克' \
  --prompt '保持原片写实风格，输出完整全身参考图' \
  --output outputs/CHR001_reference.png
```

场景（主锚点放第一张，辅助视角随后）：

```bash
python i2i_tools/restore_scene_reference.py \
  --images anchor_main.jpg anchor_aux_01.jpg anchor_aux_02.jpg \
  --prompt '清除人物，保持卧室布局和午后光线' \
  --output outputs/SCENE_001_reference.png
```

关键物品（优先使用 crop/rgba，多时刻或多视角一起输入）：

```bash
python i2i_tools/restore_key_object_reference.py \
  --images crop_f000.jpg crop_f028.jpg rgba_f057.png \
  --asset-description '盘中食物被摆成人脸，是剧情关键物品' \
  --prompt '保留食物摆放关系和盘子材质，完整展示物品' \
  --output outputs/OBJ001_reference.png
```

先用 `--dry-run` 校验图片、查看最终 prompt 和请求参数，不会读取 key 或发起请求：

```bash
python i2i_tools/restore_character_reference.py \
  --images a.jpg b.jpg --prompt '保持角色一致' \
  --output out.png --dry-run
```

## 角色三视图 + 服饰参考（批量）

给定 Stage4 输出的角色视觉资产库（每个角色按 `appearance_state_id` 分组），批量生
成三视图与服饰参考：

```bash
export UNIAPI_API_KEY='你的 token'

python i2i_tools/restore_character_three_view_and_costume.py \
  --character-root workspace/movie_full_pipeline_20260912_v3/characters/characters \
  --output-root workspace/movie_full_pipeline_20260912_v3/characters/reference \
  --character CHR002 --character CHR005 \
  --model gemini-3-pro-image --image-size 2K
```

每个 `(CHR, APP)` 对会输出两张图 + 两份溯源 JSON：

```text
reference/CHR002/CHR002_APP001/CHR002_APP001_three_view.png
reference/CHR002/CHR002_APP001/CHR002_APP001_three_view.png.json
reference/CHR002/CHR002_APP001/CHR002_APP001_costume.png
reference/CHR002/CHR002_APP001/CHR002_APP001_costume.png.json
```

不指定 `--character` 时处理全部角色；先用 `--dry-run` 校验图片选择和提示词再发起
请求。

```bash
python i2i_tools/restore_character_reference.py \
  --images a.jpg b.jpg --prompt '保持角色一致' \
  --output out.png --dry-run
```

默认生成 `2K` 图片，并在图片旁写出 `.json` 溯源文件（输入图 SHA-256、最终 prompt、
模型、usage、模型返回文本和输出图 SHA-256）。已有输出默认不覆盖；确认覆盖时使用
`--overwrite`。人物、场景、物品的默认宽高比分别为 `2:3`、`3:2`、`1:1`，可用
`--aspect-ratio` 覆盖。

常用高级参数可通过 `--help` 查看，包括 `--model`、`--image-size`、`--aspect-ratio`、
`--image-only`、`--temperature`、`--timeout` 和 `--max-retries`。

三个任务文件也分别导出了同名 Python 函数，可以在流水线中直接调用；先构造
`UniAPIGeminiImageClient`，再传入 `image_paths/prompt/output_path`。函数返回
`GeminiRestorationResult`，其中包含最终 prompt、输入图信息、模型返回文本、API 原始
响应和已保存图片。原 `UniAPIImageClient` 与 `restore_reference` 仍可用于必须走
`/v1/images/edits` 的兼容模型。

## 角色服装拆分（透明背景）

`extract_character_clothing.py` 接受 1..N 张同一角色的参考图，分别（也可以一次性合
成）提取 **上衣** 和 **裤子**，输出 PNG 要求背景完全透明（alpha = 0），可以直接用
作下游合成素材。

默认生成两张独立 PNG：

- `out_top.png`：上衣（jacket / coat / shirt / sweater / vest / hoodie），1:1。
- `out_pants.png`：裤子（trousers / jeans / cargo pants / shorts / skirt），1:1。

如果 `--output` 以 `.png` 结尾（如 `--output outputs/CHR001.png`），拆分会写到
`outputs/CHR001_top.png` / `outputs/CHR001_pants.png`；否则把 `--output` 当作目录，
里面写 `top.png` / `pants.png`。每张图旁边都会落一份 `<file>.png.json` 溯源。

```bash
export UNIAPI_API_KEY='你的 token'

# 默认：拆 top + pants（两次生成）
python i2i_tools/extract_character_clothing.py \
  --images character_a.jpg character_b.jpg character_c.jpg \
  --output outputs/CHR001.png \
  --character-label CHR001

# 一次只拆上衣
python i2i_tools/extract_character_clothing.py \
  --images character_a.jpg character_b.jpg \
  --output outputs/CHR001.png \
  --parts top

# 一次性合一张：上半 top + 下半 pants（2:3，每一半各自透明背景）
python i2i_tools/extract_character_clothing.py \
  --images character_a.jpg character_b.jpg \
  --output outputs/CHR001.png \
  --parts combined

# 切换模型 / 分辨率
python i2i_tools/extract_character_clothing.py \
  --images character_a.jpg character_b.jpg \
  --output outputs/CHR001.png \
  --model gemini-3.1-flash-image --image-size 2K

# 先看 prompt 和输入，不会读 key 也不会发请求
python i2i_tools/extract_character_clothing.py \
  --images character_a.jpg character_b.jpg \
  --output outputs/CHR001.png \
  --dry-run
```

`--parts` 接受逗号分隔的任意子集：`top`、`pants`、`combined`，可同时指定多个，例如
`--parts top,combined` 会同时输出上衣独立 PNG 和一张上下拼接的合并 PNG。

程序化入口：

```python
from i2i_tools import (
    UniAPIGeminiImageClient,
    extract_character_top,
    extract_character_pants,
    extract_character_combined,
    plan_clothing_extraction,
)

client = UniAPIGeminiImageClient(api_key)
plan = plan_clothing_extraction(
    ["character_a.jpg", "character_b.jpg"],
    output="outputs/CHR001.png",
    parts=["top", "pants"],
    character_label="CHR001",
)
top_result = extract_character_top(
    client,
    image_paths=plan.image_paths,
    output_path=plan.top_output,
)
pants_result = extract_character_pants(
    client,
    image_paths=plan.image_paths,
    output_path=plan.pants_output,
)
```

`extract_character_top` / `extract_character_pants` / `extract_character_combined` 都
接受 `**generation_options`，可传入 `model`、`image_size`、`temperature`、
`aspect_ratio` 等与 `restore_gemini_reference` 相同的字段。

## 角色服装拆分（OpenAI / GPT-Image 系列）

`extract_character_clothing.py` 现在支持 `--provider openai` 走
`UniAPIImageClient`（UniAPI 的 OpenAI `/v1/images/edits` 接口），模型可选
`gpt-image-2`、`gpt-image-2.5-sunburst` 或 `gpt-image-1`。可用 `--base-url`
覆盖端点（默认 `https://api.uniapi.io/v1`），输出仍然是 `1:1` 透明背景的
上衣和裤子 PNG，并由 `bake_alpha` 后处理真正写入 RGBA。

```bash
export UNIAPI_API_KEY='你的 token'

# gpt-image-2 直接做服装拆分
python i2i_tools/extract_character_clothing.py \
  --provider openai --model gpt-image-2 \
  --images character_a.jpg character_b.jpg \
  --output outputs/CHR001_openai.png \
  --bake-alpha

# gpt-image-2.5-sunburst
python i2i_tools/extract_character_clothing.py \
  --provider openai --model gpt-image-2.5-sunburst \
  --images character_a.jpg character_b.jpg \
  --output outputs/CHR001_openai25.png \
  --bake-alpha
```

也有独立的 OpenAI 入口模块 `extract_character_clothing_openai.py`，命令行
用法一致；程序化入口为 `extract_character_top_openai`、
`extract_character_pants_openai`、`extract_character_combined_openai` 和
`execute_plan_openai`。

## 两阶段服装提取（先做人物三视图，再做服装拆分）

`two_stage_clothing_pipeline.py` 把原来“直接拿视频裁剪图给模型做服装拆分”
的单阶段流程拆成两步：

1. **Stage 1**：把原视频里裁剪出的人物全身照作为参考，调用所选模型生成
   一张高质量的人物三视图参考图（默认 3:4 竖图，复用
   `restore_character_three_view_and_costume` 的 `THREE_VIEW_INSTRUCTIONS`）。
2. **Stage 2**：把 Stage 1 生成的三视图（作为高质量身份/服装锚点）+ 原图
   一起送回同一个模型，要求输出 1:1 透明背景的上衣 / 裤子。

每个模型都会得到独立目录，所有输出都带 JSON 溯源。脚本接受 5 个模型，
默认全部执行，也可用 `--model` 选一个或多个：

```text
gemini-3-pro-image           (provider=gemini)
gemini-3.1-flash-image       (provider=gemini)
gemini-3.1-flash-lite-image  (provider=gemini)
gpt-image-2                  (provider=openai)
gpt-image-2.5-sunburst       (provider=openai)
```

完整示例（CHR004 的 3 张全身裁剪）：

```bash
export UNIAPI_API_KEY='你的 token'

# 默认跑全部 5 个模型
python i2i_tools/two_stage_clothing_pipeline.py \
  --images workspace/i2i_test/CHR004_BODY_0007.jpg \
            workspace/i2i_test/CHR004_BODY_0009.jpg \
            workspace/i2i_test/CHR004_BODY_0010.jpg \
  --character-label CHR004 \
  --output-root workspace/two_stage_clothing/CHR004

# 只跑一个模型
python i2i_tools/two_stage_clothing_pipeline.py \
  --images ... --character-label CHR004 \
  --output-root workspace/two_stage_clothing/CHR004 \
  --model gpt-image-2.5-sunburst

# 选两到三个模型
python i2i_tools/two_stage_clothing_pipeline.py \
  --images ... --character-label CHR004 \
  --output-root workspace/two_stage_clothing/CHR004 \
  --model gemini-3-pro-image --model gpt-image-2

# 先看每个模型的 plan（不发请求、不读 key）
python i2i_tools/two_stage_clothing_pipeline.py \
  --images ... --character-label CHR004 \
  --output-root workspace/two_stage_clothing/CHR004 \
  --dry-run
```

输出结构：

```text
<output-root>/<model>/<character>_three_view.png
<output-root>/<model>/<character>_three_view.png.json
<output-root>/<model>/<character>_top.png
<output-root>/<model>/<character>_top.png.png.json
<output-root>/<model>/<character>_pants.png
<output-root>/<model>/<character>_pants.png.png.json
```

`--bake-alpha` 启用后会把模型返回的非透明 PNG 转成 RGBA，方便下游合成。
同一脚本同时驱动 Gemini 和 OpenAI 模型，因此可以直接横向对比不同模型
在两阶段流程下的拆衣效果。
