# 服装提取流程 — Prompt 与协议整理

本文档汇总 `i2i_tools/` 下服装提取相关流程的 **所有 prompt 模板**、**模型配置**、
**输入约束** 与 **后处理步骤**，方便后续调 prompt / 调模型 / 写新接入时查阅。

源代码：
- `i2i_tools/extract_character_clothing.py`（单阶段：直接拿原图拆衣）
- `i2i_tools/extract_character_clothing_openai.py`（单阶段 OpenAI 版）
- `i2i_tools/two_stage_clothing_pipeline.py`（两阶段：先三视图，再拆衣）
- `i2i_tools/uniapi_gemini_image_client.py`（Gemini `generateContent` 客户端）
- `i2i_tools/uniapi_image_client.py`（OpenAI `/v1/images/edits` 客户端）
- `i2i_tools/restore_character_three_view_and_costume.py`（Stage 1 用到的三视图 prompt）

---

## 1. 单阶段流程（直接拆衣）

### 1.1 输入

- 1..N 张 **同一角色、同一套衣服** 的参考图，按身份清晰度从最强到最弱排序；
  脚本里 `inspect_images()` 会拒掉非 PNG/JPEG/WEBP / magic bytes 不匹配的文件。
- 默认上限 14 张（Gemini `MAX_REFERENCE_IMAGES`），实际建议 3–6 张。

### 1.2 输出

由 `--parts` 决定，可以是 `top`、`pants`、`combined` 的任意子集：

- `top`：`{output}_top.png` — 1:1 上衣
- `pants`：`{output}_pants.png` — 1:1 裤子
- `combined`：`{output}_combined.png` — 2:3 上下两半（每半各自透明背景）

最终 PNG 都是 RGBA。Gemini 模型默认输出 `image/jpeg` 或 RGB `image/png`，
本流程对所有 `image/*` 都接受，写到 `.png` 后用 Pillow `bake_alpha` 把
背景灰卡色（用边界像素中位数估出来）擦成 alpha=0；OpenAI 模型原生支持
`background=transparent`，输出已经带 alpha，但脚本仍走同一个 `bake_alpha`
后处理以便跨模型结果一致。

### 1.3 协议差异

| Provider | 端点 | Auth Header | 请求体 | 响应字段 |
| --- | --- | --- | --- | --- |
| Gemini | `POST {base}/v1beta/models/{model}:generateContent` | `x-goog-api-key` | `{contents:[{role:user, parts:[{inlineData:…},…,{text:prompt}]}], generationConfig:{responseModalities:[…], imageConfig:{aspectRatio, imageSize}}}` | `candidates[0].content.parts[]` 里的 `inlineData` / `text` |
| OpenAI | `POST {base}/v1/images/edits` (multipart) | `Authorization: Bearer` | 表单字段 `model / prompt / image[] / size / quality / n / output_format / input_fidelity / background / moderation / output_compression` | `{data: [{b64_json | url, revised_prompt}]}` |

### 1.4 Prompt 三段式

所有 prompt 通过 `compose_prompt()` 组装：

```text
[1] task_instructions   ← 角色 + 硬约束（透明背景、禁止改动等）
[2] asset_description   ← "CHR004: extracting top, pants from 3 input image(s)."
[3] user_prompt         ← 短指令（“Lay out this character's … on alpha=0…”）
```

#### 1.4.1 `TOP_INSTRUCTIONS`（上衣）

```text
You are a professional costume asset extractor. Treat all supplied images as
observations of the SAME person wearing the SAME upper-body garment. The first
image is the primary garment reference; later images contribute complementary
fabric, color, pattern, layering, and accessory evidence.

The output must be a single production garment plate showing ONLY the
upper-body garment the character is wearing in the references — typically a
jacket, coat, shirt, sweater, vest, or hoodie. The garment is shown laid out
flat, fully spread and unoccluded, as if prepared for a costume reference.

[5 条硬约束 — 见 § 1.4.4]
```

#### 1.4.2 `PANTS_INSTRUCTIONS`（裤子）

```text
You are a professional costume asset extractor. Treat all supplied images as
observations of the SAME person wearing the SAME lower-body garment. The first
image is the primary garment reference; later images contribute complementary
fabric, color, pattern, layering, belt-loop, pocket, and footwear-above-the
-hem evidence.

The output must be a single production garment plate showing ONLY the
lower-body garment the character is wearing in the references — typically
trousers, jeans, cargo pants, shorts (treated as pants for this task), or a
skirt. The garment is shown laid out flat, fully spread and unoccluded, as
if prepared for a costume reference. The waistband is at the top of the
canvas, the hem at the bottom, and any belt, drawstring, or waist tie is
preserved.

[5 条硬约束 — 见 § 1.4.4]
```

#### 1.4.3 `COMBINED_INSTRUCTIONS`（上下两半组合）

```text
You are a professional costume asset extractor. Treat all supplied images as
observations of the SAME person wearing the SAME complete outfit. The first
image is the primary reference; later images contribute complementary fabric,
color, pattern, and layering evidence.

The output must be a single composite production plate showing ONLY the two
requested garments, both isolated and both with fully transparent backgrounds
(alpha = 0 everywhere except the garment pixels). The plate is split into two
halves:

* Top half — the upper-body garment (jacket / coat / shirt / sweater / vest
  / hoodie) laid out flat and fully spread.
* Bottom half — the lower-body garment (trousers / jeans / cargo pants /
  shorts / skirt) laid out flat and fully spread, waistband at the top of its
  half, hem at the bottom.

Both halves share the same canvas but each garment is on its OWN transparent
background; the two halves do not overlap and there is no shared card, table,
ground, or shadow between them. The plate is intended for downstream
compositing, not as a worn-by-character photo.

[5 条硬约束 — 见 § 1.4.4]
```

#### 1.4.4 `_CLOTHING_GUARDRAILS`（硬约束，三段共享）

```text
Hard requirements for the output PNG:

* The garment fills the central ~85% of the canvas; it is fully visible with a
  small even margin on all four sides.
* The background MUST be fully transparent (alpha = 0 everywhere except the
  garment pixels). No checkerboard, white card, grey card, gradient, shadow,
  ground, table, mannequin, or environmental detail.
* The garment keeps its exact silhouette, fabric texture, color, pattern,
  print, embroidery, seam, hem, fold, layering, and any story-relevant wear
  (dirt, tear, blood, scorch, repair).
* Recover fine detail lost to blur, compression, low resolution, motion,
  partial occlusion, and harsh lighting — but never redesign, restyle,
  recolor, add accessories, add a second identical garment, or invent text
  or logos that cannot be verified from the references.
* The output is a single garment asset, not a collage, contact sheet,
  fashion editorial, or worn-by-character photo. Do not include the
  character's face, hair, skin, hands, feet, body, mannequin, props, labels,
  borders, or watermarks.
```

#### 1.4.5 `user_prompt` 三段（短指令，紧跟 task_instructions 之后）

上衣：

```text
Lay out this character's upper-body garment flat, fully spread and
unoccluded, on a fully transparent background (alpha = 0). Show only the
top — no body, no pants, no shoes, no mannequin, no props, no card, no
shadow. Preserve exact fabric, color, pattern, and condition.
```

裤子：

```text
Lay out this character's lower-body garment flat, fully spread and
unoccluded, on a fully transparent background (alpha = 0). Show only the
pants — no body, no top, no shoes, no mannequin, no props, no card, no
shadow. Waistband at the top of the canvas, hem at the bottom. Preserve
exact fabric, color, pattern, and condition.
```

组合（两半独立 alpha=0）：

```text
Produce a two-half plate: the top half shows the upper-body garment laid
flat, the bottom half shows the lower-body garment laid flat (waistband up,
hem down). Each half is on its OWN fully transparent background (alpha = 0);
do not draw a shared card, table, ground, or shadow. Preserve exact
fabrics, colors, patterns, and condition for both garments.
```

#### 1.4.6 `asset_description`（自动生成）

```text
{character_label}: extracting {parts} from {N} input image(s).
```

例如：`CHR004: extracting pants, top from 3 input image(s).`

### 1.5 调用的函数

| Provider | Helper | 输出 |
| --- | --- | --- |
| Gemini | `extract_character_top(client, image_paths=…, output_path=…, **gen)` | `GeminiRestorationResult` |
| Gemini | `extract_character_pants(client, …)` | `GeminiRestorationResult` |
| Gemini | `extract_character_combined(client, …)` | `GeminiRestorationResult` |
| OpenAI | `extract_character_top_openai(client, …)` | `RestorationResult` |
| OpenAI | `extract_character_pants_openai(client, …)` | `RestorationResult` |
| OpenAI | `extract_character_combined_openai(client, …)` | `RestorationResult` |

### 1.6 关键生成参数

| 参数 | Gemini 默认 | OpenAI 默认 |
| --- | --- | --- |
| 尺寸 | 由 `--image-size` 控制；`auto` → Pro/Flash `2K`, Lite `1K`, Legacy 不发 | `auto` → 由 aspect 推断：`1:1 → 1024x1024`, `2:3 → 1024x1536`, `3:4 → 1024x1536` |
| 长宽比 | `1:1`（top/pants）、`2:3`（combined） | 客户端按相同规则自动映射 |
| 质量 | n/a | `high` |
| 输入保真度 | n/a | `None`（profile 自动决定；`gpt-image-1` 支持，其余被剥离） |
| 背景 | `transparent`（prompt 要求 + 后处理 alpha bake） | `transparent`（multipart 字段） |
| 输出格式 | 强制 `image/png`（脚本容忍 jpeg，写入 `.png`） | `png` |
| moderation | n/a | `low`（默认；调高易触发 Azure 安全拦截） |

---

## 2. 两阶段流程（先三视图，再拆衣）

`two_stage_clothing_pipeline.py` 把"原图直拆"拆成两阶段：

```
原视频裁剪的人物全身照 ──► Stage 1 ──► 高质量人物三视图 (3:4 / 1024x1536)
                                              │
                                              └─► Stage 2 ──► top (1:1, RGBA)
                                                            └─► pants (1:1, RGBA)
```

Stage 2 输入顺序固定：先放 Stage 1 生成的三视图（身份/服装锚点），再追加原图
作为补充证据。每个模型独立目录：

```
<output-root>/<model>/<character>_three_view.png
<output-root>/<model>/<character>_top.png
<output-root>/<model>/<character>_pants.png
（每张图 + .json 溯源）
```

### 2.1 Stage 1：三视图 / Turnaround

#### 2.1.1 `task_instructions = THREE_VIEW_INSTRUCTIONS`

```text
Create one high-quality, photorealistic character turnaround sheet by
reconciling all supplied images as observations of the SAME person wearing the
SAME costume. The first image defines the primary identity reference; later
images contribute complementary face, hair, body proportion, wardrobe, and
material evidence.

The output must be a single, production-grade character reference showing
EXACTLY this one person in EXACTLY this costume, rendered as a clean
three-quarter / side / back / front turnaround on a plain neutral background
with even, studio-like illumination. The four poses must read as the same
person: identical identity (face, skin, hair, age, ethnicity), identical
body type and posture proportions, identical wardrobe (every garment,
accessory, color, pattern, texture, and footwear). Recover fine detail lost
to blur, compression, low resolution, motion, crop, occlusion, and harsh
lighting — but never redesign, restyle, swap outfits, change age, change
ethnicity, or invent accessories. The reference is not a collage, storyboard,
poster, or dramatic scene; do not add text, labels, logos, borders,
watermarks, extra people, or duplicated limbs.
```

#### 2.1.2 `user_prompt`

```text
Produce a clean character turnaround/three-view reference sheet for this
character. Keep identity, hairstyle, body proportions and complete outfit
consistent across all poses. The image is intended as the single
highest-quality reference that downstream pipelines can use to extract
individual garments.
```

#### 2.1.3 `asset_description`

```text
Stage 1: building a high-quality three-view turnaround reference for the
character from {N} full-body crops.
```

#### 2.1.4 输出参数

| Provider | aspect | size |
| --- | --- | --- |
| Gemini | `3:4` | `--image-size`（默认 `auto` → Pro/Flash `2K`, Lite `1K`） |
| OpenAI | `3:4` | `_size_for_aspect("3:4", model, …)` → `1024x1536` |

背景：OpenAI 走 `background=auto`；Gemini 不显式控制，prompt 已要求 plain
neutral background。

### 2.2 Stage 2：上衣 + 裤子（用 Stage 1 三视图当锚）

#### 2.2.1 输入顺序

```text
[1] Stage 1 生成的三视图（高质量身份/服装锚）
[2] 原视频裁剪的全身照 1
[3] 原视频裁剪的全身照 2
…
```

总数不要超过 Gemini 的 14 张上限。

#### 2.2.2 `task_instructions`

上衣用 `TOP_INSTRUCTIONS`，裤子用 `PANTS_INSTRUCTIONS`（同 § 1.4.1 / 1.4.2）。

#### 2.2.3 `user_prompt`

上衣：

```text
Lay out this character's upper-body garment flat, fully spread and
unoccluded, on a fully transparent background (alpha = 0). Show only the
top — no body, no pants, no shoes, no mannequin, no props, no card, no
shadow. Preserve exact fabric, color, pattern, and condition.
```

裤子：

```text
Lay out this character's lower-body garment flat, fully spread and
unoccluded, on a fully transparent background (alpha = 0). Show only the
pants — no body, no top, no shoes, no mannequin, no props, no card, no
shadow. Waistband at the top of the canvas, hem at the bottom. Preserve
exact fabric, color, pattern, and condition.
```

#### 2.2.4 `asset_description`

```text
Stage 2: extracting the {upper|lower}-body garment using the high-quality
turnaround plus the original full-body crops.
```

### 2.3 调用的函数

| 阶段 | Gemini | OpenAI |
| --- | --- | --- |
| Stage 1 | `restore_gemini_reference(client, image_paths=原图, …, generation_options={model, aspect_ratio:"3:4", image_size})` | `restore_reference(client, image_paths=原图, …, edit_options={model, size:"1024x1536", quality:"high", n:1, output_format:"png", input_fidelity:"high", background:"auto"})` |
| Stage 2 top | `restore_gemini_reference(…, image_paths=[三视图, *原图], user_prompt=_stage2_top_user_prompt(), task_instructions=TOP_INSTRUCTIONS, …)` | `extract_character_top_openai(…, image_paths=[三视图, *原图], aspect_ratio="1:1", moderation="low", …)` |
| Stage 2 pants | `restore_gemini_reference(…, task_instructions=PANTS_INSTRUCTIONS, …)` | `extract_character_pants_openai(…, aspect_ratio="1:1", moderation="low", …)` |

---

## 3. Provider × 模型矩阵

`ALL_MODELS = (gemini-3-pro-image, gemini-3.1-flash-image, gemini-3.1-flash-lite-image, gpt-image-2, gpt-image-2.5-sunburst)`

| 模型 | Provider | Base URL | 默认 size | input_fidelity | moderation | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| `gemini-3-pro-image` | gemini | `https://api.uniapi.io/gemini/v1beta` | `2K` | n/a | n/a | 最高质量，速度最慢 |
| `gemini-3.1-flash-image` | gemini | 同上 | `2K` | n/a | n/a | 质量/速度/价格平衡 |
| `gemini-3.1-flash-lite-image` | gemini | 同上 | `1K`（仅支持 `1K`） | n/a | n/a | 1K 锁死；alpha bake 效果弱 |
| `gpt-image-2` | openai | `https://api.uniapi.io/v1` | `1024x1024` / `1024x1536` / `1536x1024` | ✗（profile 剥离） | `low` | 上游 Azure moderation 偶尔拦截 pants，重试即可 |
| `gpt-image-2.5-sunburst` | openai | 同上 | 同上 | ✗（profile 剥离） | `low` | 同上 |

环境变量：

```bash
export UNIAPI_API_KEY='sk-…'                 # 必填
export UNIAPI_GEMINI_BASE_URL='…'           # 可选，默认 https://api.uniapi.io/gemini/v1beta
export UNIAPI_OPENAI_BASE_URL='…'            # 可选，默认 https://api.uniapi.io/v1
export UNIAPI_GEMINI_IMAGE_MODEL='…'         # 可选，全局 Gemini 默认模型
export UNIAPI_OPENAI_IMAGE_MODEL='…'         # 可选，全局 OpenAI 默认模型
export UNIAPI_IMAGE_PROVIDER='gemini|openai' # 可选，extract_character_clothing 的 --provider 默认值
```

---

## 4. 后处理：`bake_alpha`

因为 Gemini 多数时候返回 `image/jpeg` / RGB PNG，OpenAI 在 `background=transparent`
时也已经带 alpha，但底层灰卡可能仍然有渐变；脚本统一用 Pillow 估算背景色并
把接近背景的像素强制 alpha=0。

签名：

```python
bake_saved_image(
    path: Path,
    *,
    threshold: int = 18,   # 每通道颜色距离阈值：≤ threshold → alpha=0
    softness: int = 32,    # (threshold, threshold+softness) 区间内线性过渡到 255
) -> dict
```

- `estimate_background_color()` 取图像四边各 `border_fraction=0.02` 内的中位数；
  跳过边缘避免衣物本体像素污染。
- 距离用 L-inf（`max(|r-bg_r|, |g-bg_g|, |b-bg_b|)`），这样灰卡色像素必被擦，
  但饱和衣物像素完全保留。
- `--bake-alpha-threshold` 默认 18，`--bake-alpha-softness` 默认 32。
- Gemini `image/jpeg` 输入 + 灰卡背景下，bake 通常能把 60–80% 背景擦成 alpha=0；
  1K 的 Lite 模型因为背景简单，bake 后透明像素只有 ~3%（衣物更紧凑）。
- OpenAI 模型本身输出透明背景 + 灰卡，bake 通常能把 50–70% 背景擦成 alpha=0。

> 不调 alpha bake 时，文件保持 API 原始字节（仍是 `.png`，可能是 RGB）。
> 调 alpha bake 后输出 RGBA 真透明，可直接给下游合成。

---

## 5. CLI 速查

```bash
# 单阶段 · Gemini（默认 provider）
python i2i_tools/extract_character_clothing.py \
  --provider gemini --model gemini-3-pro-image \
  --images a.jpg b.jpg c.jpg \
  --output outputs/CHR004.png --bake-alpha

# 单阶段 · OpenAI
python i2i_tools/extract_character_clothing.py \
  --provider openai --model gpt-image-2.5-sunburst \
  --images a.jpg b.jpg c.jpg \
  --output outputs/CHR004.png --bake-alpha

# 单阶段 · 只跑上衣
python i2i_tools/extract_character_clothing.py \
  --images a.jpg b.jpg --output outputs/CHR004.png --parts top

# 单阶段 · combined（上下两半）
python i2i_tools/extract_character_clothing.py \
  --images a.jpg b.jpg --output outputs/CHR004.png --parts combined

# 两阶段 · 默认跑 5 个模型
python i2i_tools/two_stage_clothing_pipeline.py \
  --images a.jpg b.jpg c.jpg \
  --character-label CHR004 \
  --output-root workspace/two_stage/CHR004 \
  --bake-alpha

# 两阶段 · 选特定模型
python i2i_tools/two_stage_clothing_pipeline.py \
  --images a.jpg b.jpg c.jpg \
  --character-label CHR004 \
  --output-root workspace/two_stage/CHR004 \
  --model gpt-image-2 --model gemini-3-pro-image

# Dry-run：只打印 plan，不发请求、不读 key
python i2i_tools/extract_character_clothing.py --images a.jpg --output out.png --dry-run
python i2i_tools/two_stage_clothing_pipeline.py --images a.jpg --character-label X \
  --output-root out --dry-run
```

---

## 6. 已知坑 & 提示

1. **Gemini 3.x 返回 `image/jpeg`**：脚本里已经接受任意 `image/*` 写入 `.png`
   路径，再交给 `bake_alpha` 转 RGBA。如果不要 alpha bake，原始字节就是 JPEG
   解码出的 PNG（RGB），不是真透明。
2. **Azure 上游 moderation**：`gpt-image-2 / 2.5-sunburst` 对"提取衣物"任务
   偶尔会被 `moderation_blocked`（HTTP 400）。默认 `moderation=low` 已经设了；
   触发了 retry 一次通常就能过。极端情况可以考虑换 prompt 关键词，但保留
   `PANTS_INSTRUCTIONS / asset_description` 比"简化版 prompt"过审率更高。
3. **gpt-image-2 / 2.5 不支持 `input_fidelity`**：在 `OpenAIImageModelProfile` 里
   通过 `supports_input_fidelity=False` 标记；`UniAPIImageClient.edit()` 在
   multipart 编码前自动剥离。调用方仍可传，client 会按 profile 决定要不要发。
4. **gemini-3.1-flash-lite-image 只支持 1K**：传 `4K` 会在 `resolve_model_image_size`
   里直接 raise；`image_size=auto` 才会安全落到 1K。注意 Lite 输出的透明背景
   弱，需要更高 `bake_alpha_threshold` 或者干脆走 `--bake-alpha-threshold 40`。
5. **OpenAI `output_format=png`**：必须是 `png`/`jpeg`/`webp`，被
   `_output_format_from_path` 校验；默认与 `--output` 扩展名一致。
6. **dry-run 不会读 API key**：可以放心用来校验输入和 prompt 拼装。
7. **失败部分可见**：脚本总是写一份 `*.json` 溯源，包含 SHA-256、prompt、
   model_version、usage、alpha_bake 状态、raw_mime / raw_sha256。即使模型被
   拦了（moderation / rate limit），上一次成功的产物 + 溯源都还在，可用于
   对比各模型的中间产物。
