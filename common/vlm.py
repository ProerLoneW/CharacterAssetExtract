# -*- coding: utf-8 -*-
"""MiMo VLM/LLM 客户端（基于 OpenAI SDK，异步并发）。

⑤ 非说话人 VLM 识别、⑥ clip caption（MiMo）、⑦ 连续性判定小 LLM 共用。
支持文本、图片（base64 image_url）、音频（base64 audio_url）多模态消息。

调用模式与 mimo_annotate/batch_mimo_audio_caption.py 对齐：
  - 服务端：sglang 部署的 MiMo-V2.5，OpenAI 兼容 API
  - 客户端：openai AsyncOpenAI + asyncio.Semaphore 并发
  - thinking 模式通过 extra_body.chat_template_kwargs.enable_thinking 显式控制
  - 默认服务地址：http://tcp30000-1058.default.svc.cluster.local:30000/v1

环境变量（与 mimo_annotate 脚本对齐）：
  OPENAI_BASE_URL / OPENAI_API_BASE / API_BASE  → base_url
  MODEL_NAME                                     → model
  MAX_COMPLETION_TOKENS                          → max_completion_tokens
  REQUEST_TIMEOUT                                → timeout（默认 1200s）
  RETRIES                                        → retries
  THINKING                                       → thinking（"1" 开启）
  WORKERS / CONCURRENT                           → 并发数
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import AsyncOpenAI, OpenAI  # type: ignore
except ImportError as _e:  # openai missing -> LLMClient unusable but module still importable
    AsyncOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore
    _OPENAI_IMPORT_ERROR = _e
else:
    _OPENAI_IMPORT_ERROR = None


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


DEFAULT_BASE_URL = "http://tcp30000-1058.default.svc.cluster.local:30000/v1"
DEFAULT_MODEL = "MiMo-V2.5"
DEFAULT_TIMEOUT = 1200


# ---------------------------------------------------------------------------
# 响应解析（与 batch_mimo_audio_caption.py 对齐）
# ---------------------------------------------------------------------------

def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("json\n"):
        cleaned = cleaned[5:].strip()
    return cleaned


def remove_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_first_json_object(text: str) -> str:
    cleaned = strip_code_fences(remove_thinking(text))
    start = cleaned.find("{")
    if start < 0:
        return cleaned
    depth = 0
    in_string = False
    escape_next = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start:index + 1]
    return cleaned[start:]


def repair_json_text(text: str) -> str:
    cleaned = extract_first_json_object(text)
    cleaned = cleaned.replace(": None", ": null")
    cleaned = cleaned.replace(": True", ": true").replace(": False", ": false")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def extract_json(text: str) -> dict:
    """从模型输出中提取第一个 JSON 对象（容忍 think 标签、code fence、前后废话）。"""
    cleaned = repair_json_text(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"输出中无 JSON 对象: {text[:200]}")
    return json.loads(m.group(0))


def extract_message_text(message_content: Any) -> str:
    """从 response message.content 提取纯文本（兼容 str 和 list 格式）。"""
    if isinstance(message_content, str):
        return message_content.strip()
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return str(message_content).strip()


# ---------------------------------------------------------------------------
# 多模态内容编码
# ---------------------------------------------------------------------------

def image_url_b64(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    return f"data:{mime};base64,{b64}"


def audio_url_b64(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    mime = mimetypes.guess_type(path)[0] or "audio/wav"
    return f"data:{mime};base64,{b64}"


def load_prompt_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# 构建 messages
# ---------------------------------------------------------------------------

def build_user_content(prompt: str, images: Optional[List[str]] = None,
                       audio: Optional[str] = None) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images or []:
        content.append({"type": "image_url",
                        "image_url": {"url": image_url_b64(img)}})
    if audio:
        content.append({"type": "audio_url",
                        "audio_url": {"url": audio_url_b64(audio)}})
    return content


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI 兼容 chat completions 客户端。

    同步接口 chat() 供 m06/m07 逐条调用；
    异步接口 chat_batch_async() 供批量并发打标。
    """

    def __init__(self, base_url: str = "",
                 model: str = "", api_key: str = "EMPTY",
                 timeout: float = 0, retries: int = 0,
                 max_completion_tokens: int = 0,
                 thinking: Optional[bool] = None,
                 workers: int = 0,
                 temperature: float = 0.2,
                 max_api_seconds: float = 0):
        self.base_url = (base_url
                         or _env("OPENAI_BASE_URL")
                         or _env("OPENAI_API_BASE")
                         or _env("API_BASE")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or _env("MODEL_NAME") or DEFAULT_MODEL
        self.api_key = api_key or _env("OPENAI_API_KEY") or "EMPTY"
        self.timeout = timeout or float(_env("REQUEST_TIMEOUT") or str(DEFAULT_TIMEOUT))
        self.retries = retries or int(_env("RETRIES") or "3")
        self.max_completion_tokens = (max_completion_tokens
                                      or int(_env("MAX_COMPLETION_TOKENS") or "4096"))
        if thinking is None:
            thinking = _env("THINKING", "0") == "1"
        self.thinking = thinking
        self.workers = max(1, workers or int(_env("WORKERS") or _env("CONCURRENT") or "8"))
        self.temperature = temperature
        self.max_api_seconds = (max_api_seconds
                                or float(_env("MAX_API_SECONDS") or "30"))

        self._sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,
        )
        self._async_client: Optional[AsyncOpenAI] = None

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        return self._async_client

    # ------------------------------------------------------------------
    # 核心调用参数
    # ------------------------------------------------------------------

    def _build_kwargs(self, messages: List[Dict[str, Any]],
                      temperature: Optional[float] = None) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_completion_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self.thinking}},
            "timeout": self.timeout,
        }

    def _build_messages(self, prompt: str, images: Optional[List[str]] = None,
                        system: str = "", audio: Optional[str] = None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": build_user_content(prompt, images, audio),
        })
        return messages

    # ------------------------------------------------------------------
    # 同步接口（m06/m07 逐条调用）
    # ------------------------------------------------------------------

    def chat(self, prompt: str, images: Optional[List[str]] = None,
             system: str = "", temperature: Optional[float] = None,
             audio: Optional[str] = None) -> str:
        messages = self._build_messages(prompt, images, system, audio)
        kwargs = self._build_kwargs(messages, temperature)

        last_err: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                response = self._sync_client.chat.completions.create(**kwargs)
                return extract_message_text(response.choices[0].message.content)
            except Exception as e:
                last_err = e
                time.sleep(min(5 * (attempt + 1), 20))
        raise RuntimeError(f"LLM 请求失败（{self.model}@{self.base_url}）: {last_err}")

    # ------------------------------------------------------------------
    # 异步接口（批量并发）
    # ------------------------------------------------------------------

    async def chat_async(self, prompt: str, images: Optional[List[str]] = None,
                         system: str = "", temperature: Optional[float] = None,
                         audio: Optional[str] = None,
                         semaphore: Optional[asyncio.Semaphore] = None) -> str:
        client = self._get_async_client()
        messages = self._build_messages(prompt, images, system, audio)
        kwargs = self._build_kwargs(messages, temperature)

        async def _do() -> str:
            last_err: Optional[Exception] = None
            for attempt in range(self.retries):
                try:
                    request = client.chat.completions.create(**kwargs)
                    if self.max_api_seconds > 0:
                        response = await asyncio.wait_for(request, timeout=self.max_api_seconds)
                    else:
                        response = await request
                    return extract_message_text(response.choices[0].message.content)
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"api_time_limit_exceeded: limit={self.max_api_seconds:.1f}s "
                        f"({self.model}@{self.base_url})")
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(min(5 * (attempt + 1), 20))
            raise RuntimeError(f"LLM 请求失败（{self.model}@{self.base_url}）: {last_err}")

        if semaphore:
            async with semaphore:
                return await _do()
        return await _do()

    async def chat_batch_async(
        self,
        requests: List[Tuple[str, Optional[List[str]], str]],
        temperature: Optional[float] = None,
        audio_paths: Optional[List[Optional[str]]] = None,
    ) -> List[Optional[str]]:
        """并发批量请求。

        requests: [(prompt, images, system), ...]
        audio_paths: 与 requests 等长的音频路径列表（可选）
        返回与 requests 等长的结果列表（失败为 None）。
        """
        semaphore = asyncio.Semaphore(self.workers)
        results: List[Optional[str]] = [None] * len(requests)

        async def _task(idx: int) -> None:
            prompt, images, system = requests[idx]
            audio = audio_paths[idx] if audio_paths else None
            try:
                results[idx] = await self.chat_async(
                    prompt, images, system, temperature, audio, semaphore)
            except Exception as e:
                print(f"[vlm] 批量请求 #{idx} 失败：{e}", flush=True)
                results[idx] = None

        await asyncio.gather(*[_task(i) for i in range(len(requests))])
        return results

    def chat_batch(self, requests: List[Tuple[str, Optional[List[str]], str]],
                   temperature: Optional[float] = None) -> List[Optional[str]]:
        """同步包装的批量并发请求。"""
        return asyncio.run(self.chat_batch_async(requests, temperature))

    # ------------------------------------------------------------------
    # 连通性检查
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        try:
            self._sync_client.models.list(timeout=10)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # 兼容旧接口（静态方法）
    # ------------------------------------------------------------------

    @staticmethod
    def image_url_b64(path: str) -> str:
        return image_url_b64(path)

    @staticmethod
    def audio_url_b64(path: str) -> str:
        return audio_url_b64(path)

    @staticmethod
    def load_prompt_file(path: str) -> str:
        return load_prompt_file(path)
