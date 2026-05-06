from __future__ import annotations

import json
import re
from typing import Any

import requests

from .config import DEFAULT_MODEL_NAME, DEFAULT_OLLAMA_URL


def ollama_chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL_NAME,
    url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.1,
    timeout: int = 180,
    num_predict: int = 512,
    format_json: bool = True,
    think: bool = False,
) -> str:
    options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
        "think": think,
    }
    if format_json:
        payload["format"] = "json"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    message = body.get("message", {})
    content = message.get("content", "")
    if content is None:
        content = ""
    return str(content)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found in LLM output:\n{text}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM output JSON must be an object.")
    return parsed
