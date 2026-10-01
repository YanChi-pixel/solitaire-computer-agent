"""
LLM strategist: one interface (LLMBackend), several implementations.

Idea: orchestrator and rule_engine know nothing about where the model's
answer came from. Today it's a free local Ollama; tomorrow, for benchmarks,
a paid cloud API (DeepSeek, OpenRouter, etc. — all OpenAI-compatible).
Only the backend changes; the rest of the code stays untouched.
"""

from __future__ import annotations

import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

from orchestrator.rule_engine import Board, Move
from planner.prompts import build_prompt, parse_response


@dataclass
class PlannerResult:
    move_index: int
    reasoning: str
    raw_response: str
    model_name: str


class LLMBackend(ABC):
    """Common interface for any model source."""

    model_name: str

    @abstractmethod
    def complete(self, messages: list[dict]) -> str:
        """Send messages to the model, return the text reply."""
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Local model via Ollama (free, no external network)."""

    def __init__(self, model_name: str = "qwen2.5:14b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def complete(self, messages: list[dict]) -> str:
        # Ollama supports both its own /api/chat and the OpenAI-compatible
        # /v1/chat/completions — use the native one, easier to parse.
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]


class OpenAICompatibleBackend(LLMBackend):
    """
    Any cloud provider with an OpenAI-compatible API:
    DeepSeek, OpenRouter, Together, etc. — all share the same request format;
    only base_url, api_key and model name differ.
    """

    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def complete(self, messages: list[dict]) -> str:
        payload = {"model": self.model_name, "messages": messages}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class MockBackend(LLMBackend):
    """
    Offline stub — for tests and for debugging the orchestrator without a
    real model. Always picks the first valid move.
    """

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def complete(self, messages: list[dict]) -> str:
        return (
            "Taking the safest move from the list to avoid unnecessary "
            "column reveals.\n"
            '{"move_index": 0}'
        )


def choose_move(backend: LLMBackend, board: Board, valid_moves: list[Move]) -> PlannerResult:
    """Single entry point: regardless of backend, the call looks the same."""
    messages = build_prompt(board, valid_moves)
    raw_response = backend.complete(messages)
    move_index, reasoning = parse_response(raw_response, valid_moves)
    return PlannerResult(
        move_index=move_index,
        reasoning=reasoning,
        raw_response=raw_response,
        model_name=backend.model_name,
    )
