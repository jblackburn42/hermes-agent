"""Fast, deterministic model routing for interactive and scheduled turns.

The router deliberately makes no LLM call. It uses conservative local signals,
defaults ambiguous work to the balanced lane, and only routes when the live
provider cache confirms that the configured target model exists.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "fast": "gpt-5.6-luna",
    "balanced": "gpt-5.6-terra",
    "frontier": "gpt-5.6-sol",
}
_DEFAULT_REASONING = {"fast": "low", "balanced": "medium", "frontier": "high"}

_HIGH_STAKES_RE = re.compile(
    r"\b(?:medical|diagnos(?:e|is)|medication|legal|lawsuit|contract|tax|financial|"
    r"investment|security|credential|privacy|production outage|data loss)\b",
    re.IGNORECASE,
)
_FRONTIER_RE = re.compile(
    r"\b(?:implement|debug|root cause|architecture|architect|refactor|migrat(?:e|ion)|"
    r"code review|threat model|audit|benchmark|optimi[sz]e|regression|race condition|"
    r"database schema|design system|multi[- ]?agent)\b",
    re.IGNORECASE,
)
_BALANCED_RE = re.compile(
    r"\b(?:analy[sz]e|compare|research|plan|strategy|brainstorm|draft|rewrite|summari[sz]e|"
    r"explain|recommend|prioriti[sz]e|organize|synthesi[sz]e|evaluate)\b",
    re.IGNORECASE,
)
_FAST_RE = re.compile(
    r"\b(?:what(?:'s| is) (?:on )?my schedule|today'?s schedule|mark .+ complete|"
    r"completed tasks?|update (?:my )?(?:schedule|task|calendar)|status of|look up|"
    r"find|when is|what time|list|show me|read|check my|remind me|add .+ to my)\b",
    re.IGNORECASE,
)
_SHORT_FAST_RE = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|yes|no|who is|what is|what's|when is|"
    r"where is|how many|define|convert|calculate)\b",
    re.IGNORECASE,
)
_MULTISTEP_RE = re.compile(
    r"(?:\bfirst\b.+\bthen\b|\bstep\s+\d|\b(?:three|four|five|six|seven|eight)\s+"
    r"(?:things|steps|suggestions|tasks)|\ball\s+of\s+these\b)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ModelRouteDecision:
    model: str
    tier: str
    reasoning_effort: Optional[str]
    reason: str
    routed: bool


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message.strip()
    if not isinstance(message, list):
        return str(message or "").strip()
    parts: list[str] = []
    for item in message:
        if not isinstance(item, Mapping):
            continue
        if item.get("type") in {"text", "input_text"}:
            parts.append(str(item.get("text") or ""))
    return "\n".join(parts).strip()


def classify_task_tier(message: Any) -> tuple[str, str]:
    """Classify a turn without network or model latency."""
    text = _message_text(message)
    if not text:
        return "balanced", "empty or non-text turn"

    score = 0
    reasons: list[str] = []
    if _HIGH_STAKES_RE.search(text):
        score += 5
        reasons.append("high-stakes domain")
    if _FRONTIER_RE.search(text):
        score += 3
        reasons.append("complex implementation or diagnosis")
    if "```" in text or re.search(r"\b(?:traceback|stack trace|exception|pytest|git diff)\b", text, re.I):
        score += 3
        reasons.append("code or error evidence")
    if _MULTISTEP_RE.search(text):
        score += 2
        reasons.append("multi-step request")
    if len(text) >= 1800:
        score += 3
        reasons.append("large prompt")
    elif len(text) >= 700:
        score += 1
        reasons.append("substantial prompt")

    if score >= 3:
        return "frontier", ", ".join(reasons)
    if score >= 2:
        return "balanced", ", ".join(reasons)
    if len(text) <= 600 and _FAST_RE.search(text):
        return "fast", "routine retrieval or update"
    if len(text) <= 180 and _SHORT_FAST_RE.search(text):
        return "fast", "short, well-bounded request"
    return "balanced", "general-purpose request"


def _configured_default_model(config: Mapping[str, Any]) -> str:
    value = config.get("model") or ""
    if isinstance(value, Mapping):
        value = value.get("default") or value.get("model") or ""
    return str(value).strip()


def _load_available_models(provider: str, hermes_home: Optional[Path]) -> set[str]:
    if hermes_home is None:
        try:
            from hermes_cli.config import get_hermes_home

            hermes_home = get_hermes_home()
        except Exception:
            return set()
    try:
        payload = json.loads(
            (Path(hermes_home) / "provider_models_cache.json").read_text(encoding="utf-8")
        )
        entry = payload.get(provider) or {}
        return {
            str(model).strip()
            for model in (entry.get("models") or [])
            if str(model).strip()
        }
    except Exception:
        return set()


def resolve_model_route(
    message: Any,
    *,
    provider: str,
    current_model: str,
    config: Mapping[str, Any],
    preserve_model: bool = False,
    available_models: Optional[set[str]] = None,
    hermes_home: Optional[Path] = None,
) -> ModelRouteDecision:
    """Return the effective model lane while preserving explicit choices."""
    raw = config.get("smart_model_routing") or {}
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, Mapping) or not raw.get("enabled", False):
        return ModelRouteDecision(current_model, "disabled", None, "routing disabled", False)

    target_provider = str(raw.get("provider") or "openai-codex").strip().lower()
    active_provider = str(provider or "").strip().lower()
    if active_provider != target_provider:
        return ModelRouteDecision(current_model, "disabled", None, "provider not routed", False)

    configured_default = _configured_default_model(config)
    if preserve_model or (
        raw.get("preserve_nondefault_model", True)
        and configured_default
        and current_model
        and current_model != configured_default
    ):
        return ModelRouteDecision(current_model, "manual", None, "explicit model preserved", False)

    tier, reason = classify_task_tier(message)
    model_cfg = raw.get("models") if isinstance(raw.get("models"), Mapping) else {}
    reasoning_cfg = raw.get("reasoning") if isinstance(raw.get("reasoning"), Mapping) else {}
    candidates = {
        name: str(model_cfg.get(name) or default).strip()
        for name, default in _DEFAULT_MODELS.items()
    }
    available = (
        available_models
        if available_models is not None
        else _load_available_models(active_provider, hermes_home)
    )
    candidate = candidates[tier]
    if available and candidate not in available:
        order = {
            "fast": ("balanced", "frontier"),
            "balanced": ("frontier", "fast"),
            "frontier": ("balanced", "fast"),
        }[tier]
        replacement = next(
            (candidates[name] for name in order if candidates[name] in available),
            "",
        )
        if replacement:
            candidate = replacement
            reason = f"{reason}; requested lane unavailable"
        else:
            return ModelRouteDecision(
                current_model, tier, None, "no configured route model is available", False
            )
    elif not available and raw.get("require_live_model", True):
        return ModelRouteDecision(current_model, tier, None, "live model list unavailable", False)

    effort = str(reasoning_cfg.get(tier) or _DEFAULT_REASONING[tier]).strip().lower()
    logger.info(
        "smart model route: tier=%s model=%s reasoning=%s reason=%s",
        tier,
        candidate,
        effort,
        reason,
    )
    return ModelRouteDecision(candidate, tier, effort or None, reason, candidate != current_model)
