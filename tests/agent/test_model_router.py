import json

from agent.model_router import classify_task_tier, resolve_model_route


MODELS = {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.4"}
CONFIG = {
    "model": {"default": "gpt-5.4", "provider": "openai-codex"},
    "smart_model_routing": {
        "enabled": True,
        "provider": "openai-codex",
        "require_live_model": True,
        "preserve_nondefault_model": True,
        "models": {
            "fast": "gpt-5.6-luna",
            "balanced": "gpt-5.6-terra",
            "frontier": "gpt-5.6-sol",
        },
        "reasoning": {"fast": "low", "balanced": "medium", "frontier": "high"},
    },
}


def _route(message, **kwargs):
    return resolve_model_route(
        message,
        provider="openai-codex",
        current_model="gpt-5.4",
        config=CONFIG,
        available_models=MODELS,
        **kwargs,
    )


def test_routine_schedule_lookup_uses_fast_lane():
    decision = _route("What is on my schedule today?")
    assert decision.model == "gpt-5.6-luna"
    assert decision.tier == "fast"
    assert decision.reasoning_effort == "low"


def test_general_planning_uses_balanced_lane():
    decision = _route("Help me plan next week around school, work, and three appointments.")
    assert decision.model == "gpt-5.6-terra"
    assert decision.tier == "balanced"
    assert decision.reasoning_effort == "medium"


def test_ambiguous_short_request_stays_balanced_for_accuracy():
    decision = _route("Fix it and make sure it cannot happen again.")
    assert decision.model == "gpt-5.6-terra"
    assert decision.tier == "balanced"


def test_complex_implementation_uses_frontier_lane():
    decision = _route("Implement and debug a multi-step database migration with regression tests.")
    assert decision.model == "gpt-5.6-sol"
    assert decision.tier == "frontier"
    assert decision.reasoning_effort == "high"


def test_high_stakes_request_uses_frontier_lane():
    tier, reason = classify_task_tier("Analyze this medication interaction before I call my doctor.")
    assert tier == "frontier"
    assert "high-stakes" in reason


def test_manual_model_override_is_preserved():
    decision = resolve_model_route(
        "What is on my schedule today?",
        provider="openai-codex",
        current_model="gpt-5.6-sol",
        config=CONFIG,
        preserve_model=True,
        available_models=MODELS,
    )
    assert decision.model == "gpt-5.6-sol"
    assert decision.routed is False


def test_nondefault_session_model_is_preserved():
    decision = resolve_model_route(
        "What is on my schedule today?",
        provider="openai-codex",
        current_model="gpt-5.5",
        config=CONFIG,
        available_models=MODELS | {"gpt-5.5"},
    )
    assert decision.model == "gpt-5.5"
    assert decision.routed is False


def test_missing_live_model_cache_preserves_current_model(tmp_path):
    decision = resolve_model_route(
        "What is on my schedule today?",
        provider="openai-codex",
        current_model="gpt-5.4",
        config=CONFIG,
        hermes_home=tmp_path,
    )
    assert decision.model == "gpt-5.4"
    assert decision.routed is False


def test_live_cache_is_used_for_model_validation(tmp_path):
    (tmp_path / "provider_models_cache.json").write_text(
        json.dumps({"openai-codex": {"models": sorted(MODELS)}}),
        encoding="utf-8",
    )
    decision = resolve_model_route(
        "Show me today's schedule.",
        provider="openai-codex",
        current_model="gpt-5.4",
        config=CONFIG,
        hermes_home=tmp_path,
    )
    assert decision.model == "gpt-5.6-luna"


def test_other_providers_are_not_routed():
    decision = resolve_model_route(
        "Implement a complex migration.",
        provider="openrouter",
        current_model="openai/gpt-5.4",
        config=CONFIG,
        available_models=MODELS,
    )
    assert decision.model == "openai/gpt-5.4"
    assert decision.routed is False
