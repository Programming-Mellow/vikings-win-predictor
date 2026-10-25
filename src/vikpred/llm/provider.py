"""Which Claude endpoint the chat uses. Switching providers is configuration only.

Env:
    LLM_PROVIDER   "anthropic" (default) or "bedrock"
    LLM_MODEL      model id; defaults to Claude Sonnet 5.5 for the chosen provider
    AWS_REGION     required for bedrock
"""

import os
from dataclasses import dataclass

import anthropic


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    # Server-side refusal fallback is a Claude API feature; Bedrock doesn't offer it.
    supports_fallbacks: bool
    # Mid-conversation `role: "system"` messages (used for the learner profile).
    # Left off for Bedrock until verified there; the agent falls back to a user-turn block.
    supports_system_messages: bool = True


def load_config() -> LLMConfig:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "bedrock":
        return LLMConfig("bedrock", os.environ.get("LLM_MODEL", "anthropic.claude-sonnet-5-5"), False, False)
    return LLMConfig("anthropic", os.environ.get("LLM_MODEL", "claude-sonnet-5-5"), True, True)


def make_client(config: LLMConfig):
    if config.provider == "bedrock":
        from anthropic import AnthropicBedrockMantle

        return AnthropicBedrockMantle(aws_region=os.environ["AWS_REGION"])
    return anthropic.Anthropic()


def credentials_available(config: LLMConfig) -> bool:
    if config.provider == "bedrock":
        return bool(os.environ.get("AWS_REGION"))
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
