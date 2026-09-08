"""OpenAI API model wrapper."""
from __future__ import annotations

import os
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_model import BaseModel, ModelResponse


class OpenAIModel(BaseModel):
    """Wrapper for OpenAI chat completion models (e.g. GPT-4o)."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key required. Set the OPENAI_API_KEY environment "
                "variable or pass api_key=... to the constructor."
            )
        self.client = OpenAI(api_key=resolved_key, timeout=120.0)
        self._use_completion_tokens = any(
            model_name.startswith(p)
            for p in ("gpt-5", "o1", "o3")
        )

    def _token_kwargs(self, max_tokens: int) -> dict[str, int]:
        """Return the correct max-tokens parameter for this model."""
        if self._use_completion_tokens:
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    def _sampling_kwargs(self, temperature: float, top_p: float | None = None) -> dict[str, Any]:
        """Return sampling params, omitting unsupported ones for newer models."""
        if self._use_completion_tokens:
            return {}  # gpt-5-*, o1, o3 only support default temperature
        kw: dict[str, Any] = {"temperature": temperature}
        if top_p is not None:
            kw["top_p"] = top_p
        return kw

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=60))
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 384,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from a single prompt via the OpenAI API.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling probability.
            **kwargs: Extra parameters forwarded to chat.completions.create
                      (e.g. response_format={"type": "json_object"}).

        Returns:
            ModelResponse with generated text and token usage.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self._sampling_kwargs(temperature, top_p),
            **self._token_kwargs(max_tokens),
            **kwargs,
        )
        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return ModelResponse(
            text=choice.message.content or "",
            model_name=self.model_name,
            usage=usage,
            raw_response=response,
        )

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 384,
        **kwargs: Any,
    ) -> list[ModelResponse]:
        """Generate responses for a list of prompts (sequential API calls).

        Args:
            prompts: List of user prompts.
            system_prompt: Optional system-level instruction applied to each prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate per response.
            **kwargs: Extra parameters forwarded to generate().

        Returns:
            List of ModelResponse objects in the same order as prompts.
        """
        return [
            self.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            for prompt in prompts
        ]

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=60))
    def generate_with_history(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 384,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response continuing a multi-turn conversation.

        Args:
            messages: Full conversation history as a list of role/content dicts,
                e.g. [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}].
                A system message may be included as the first element.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Extra parameters forwarded to chat.completions.create
                      (e.g. response_format={"type": "json_object"}).

        Returns:
            ModelResponse for the next turn.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            **self._sampling_kwargs(temperature),
            **self._token_kwargs(max_tokens),
            **kwargs,
        )
        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return ModelResponse(
            text=choice.message.content or "",
            model_name=self.model_name,
            usage=usage,
            raw_response=response,
        )
