"""Abstract base class for model wrappers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """Standardized response from any model wrapper."""

    text: str
    model_name: str
    usage: dict = field(default_factory=dict)
    raw_response: Any = field(default=None, repr=False)


class BaseModel(ABC):
    """Abstract base class that all model wrappers must implement."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 384,
        top_p: float = 0.95,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from a single prompt.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling probability.
            **kwargs: Additional model-specific parameters.

        Returns:
            A ModelResponse containing the generated text and metadata.
        """

    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 384,
        **kwargs: Any,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of user prompts.
            system_prompt: Optional system-level instruction applied to all prompts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate per response.
            **kwargs: Additional model-specific parameters.

        Returns:
            A list of ModelResponse objects, one per prompt.
        """

    @abstractmethod
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
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional model-specific parameters.

        Returns:
            A ModelResponse for the next turn.
        """
