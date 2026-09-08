"""Model wrappers for MisconceptDialogue experiments.

Imports are intentionally lazy so lightweight commands such as ``--help`` and
static checks do not require every optional backend dependency (for example
``torch`` for HuggingFace models or API clients for hosted models).
"""

from .base_model import BaseModel, ModelResponse

__all__ = ["BaseModel", "ModelResponse", "OpenAIModel", "HFModel"]


def __getattr__(name):
    if name == "OpenAIModel":
        from .openai_model import OpenAIModel
        return OpenAIModel
    if name == "HFModel":
        from .hf_model import HFModel
        return HFModel
    raise AttributeError(name)
