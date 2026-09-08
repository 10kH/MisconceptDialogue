"""HuggingFace local model wrapper."""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_model import BaseModel, ModelResponse


MODEL_MAP: dict[str, str] = {
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    # 2025-2026 models
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "glm-4.7-flash": "zai-org/GLM-4.7-Flash",
    "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "phi-4-reasoning": "microsoft/Phi-4-reasoning-plus",
    "phi-4": "microsoft/phi-4",
}


# Models that produce <think>...</think> reasoning blocks
REASONING_MODELS = {
    "phi-4-reasoning", "qwen3.5-9b",
    "microsoft/Phi-4-reasoning-plus", "Qwen/Qwen3.5-9B",
    "glm-4.7-flash", "zai-org/GLM-4.7-Flash",
}

# Models that output meta-reasoning as plain text (not <think> tags)
# and need model-specific post-processing to extract the actual response.
_META_REASONING_MODELS = {
    "glm-4.7-flash": "glm",
    "zai-org/GLM-4.7-Flash": "glm",
    "phi-4-reasoning": "phi4",
    "microsoft/Phi-4-reasoning-plus": "phi4",
}

# Minimum max_tokens for reasoning models to fit think block + answer
_REASONING_MIN_TOKENS = 2048
_PHI4_MIN_TOKENS = 4096

# Phi-4 loops endlessly inside <think> without repetition penalty
_PHI4_REPETITION_PENALTY = 1.3

# Large / MoE models that need multi-GPU even though weights barely fit one GPU
# (KV-cache causes OOM).  We cap per-GPU memory to force distribution.
_LARGE_MODELS = {"llama-4-scout"}
_PER_GPU_MAX_MEM = "70GiB"


class HFModel(BaseModel):
    """Wrapper for local HuggingFace causal language models.

    Short model names (e.g. "qwen-2.5-7b") are resolved via MODEL_MAP;
    any other string is treated as a raw HuggingFace model ID or local path.
    """

    def __init__(self, model_name: str, gpu_index: int | None = None,
                 quantize: str | None = None, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        hf_id = MODEL_MAP.get(model_name, model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = dict(
            dtype=torch.bfloat16,
            device_map={"": gpu_index} if gpu_index is not None else "auto",
            trust_remote_code=True,
        )
        # 4-bit quantization via bitsandbytes (reduces ~14GB model to ~4GB)
        if quantize == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            del load_kwargs["dtype"]  # incompatible with quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        self.model.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_think_block(text: str) -> str:
        """Remove <think>...</think> reasoning blocks, keeping only the final answer."""
        import re
        m = re.search(r"</think>\s*", text, re.DOTALL)
        if m:
            return text[m.end():].strip()
        # If starts with <think> but no closing tag, response was truncated
        if text.lstrip().startswith("<think>"):
            return ""
        return text

    @staticmethod
    def _strip_glm_reasoning(text: str) -> str:
        """Strip GLM's numbered analysis pattern (e.g. '1. **Analyze the Request:**...')."""
        import re
        if not re.match(r'\s*1\.\s+\*\*', text):
            return text

        # Strategy 1: Extract quoted response after drafting/response marker
        draft = re.search(
            r'(?:\*?Drafting[^*\n]*\*?|Draft(?:ing)?\s+(?:the\s+)?response'
            r'|(?:My |The |Final )?[Rr]esponse\s*(?:to the student)?)'
            r'[:\s]*\n?\s*"',
            text, re.IGNORECASE,
        )
        if draft:
            response = text[draft.end():].strip()
            return response.rstrip('"').strip()

        # Strategy 2: Find last numbered step, take everything after it
        # that doesn't look like analysis
        last_step = None
        for m in re.finditer(r'\n\d+\.\s+\*\*', text):
            last_step = m
        if last_step:
            after = text[last_step.start():]
            # Find end of the last numbered step's content
            step_end = re.search(r'\n\n+', after)
            if step_end:
                remainder = after[step_end.end():].strip()
                # Check if remainder looks like actual response (not more analysis)
                if remainder and not re.match(r'\d+\.\s+\*\*', remainder):
                    return remainder.rstrip('"').strip()

        # No extractable response — entire output is reasoning
        return ""

    @staticmethod
    def _strip_phi4_reasoning(text: str) -> str:
        """Strip Phi-4's meta-reasoning prefix from the generated text.

        Phi-4 often produces planning/analysis text before (or instead of) the
        actual response.  Common patterns:
          - "We are Phi...", "We have a conversation...", "We must not reveal..."
          - "Let's think:", "Let me think", "I need to"
          - "The student's misconception is..."
          - "But we need to / But instructions say"

        Strategy: detect if text starts with a meta pattern, then scan paragraphs
        to find the first one that looks like a direct student-facing response
        (starts with a greeting, question, or instructional sentence rather than
        meta-commentary).
        """
        import re

        # Quick exit: if text doesn't start with a known meta pattern, keep it
        _META_START = re.compile(
            r"\s*(?:"
            r"We (?:are|have|need|should|must|can)\b"
            r"|The (?:user|student(?:'s)?)\b"
            r"|User says\b"
            r"|Let(?:'s| me| us) (?:think|consider|analyze|start|begin|try)"
            r"|I (?:need|should|must|have|will|'ll) (?:to\b|produce|help|explain|provide|address)"
            r"|But (?:we|the|instructions)\b"
            r"|Now[,:]?\s"
            r"|OK[,.]?\s"
            r"|So[,:]?\s(?:we|the|I)\b"
            r"|First[,:]?\s(?:we|the|I|let)\b"
            r"|Here(?:'s| is) (?:the|my|a)\b"
            r")",
            re.IGNORECASE,
        )
        if not _META_START.match(text):
            return text

        # Split on double newlines, skip meta paragraphs
        paragraphs = re.split(r'\n\n+', text)
        meta_re = re.compile(
            r"(?:"
            r"We (?:are|have|need|should|must|can|'ll)\b"
            r"|The (?:user|student)\b"
            r"|User says\b"
            r"|Let(?:'s| me| us)\b"
            r"|I (?:need|should|must|'ll|will)\b"
            r"|But (?:we|the|instructions)\b"
            r"|Now respond|Remember:"
            r"|\*\*Role\*\*|\*\*Goal\*\*|\*\*Strategy\*\*"
            r"|Problem:|Student's answer:|Misconception:|Diagnosed"
            r"|not reveal the correct answer"
            r"|counterexample|analogy|cognitive.conflict|direct.instruction|socratic"
            r"|I'll produce|I'll help|I'll explain"
            r")",
            re.IGNORECASE,
        )
        for i, para in enumerate(paragraphs):
            if i == 0:
                continue
            stripped = para.strip()
            if not meta_re.search(stripped[:150]) and len(stripped) > 20:
                return "\n\n".join(paragraphs[i:]).strip()

        # Fallback: try to find a quoted response within the meta text
        quoted = re.search(r'"([^"]{30,})"', text)
        if quoted:
            return quoted.group(1).strip()

        return ""

    def _strip_model_reasoning(self, text: str) -> str:
        """Apply model-specific reasoning/meta-prefix stripping."""
        model_type = _META_REASONING_MODELS.get(self.model_name)
        if model_type == "glm":
            return self._strip_glm_reasoning(text)
        if model_type == "phi4":
            return self._strip_phi4_reasoning(text)
        return text

    def _decode_new_tokens(self, output_ids: torch.Tensor, prompt_len: int) -> str:
        new_ids = output_ids[prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        text = self._strip_think_block(text)
        return self._strip_model_reasoning(text)

    def _usage(self, prompt_tokens: int, completion_tokens: int) -> dict:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _boost_tokens(self, max_tokens: int) -> int:
        """Ensure reasoning models have enough tokens for <think> + answer."""
        if _META_REASONING_MODELS.get(self.model_name) == "phi4":
            return max(max_tokens, _PHI4_MIN_TOKENS)
        if self.model_name in REASONING_MODELS:
            return max(max_tokens, _REASONING_MIN_TOKENS)
        return max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
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
            temperature: Sampling temperature (0 = greedy).
            max_tokens: Maximum new tokens to generate.
            top_p: Nucleus sampling probability.
            **kwargs: Extra arguments forwarded to model.generate().

        Returns:
            ModelResponse with generated text and token usage.
        """
        max_tokens = self._boost_tokens(max_tokens)
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # For reasoning models whose template opens a <think> block,
        # pre-fill a closing tag to skip chain-of-thought.
        if self.model_name in REASONING_MODELS:
            if formatted.rstrip().endswith("<think>"):
                # Template already opened <think> (e.g. Qwen3.5)
                formatted += "\n</think>\n\n"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if _META_REASONING_MODELS.get(self.model_name) == "phi4":
            gen_kwargs.setdefault("repetition_penalty", _PHI4_REPETITION_PENALTY)
        gen_kwargs.update(kwargs)
        output_ids = self.model.generate(**inputs, **gen_kwargs)

        text = self._decode_new_tokens(output_ids[0], prompt_len)
        completion_tokens = output_ids.shape[1] - prompt_len
        return ModelResponse(
            text=text,
            model_name=self.model_name,
            usage=self._usage(prompt_len, completion_tokens),
            raw_response=output_ids,
        )

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 384,
        **kwargs: Any,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of prompts in a single forward pass.

        All prompts are tokenized together with left-padding so they can be
        processed as one batch on the GPU.

        Args:
            prompts: List of user prompts.
            system_prompt: Optional system-level instruction applied to each prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum new tokens to generate per sequence.
            **kwargs: Extra arguments forwarded to model.generate().

        Returns:
            List of ModelResponse objects in the same order as prompts.
        """
        max_tokens = self._boost_tokens(max_tokens)
        formatted_texts: list[str] = []
        for prompt in prompts:
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            fmt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if self.model_name in REASONING_MODELS:
                if fmt.rstrip().endswith("<think>"):
                    fmt += "\n</think>\n\n"
            formatted_texts.append(fmt)

        # Left-pad so all sequences are the same length.
        original_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            formatted_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self.model.device)
        self.tokenizer.padding_side = original_padding

        prompt_len = inputs.input_ids.shape[1]
        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=0.95,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if _META_REASONING_MODELS.get(self.model_name) == "phi4":
            gen_kwargs.setdefault("repetition_penalty", _PHI4_REPETITION_PENALTY)
        gen_kwargs.update(kwargs)
        output_ids = self.model.generate(**inputs, **gen_kwargs)

        results: list[ModelResponse] = []
        for seq in output_ids:
            text = self._decode_new_tokens(seq, prompt_len)
            completion_tokens = seq.shape[0] - prompt_len
            results.append(
                ModelResponse(
                    text=text,
                    model_name=self.model_name,
                    usage=self._usage(prompt_len, completion_tokens),
                    raw_response=seq,
                )
            )
        return results

    @torch.no_grad()
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
            max_tokens: Maximum new tokens to generate.
            **kwargs: Extra arguments forwarded to model.generate().

        Returns:
            ModelResponse for the next turn.
        """
        max_tokens = self._boost_tokens(max_tokens)
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.model_name in REASONING_MODELS:
            if formatted.rstrip().endswith("<think>"):
                formatted += "\n</think>\n\n"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
        prompt_len = inputs.input_ids.shape[1]

        do_sample = temperature > 0.0
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=0.95,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if _META_REASONING_MODELS.get(self.model_name) == "phi4":
            gen_kwargs.setdefault("repetition_penalty", _PHI4_REPETITION_PENALTY)
        gen_kwargs.update(kwargs)
        output_ids = self.model.generate(**inputs, **gen_kwargs)

        text = self._decode_new_tokens(output_ids[0], prompt_len)
        completion_tokens = output_ids.shape[1] - prompt_len
        return ModelResponse(
            text=text,
            model_name=self.model_name,
            usage=self._usage(prompt_len, completion_tokens),
            raw_response=output_ids,
        )
