"""Experiment 1: Misconception Detection in Dialogue Context

Tests 10+ models across single-turn and multi-turn conditions.
Compares with BEA 2025 results.

Research Question 1 (RQ1): Does dialogue context improve misconception detection?
  Conditions:
    single_turn    — only the final student response (no prior context)
    multi_turn_2   — include 2 most recent prior turns
    multi_turn_full — include full dialogue history

Tracks (BEA 2025 style):
    mistake_identification — Track 1: correct / incorrect / ambiguous
    mistake_location       — Track 2: where is the error
    guidance_quality       — Track 3: rate tutor response (1/2/3)
    actionability          — Track 4: is feedback actionable (yes/no)
    misconception_classification — Track 5 (NEW): which misconception

Usage:
    # Quick smoke test (10 samples, single turn, Track 1)
    python -m src.experiments.exp1_detection.run --model qwen-2.5-7b --max_samples 10

    # Full run (API model, all tracks, multi-turn)
    python -m src.experiments.exp1_detection.run \\
        --model gpt-4o \\
        --condition multi_turn_full \\
        --tracks mistake_identification mistake_location actionability \\
        --api_key $OPENAI_API_KEY

    # Local HF model
    python -m src.experiments.exp1_detection.run \\
        --model llama-3.1-8b \\
        --condition multi_turn_2 \\
        --tracks mistake_identification misconception_classification \\
        --device auto
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Logging setup (before any project imports so logging is ready)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.models.base_model import ModelResponse
from src.data.loaders import load_bea2025, load_mathdial, load_bridge, TutoringInstance
from src.evaluation.metrics import compute_all_detection_metrics, macro_f1
from src.utils.common import set_seed, save_json, ensure_dir, RESULTS_DIR, LOGS_DIR

from src.experiments.exp1_detection.prompts import (
    SYSTEM_PROMPT_DETECTION,
    build_mistake_identification_prompt,
    build_mistake_location_prompt,
    build_guidance_quality_prompt,
    build_actionability_prompt,
    build_misconception_classification_prompt,
    _slice_context,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CONDITIONS = ("single_turn", "multi_turn_2", "multi_turn_full")

VALID_TRACKS = (
    "mistake_identification",
    "mistake_location",
    "guidance_quality",
    "actionability",
    "misconception_classification",
)

# Models that produce <think>...</think> reasoning blocks and need higher max_tokens
REASONING_MODELS = {"phi-4-reasoning", "qwen3.5-9b"}

OPENAI_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o1-mini",
    "o1-preview",
    "o3-mini",
    "claude-3-5-sonnet",
    "claude-3-5-haiku",
    "claude-opus-4",
}

# Map condition name -> context_turns argument for _slice_context
_CONDITION_TURNS: dict[str, int | None] = {
    "single_turn": 0,
    "multi_turn_2": 2,
    "multi_turn_full": -1,
}

# Labels for each track (for metric computation)
_TRACK_LABELS: dict[str, list[str]] = {
    "mistake_identification": ["correct", "incorrect", "ambiguous"],
    "mistake_location": [],  # free-form — metrics computed differently
    "guidance_quality": ["1", "2", "3"],
    "actionability": ["yes", "no"],
    "misconception_classification": [],  # dynamic, depends on misconception list
}


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(
    model_name: str,
    api_key: Optional[str] = None,
):
    """Instantiate the correct model wrapper.

    OpenAI/Anthropic API models are wrapped with OpenAIModel (which also
    supports the Anthropic API if the key and model name are set appropriately).
    All other models are loaded as HuggingFace local models.

    Args:
        model_name: Short name or full HuggingFace model ID.
        api_key: API key (for OpenAI/Anthropic). Falls back to env vars.

    Returns:
        An instantiated model that implements BaseModel.
    """
    if model_name in OPENAI_MODELS or model_name.startswith(("gpt-", "o1-", "o3-", "claude-")):
        logger.info("Using OpenAI-compatible API for model: %s", model_name)
        from src.models.openai_model import OpenAIModel
        return OpenAIModel(model_name=model_name, api_key=api_key)

    logger.info("Loading local HuggingFace model: %s", model_name)
    from src.models.hf_model import HFModel
    return HFModel(model_name=model_name)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def _load_dataset(
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> list[TutoringInstance]:
    """Load the requested dataset and return TutoringInstance list.

    Args:
        dataset_name: One of "bea2025", "mathdial", "bridge".
        max_samples: Optional cap.

    Returns:
        List of TutoringInstance objects.
    """
    loaders = {
        "bea2025": load_bea2025,
        "mathdial": load_mathdial,
        "bridge": load_bridge,
    }
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Valid: {list(loaders.keys())}"
        )
    split_defaults = {
        "bea2025": "dev",
        "mathdial": "train",
        "bridge": "train",
    }
    instances = loaders[dataset_name](split=split_defaults[dataset_name])
    if max_samples is not None:
        instances = instances[:max_samples]
    if not instances:
        logger.warning(
            "Dataset '%s' returned 0 instances. Check data availability.", dataset_name
        )
    return instances


# ---------------------------------------------------------------------------
# Misconception list helper (for Track 5)
# ---------------------------------------------------------------------------

def _build_misconception_list(instances: list[TutoringInstance]) -> list[dict]:
    """Build a deduplicated misconception list from instance metadata.

    For Track 5 classification, we need a list of candidate misconceptions.
    This function extracts unique misconception IDs seen in the dataset and
    returns them as simple dicts suitable for build_misconception_classification_prompt.

    Args:
        instances: Dataset instances (may have misconception_id populated).

    Returns:
        List of dicts with keys: id, name, description.
    """
    seen: dict[str, dict] = {}
    for inst in instances:
        mid = inst.misconception_label.strip()
        if mid and mid not in seen:
            seen[mid] = {
                "id": mid,
                "name": f"Misconception {mid}",
                "description": inst.metadata.get("topic", ""),
            }
    if not seen:
        # Provide a minimal placeholder so Track 5 is still runnable
        return [
            {"id": f"M{i:03d}", "name": f"Misconception {i}", "description": ""}
            for i in range(1, 11)
        ]
    return list(seen.values())


# ---------------------------------------------------------------------------
# Prompt builder dispatch
# ---------------------------------------------------------------------------

def _build_prompt(
    track: str,
    instance: TutoringInstance,
    condition: str,
    misconception_list: list[dict] | None = None,
) -> str:
    """Build the appropriate prompt for a given track and condition.

    Args:
        track: Which track to evaluate.
        instance: The tutoring instance.
        condition: Dialogue context condition.
        misconception_list: Required for misconception_classification track.

    Returns:
        Prompt string.
    """
    n_turns = _CONDITION_TURNS[condition]
    context = _slice_context(instance.conversation, n_turns)

    # Extract last tutor message for tracks that need it
    tutor_msgs = [m["content"] for m in instance.conversation if m.get("role") == "tutor"]
    tutor_response = tutor_msgs[-1] if tutor_msgs else ""

    if track == "mistake_identification":
        return build_mistake_identification_prompt(
            problem=instance.problem,
            student_response=instance.student_answer,
            context_turns=context,
        )
    elif track == "mistake_location":
        return build_mistake_location_prompt(
            problem=instance.problem,
            student_response=instance.student_answer,
            context_turns=context,
        )
    elif track == "guidance_quality":
        return build_guidance_quality_prompt(
            problem=instance.problem,
            student_response=instance.student_answer,
            tutor_response=tutor_response,
        )
    elif track == "actionability":
        return build_actionability_prompt(
            problem=instance.problem,
            student_response=instance.student_answer,
            tutor_response=tutor_response,
        )
    elif track == "misconception_classification":
        mc_list = misconception_list or []
        return build_misconception_classification_prompt(
            problem=instance.problem,
            student_response=instance.student_answer,
            misconception_list=mc_list,
            context_turns=context,
        )
    else:
        raise ValueError(f"Unknown track: {track}")


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_detection_response(response_text: str, track: str) -> str:
    """Parse model response into a structured prediction label.

    Each track expects a specific key in the model response:
      - mistake_identification  -> CLASSIFICATION: <label>
      - mistake_location        -> MISTAKE_LOCATION: <text>
      - guidance_quality        -> QUALITY_RATING: <1|2|3>
      - actionability           -> ACTIONABLE: <yes|no>
      - misconception_classification -> MISCONCEPTION_ID: <id>

    Falls back to keyword scanning if the structured field is not found.

    Args:
        response_text: Raw text from the model.
        track: Which track was evaluated.

    Returns:
        Parsed prediction string (normalised to lowercase where applicable).
    """
    text = response_text.strip()

    # Strip <think>...</think> reasoning blocks from reasoning models
    think_match = re.search(r"</think>\s*", text, re.DOTALL)
    if think_match:
        text = text[think_match.end():]
    elif text.startswith("<think>"):
        # Truncated response — no </think> found, all content is reasoning
        text = ""

    if track == "mistake_identification":
        m = re.search(r"CLASSIFICATION\s*:\s*(\w+)", text, re.IGNORECASE)
        if m:
            label = m.group(1).lower()
            if label in ("correct", "incorrect", "ambiguous"):
                return label
        # Conclusion-pattern fallback (handles verbose models)
        # Look for "answer/response is <label>" — take the LAST match
        conclusion = list(re.finditer(
            r"(?:answer|response|solution)\s+is\s+(correct|incorrect|ambiguous)",
            text, re.IGNORECASE,
        ))
        if conclusion:
            return conclusion[-1].group(1).lower()
        # Simple keyword fallback (check definitive labels before ambiguous)
        tl = text.lower()
        if "incorrect" in tl or "wrong" in tl or "error" in tl or "mistake" in tl:
            return "incorrect"
        if "correct" in tl:
            return "correct"
        if "ambiguous" in tl or "unclear" in tl or "cannot" in tl:
            return "ambiguous"
        return ""

    elif track == "mistake_location":
        m = re.search(r"MISTAKE_LOCATION\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return text[:200].strip()  # return first 200 chars as fallback

    elif track == "guidance_quality":
        m = re.search(r"QUALITY_RATING\s*:\s*([123])", text, re.IGNORECASE)
        if m:
            return m.group(1)
        # Fallback: find first digit 1-3
        m2 = re.search(r"\b([123])\b", text)
        if m2:
            return m2.group(1)
        return "2"  # default: adequate

    elif track == "actionability":
        m = re.search(r"ACTIONABLE\s*:\s*(yes|no)", text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        tl = text.lower()
        if "not actionable" in tl or "non-actionable" in tl or ": no" in tl:
            return "no"
        if "actionable" in tl or ": yes" in tl:
            return "yes"
        return "no"

    elif track == "misconception_classification":
        m = re.search(r"MISCONCEPTION_ID\s*:\s*(\S+)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: look for ID-like tokens (e.g. M001, MaE01)
        m2 = re.search(r"\b(Ma[A-Z]\d{2,4}|M\d{3,4}|MISC[-_]?\d{2,4}|none|other)\b", text)
        if m2:
            return m2.group(1)
        return "other"

    return text[:100].strip()


# ---------------------------------------------------------------------------
# Metrics computation per track
# ---------------------------------------------------------------------------

def _compute_track_metrics(
    track: str,
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, Any]:
    """Compute appropriate metrics for a given track.

    Args:
        track: Track name.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict of metric name -> value.
    """
    if track == "mistake_identification":
        # 3-class classification: use compute_all_detection_metrics
        return compute_all_detection_metrics(y_true, y_pred)

    elif track == "guidance_quality":
        # 3-class ordinal
        return compute_all_detection_metrics(y_true, y_pred)

    elif track == "actionability":
        # Binary classification
        from sklearn.metrics import f1_score, accuracy_score
        return {
            "f1_binary": f1_score(y_true, y_pred, pos_label="yes", average="binary", zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        }

    elif track == "misconception_classification":
        # Multi-class — use macro F1 where we have labels
        valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t]
        if not valid_pairs:
            return {"macro_f1": 0.0, "n_evaluated": 0}
        yt, yp = zip(*valid_pairs)
        return {
            "macro_f1": macro_f1(list(yt), list(yp)),
            "accuracy": sum(t == p for t, p in zip(yt, yp)) / len(yt),
            "n_evaluated": len(yt),
        }

    elif track == "mistake_location":
        # Free-form — report exact match rate (location is "none" vs non-none)
        true_has_mistake = [1 if t.lower() != "none" else 0 for t in y_true]
        pred_has_mistake = [1 if p.lower() != "none" else 0 for p in y_pred]
        from sklearn.metrics import f1_score, accuracy_score
        return {
            "detection_f1": f1_score(
                true_has_mistake, pred_has_mistake, average="binary", zero_division=0
            ),
            "accuracy": accuracy_score(true_has_mistake, pred_has_mistake),
        }

    return {}


# ---------------------------------------------------------------------------
# Ground-truth label extraction
# ---------------------------------------------------------------------------

def _get_ground_truth(instance: TutoringInstance, track: str) -> str:
    """Extract the ground-truth label for an instance and track.

    For datasets without per-track annotations, we infer where possible:
      - mistake_identification: use instance.label if set, else "incorrect"
      - misconception_classification: use instance.misconception_id
      - guidance_quality / actionability: empty string (no gold label in most datasets)
      - mistake_location: "none" if correct, else empty string

    Args:
        instance: The TutoringInstance.
        track: Which track.

    Returns:
        String label (may be empty if unavailable).
    """
    if track == "mistake_identification":
        lbl = "correct" if instance.is_correct else "incorrect"
        if lbl in ("correct", "incorrect", "ambiguous"):
            return lbl
        # Default: treat as incorrect
        return "incorrect"

    elif track == "misconception_classification":
        return instance.misconception_label.strip()

    elif track == "mistake_location":
        # If we know the instance is correct, gold location is "none"
        if instance.is_correct:
            return "none"
        return ""  # no gold span annotation in most datasets

    elif track in ("guidance_quality", "actionability"):
        return ""  # no gold labels available in the loaded datasets

    return ""


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_detection_experiment(
    model_name: str,
    condition: str,
    tracks: list[str],
    dataset_name: str = "bea2025",
    output_dir: Optional[str] = None,
    use_api: bool = False,
    api_key: Optional[str] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run detection experiment for a single model/condition pair.

    Args:
        model_name: Model shorthand or full HF model ID.
        condition: One of "single_turn", "multi_turn_2", "multi_turn_full".
        tracks: List of track names to evaluate.
        dataset_name: Dataset to load ("bea2025", "mathdial", "bridge").
        output_dir: Directory for results JSON. Defaults to results/exp1/.
        use_api: Unused — model type is auto-detected from model_name.
        api_key: API key for OpenAI/Anthropic models.
        max_samples: Cap on number of instances (useful for testing).
        seed: Random seed.

    Returns:
        Results dict with per-track metrics, predictions, and run metadata.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. Valid: {VALID_CONDITIONS}"
        )
    for track in tracks:
        if track not in VALID_TRACKS:
            raise ValueError(
                f"Unknown track '{track}'. Valid: {VALID_TRACKS}"
            )

    set_seed(seed)

    if output_dir is None:
        output_dir = str(RESULTS_DIR / "exp1")
    ensure_dir(output_dir)

    # Setup log file
    safe_model = model_name.replace("/", "_").replace(".", "_")
    log_path = LOGS_DIR / f"exp1_{safe_model}_{dataset_name}_{condition}.log"
    ensure_dir(LOGS_DIR)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(file_handler)

    logger.info(
        "=== Exp1 Detection | model=%s | condition=%s | tracks=%s | dataset=%s ===",
        model_name, condition, tracks, dataset_name,
    )

    # --- 1. Load data ---
    logger.info("Loading dataset: %s (max_samples=%s)", dataset_name, max_samples)
    instances = _load_dataset(dataset_name, max_samples=max_samples)

    if not instances:
        logger.error("No instances loaded. Aborting.")
        return {"error": "no_instances", "model": model_name, "condition": condition}

    logger.info("Loaded %d instances", len(instances))

    # Build misconception list for Track 5 (needed before model init to avoid delay)
    misconception_list: list[dict] | None = None
    if "misconception_classification" in tracks:
        misconception_list = _build_misconception_list(instances)
        logger.info(
            "Built misconception list with %d entries for Track 5", len(misconception_list)
        )

    # --- 2. Initialize model ---
    logger.info("Initializing model: %s", model_name)
    try:
        model = _build_model(model_name, api_key=api_key)
    except Exception as exc:
        logger.error("Failed to load model '%s': %s", model_name, exc)
        raise

    # --- 3. Run inference ---
    # Per-track accumulators
    track_y_true: dict[str, list[str]] = {t: [] for t in tracks}
    track_y_pred: dict[str, list[str]] = {t: [] for t in tracks}
    raw_outputs: list[dict] = []

    # Progress tracking via tqdm (optional; degrade gracefully if not installed)
    try:
        from tqdm import tqdm
        instance_iter = tqdm(instances, desc=f"{model_name}/{condition}", unit="inst")
    except ImportError:
        instance_iter = instances  # type: ignore[assignment]
        logger.info("tqdm not installed; progress will be logged every 50 samples")

    t0 = time.time()

    for i, instance in enumerate(instance_iter):
        instance_result: dict[str, Any] = {
            "instance_id": instance.id,
            "problem": instance.problem[:300],
            "student_response": instance.student_answer[:300],
            "dataset": instance.metadata.get("source", ""),
            "tracks": {},
        }

        for track in tracks:
            try:
                prompt = _build_prompt(
                    track=track,
                    instance=instance,
                    condition=condition,
                    misconception_list=misconception_list,
                )
                _max_tok = 2048
                _sys = SYSTEM_PROMPT_DETECTION
                if model_name in REASONING_MODELS:
                    _sys += (
                        "\n\nCRITICAL: You MUST begin your response with the "
                        "required label line (e.g. CLASSIFICATION: correct). "
                        "Do NOT provide analysis before the label."
                    )
                response: ModelResponse = model.generate(
                    prompt,
                    system_prompt=_sys,
                    temperature=0.0,
                    max_tokens=_max_tok,
                )
                prediction = parse_detection_response(response.text, track)
            except Exception as exc:
                logger.warning(
                    "Error on instance %s / track %s: %s",
                    instance.id, track, exc,
                )
                prediction = ""
                response = ModelResponse(text="", model_name=model_name, usage={})

            gold = _get_ground_truth(instance, track)
            track_y_true[track].append(gold)
            track_y_pred[track].append(prediction)

            instance_result["tracks"][track] = {
                "prediction": prediction,
                "gold": gold,
                "response_text": response.text[:400],
            }

        raw_outputs.append(instance_result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info(
                "  Processed %d/%d instances (%.1fs elapsed)",
                i + 1, len(instances), elapsed,
            )

    elapsed_total = time.time() - t0
    logger.info("Inference complete. Total time: %.1fs", elapsed_total)

    # --- 4. Compute metrics ---
    logger.info("Computing metrics...")
    all_metrics: dict[str, Any] = {}

    for track in tracks:
        y_true = track_y_true[track]
        y_pred = track_y_pred[track]

        # Filter out instances with no gold label
        labeled_pairs = [(t, p) for t, p in zip(y_true, y_pred) if t]
        if not labeled_pairs:
            logger.warning(
                "Track '%s': no gold labels available — skipping metric computation.",
                track,
            )
            all_metrics[track] = {"n_labeled": 0}
            continue

        yt, yp = zip(*labeled_pairs)
        metrics = _compute_track_metrics(track, list(yt), list(yp))
        metrics["n_labeled"] = len(labeled_pairs)
        metrics["n_total"] = len(y_true)
        all_metrics[track] = metrics

        logger.info("Track %s metrics: %s", track, {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})

    # --- 5. Save results ---
    results: dict[str, Any] = {
        "experiment": "exp1_detection",
        "model": model_name,
        "condition": condition,
        "tracks": tracks,
        "dataset": dataset_name,
        "n_instances": len(instances),
        "elapsed_seconds": round(elapsed_total, 2),
        "seed": seed,
        "metrics": all_metrics,
        "predictions": raw_outputs,
    }

    tracks_tag = "_".join(sorted(tracks))
    out_path = Path(output_dir) / f"{safe_model}_{dataset_name}_{condition}_{tracks_tag}.json"
    save_json(results, out_path)
    logger.info("Results saved to %s", out_path)

    # Print concise summary
    logger.info("=== SUMMARY ===")
    for track, metrics in all_metrics.items():
        primary = (
            metrics.get("macro_f1")
            or metrics.get("f1_binary")
            or metrics.get("detection_f1")
            or metrics.get("accuracy")
        )
        if primary is not None:
            logger.info("  %-30s primary metric = %.4f", track, primary)

    # Remove file handler to avoid duplicate logs in subsequent calls
    logger.removeHandler(file_handler)
    file_handler.close()

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Misconception Detection in Dialogue Context"
    )
    parser.add_argument(
        "--model",
        default="qwen-2.5-7b",
        help="Model name (e.g. qwen-2.5-7b, gpt-4o). Default: qwen-2.5-7b",
    )
    parser.add_argument(
        "--condition",
        default="single_turn",
        choices=VALID_CONDITIONS,
        help="Dialogue context condition. Default: single_turn",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["mistake_identification"],
        choices=VALID_TRACKS,
        help="Which tracks to evaluate. Default: mistake_identification",
    )
    parser.add_argument(
        "--dataset",
        default="bea2025",
        choices=("bea2025", "mathdial", "bridge"),
        help="Dataset to evaluate on. Default: bea2025",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for HF models (auto, cuda, cpu). Default: auto",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for results JSON. Default: results/exp1/",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key for OpenAI/Anthropic models (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of instances for quick testing. Default: use all",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--all_conditions",
        action="store_true",
        help="Run all three conditions sequentially for the given model and tracks.",
    )
    args = parser.parse_args()

    conditions_to_run = list(VALID_CONDITIONS) if args.all_conditions else [args.condition]

    for condition in conditions_to_run:
        logger.info("--- Condition: %s ---", condition)
        run_detection_experiment(
            model_name=args.model,
            condition=condition,
            tracks=args.tracks,
            dataset_name=args.dataset,
            output_dir=args.output_dir,
            api_key=args.api_key,
            max_samples=args.max_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
