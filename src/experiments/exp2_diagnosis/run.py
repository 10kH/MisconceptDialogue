"""Experiment 2: Misconception Diagnosis Accuracy

Tests multiple conditions:
  A. zero_shot          — zero-shot prompt over full taxonomy
  B. few_shot_5         — 5-shot in-context examples
  C. cot_rubric         — chain-of-thought + rubric
  D. rag                — retrieve-then-rank with sentence-transformers

  E. gen_retrieve_rerank— generate diagnosis → retrieve → rerank

Evaluates using the closed-set misconception taxonomy used in the paper.
Metrics: acc@1, acc@5, MAP@25, MRR — overall and split by type / seen-unseen.
Results saved to results/exp2/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from src.evaluation.metrics import (
    accuracy_at_k,
    compute_all_diagnosis_metrics,
    mean_average_precision_at_k,
    mrr,
)
from src.utils.common import (
    DATASETS_DIR,
    RESULTS_DIR,
    ensure_dir,
    load_json,
    save_json,
    set_seed,
)
from src.experiments.exp2_diagnosis.prompts import (
    build_cot_rubric_prompt,
    build_few_shot_prompt,
    build_gen_retrieve_rerank_prompt,
    build_rag_prompt,
    build_rerank_prompt,
    build_zero_shot_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CONDITIONS = ("zero_shot", "few_shot_5", "cot_rubric", "rag", "gen_retrieve_rerank")

OPENAI_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
    "o1-mini", "o1-preview", "o3-mini",
}

# Models that produce <think>...</think> reasoning blocks and need higher max_tokens
REASONING_MODELS = {"phi-4-reasoning", "qwen3.5-9b"}

# EEDI domain keywords for taxonomy entry construction
_EEDI_DOMAIN_MAP = {
    "fraction": "fractions",
    "ratio": "fractions",
    "proportion": "fractions",
    "decimal": "fractions",
    "algebra": "algebra",
    "equation": "algebra",
    "variable": "algebra",
    "expression": "algebra",
    "geometry": "geometry",
    "angle": "geometry",
    "area": "geometry",
    "perimeter": "geometry",
    "shape": "geometry",
    "triangle": "geometry",
    "circle": "geometry",
    "statistic": "statistics",
    "probability": "statistics",
    "mean": "statistics",
    "median": "statistics",
    "function": "functions",
    "graph": "functions",
    "coordinate": "functions",
}

_EEDI_TYPE_MAP = {
    "number": "factual",
    "fact": "factual",
    "definition": "factual",
    "procedure": "procedural",
    "algorithm": "procedural",
    "rule": "procedural",
    "concept": "conceptual",
    "understand": "conceptual",
    "misunderstand": "conceptual",
    "represent": "conceptual",
}


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def _infer_domain(text: str) -> str:
    """Infer domain from misconception text using keyword matching."""
    text_lower = text.lower()
    for keyword, domain in _EEDI_DOMAIN_MAP.items():
        if keyword in text_lower:
            return domain
    return "algebra"  # default


def _infer_type(text: str) -> str:
    """Infer error type from misconception text using keyword matching."""
    text_lower = text.lower()
    for keyword, etype in _EEDI_TYPE_MAP.items():
        if keyword in text_lower:
            return etype
    return "conceptual"  # default


def load_misconception_taxonomy() -> list[dict]:
    """Load the 150-core misconception taxonomy from EEDI / benchmark data.

    Primary source: datasets/misconceptions_benchmark/repo/data/data.json
    Secondary source: datasets/eedi/ (Kaggle download — may not exist).

    Returns:
        List of dicts with keys: id, name, description, type, domain.
    """
    taxonomy: list[dict] = []
    seen_ids: set[str] = set()

    # --- Source 1: misconceptions_benchmark data.json ---
    benchmark_path = (
        DATASETS_DIR
        / "misconceptions_benchmark"
        / "repo"
        / "data"
        / "data.json"
    )
    if benchmark_path.exists():
        raw = load_json(benchmark_path)
        for item in raw:
            mid = item.get("Misconception ID", "")
            if not mid or mid in seen_ids:
                continue
            seen_ids.add(mid)
            desc = item.get("Misconception", "").strip()
            topic = item.get("Topic", "").strip()
            taxonomy.append(
                {
                    "id": mid,
                    "name": f"Misconception {mid}",
                    "description": desc,
                    "type": _infer_type(desc),
                    "domain": _infer_domain(desc + " " + topic),
                    "topic": topic,
                    "source": "benchmark",
                }
            )
        logger.info("Loaded %d entries from misconceptions_benchmark", len(taxonomy))

    # --- Source 2: EEDI Kaggle CSV (optional) ---
    eedi_misconceptions_csv = DATASETS_DIR / "eedi" / "misconception_mapping.csv"
    if eedi_misconceptions_csv.exists():
        try:
            import csv
            with open(eedi_misconceptions_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mid = row.get("MisconceptionId", "").strip()
                    name = row.get("MisconceptionName", "").strip()
                    if not mid or mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    taxonomy.append(
                        {
                            "id": mid,
                            "name": name,
                            "description": name,
                            "type": _infer_type(name),
                            "domain": _infer_domain(name),
                            "source": "eedi",
                        }
                    )
            logger.info(
                "Total taxonomy entries after EEDI merge: %d", len(taxonomy)
            )
        except Exception as exc:
            logger.warning("Could not load EEDI misconception CSV: %s", exc)

    if not taxonomy:
        logger.warning(
            "No taxonomy data found. Using placeholder entries for testing."
        )
        # Minimal placeholder so the runner is still exercisable
        taxonomy = [
            {
                "id": f"M{i:03d}",
                "name": f"Placeholder misconception {i}",
                "description": f"Placeholder description for misconception {i}.",
                "type": ["factual", "procedural", "conceptual"][i % 3],
                "domain": ["fractions", "algebra", "geometry", "statistics", "functions"][i % 5],
                "source": "placeholder",
            }
            for i in range(1, 151)
        ]

    # Truncate / cap at 150 core entries if very large
    if len(taxonomy) > 150:
        taxonomy = taxonomy[:150]

    return taxonomy


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_eedi_samples() -> list[dict]:
    """Load EEDI problem-answer pairs from available sources.

    Returns list of dicts with keys:
        problem, student_answer, correct_answer, misconception_id,
        question_id, subject (optional), seen (bool — in train split).
    """
    samples: list[dict] = []

    # Source: EEDI Kaggle train.csv / test.csv
    eedi_dir = DATASETS_DIR / "eedi"
    for csv_name in ("train.csv", "test.csv"):
        csv_path = eedi_dir / csv_name
        if not csv_path.exists():
            continue
        try:
            import csv
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # EEDI Kaggle schema columns vary — handle both known schemas
                    problem = (
                        row.get("QuestionText")
                        or row.get("ConstructDescription", "")
                        or row.get("question", "")
                    ).strip()
                    student_answer = (
                        row.get("Answer", "") or row.get("student_answer", "")
                    ).strip()
                    misconception_id = (
                        row.get("MisconceptionId", "")
                        or row.get("misconception_id", "")
                    ).strip()
                    if not problem or not misconception_id:
                        continue
                    samples.append(
                        {
                            "problem": problem,
                            "student_answer": student_answer,
                            "correct_answer": row.get("CorrectAnswer", ""),
                            "misconception_id": misconception_id,
                            "question_id": row.get("QuestionId", ""),
                            "seen": csv_name == "train.csv",
                        }
                    )
            logger.info("Loaded %d samples from %s", len(samples), csv_name)
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_path, exc)

    # Fallback: misconceptions_benchmark data.json (has problem+answer pairs)
    if not samples:
        benchmark_path = (
            DATASETS_DIR
            / "misconceptions_benchmark"
            / "repo"
            / "data"
            / "data.json"
        )
        if benchmark_path.exists():
            raw = load_json(benchmark_path)
            for item in raw:
                mid = item.get("Misconception ID", "")
                problem = item.get("Question", "").strip()
                student_answer = item.get("Incorrect Answer", "").strip()
                if not problem or not mid:
                    continue
                samples.append(
                    {
                        "problem": problem,
                        "student_answer": student_answer,
                        "correct_answer": item.get("Correct Answer", ""),
                        "misconception_id": mid,
                        "question_id": f"{mid}-Ex{item.get('Example Number', 0)}",
                        "seen": True,  # all from benchmark treated as "seen"
                    }
                )
            logger.info(
                "Loaded %d samples from misconceptions_benchmark", len(samples)
            )

    return samples


def load_eedi(max_samples: int | None = None) -> list[dict]:
    """Public loader for EEDI problem-answer pairs (wraps _load_eedi_samples).

    Args:
        max_samples: If set, truncate to this many samples (useful for dev runs).

    Returns:
        List of sample dicts.
    """
    samples = _load_eedi_samples()
    if max_samples is not None:
        samples = samples[:max_samples]
    logger.info("Using %d samples total", len(samples))
    return samples


def load_misconceptions_benchmark(max_samples: int | None = None) -> list[dict]:
    """Load misconceptions benchmark dataset.

    Args:
        max_samples: Optional cap on returned samples.

    Returns:
        List of sample dicts.
    """
    samples = _load_eedi_samples()
    # filter to benchmark source only (all from benchmark json are benchmark)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(model_name: str):
    """Instantiate the appropriate model wrapper for a given model name."""
    if model_name in OPENAI_MODELS:
        from src.models.openai_model import OpenAIModel
        return OpenAIModel(model_name=model_name)

    # HuggingFace / local model
    from src.models.hf_model import HFModel
    # Map shorthand names to HuggingFace model IDs
    hf_id_map = {
        "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
        "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
        "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
        "qwen3.5-9b": "Qwen/Qwen3.5-9B",
        "glm-4.7-flash": "zai-org/GLM-4.7-Flash",
        "llama-4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "phi-4-reasoning": "microsoft/Phi-4-reasoning-plus",
    }
    hf_id = hf_id_map.get(model_name, model_name)
    return HFModel(model_name=hf_id)


# ---------------------------------------------------------------------------
# RAG / embedding helpers
# ---------------------------------------------------------------------------

def _build_rag_index(taxonomy: list[dict]):
    """Build a FAISS-like embedding index over the taxonomy using sentence-transformers.

    Returns:
        (model, embeddings_np) tuple where embeddings_np[i] corresponds to taxonomy[i].
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for the RAG condition. "
            "Install with: pip install sentence-transformers"
        ) from exc

    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    texts = [
        f"{entry['name']}. {entry['description']}" for entry in taxonomy
    ]
    logger.info("Encoding %d taxonomy entries for RAG index...", len(texts))
    embeddings = embed_model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return embed_model, embeddings


def _rag_retrieve(
    query: str,
    embed_model,
    taxonomy_embeddings: np.ndarray,
    taxonomy: list[dict],
    top_k: int = 25,
) -> list[dict]:
    """Retrieve top-k misconceptions for a query using cosine similarity.

    Args:
        query: Query string (problem + student answer, or generated diagnosis).
        embed_model: SentenceTransformer model.
        taxonomy_embeddings: Pre-computed taxonomy embeddings (N, D).
        taxonomy: List of taxonomy entry dicts.
        top_k: Number of entries to retrieve.

    Returns:
        List of dicts (taxonomy entries with added similarity_score key), ranked.
    """
    query_emb = embed_model.encode([query], normalize_embeddings=True)[0]
    scores = taxonomy_embeddings @ query_emb  # cosine similarity (normalized)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        entry = dict(taxonomy[idx])
        entry["similarity_score"] = float(scores[idx])
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_diagnosis_response(response_text: str) -> list[str]:
    """Parse model response into a ranked list of misconception IDs.

    Looks for the RANKED_MISCONCEPTIONS: block and extracts IDs line by line.
    Falls back to scanning the entire text for ID-like tokens (e.g. MaE01, M001).

    Args:
        response_text: Raw text from the model.

    Returns:
        Ordered list of misconception ID strings (may be empty if none found).
    """
    import re

    ids: list[str] = []

    # Primary: find the RANKED_MISCONCEPTIONS block
    block_match = re.search(
        r"RANKED_MISCONCEPTIONS:\s*\n((?:[ \t]*\S+[ \t]*\n?)+)",
        response_text,
        re.IGNORECASE,
    )
    if block_match:
        block = block_match.group(1)
        for line in block.splitlines():
            token = line.strip()
            if token:
                ids.append(token)
        return ids

    # Fallback: scan for ID-like tokens anywhere in the response
    # Matches patterns like MaE01, M001, MISC-001, misconception_id_42, etc.
    pattern = re.compile(
        r"\b(Ma[A-Z]\d{2,4}|M\d{3,4}|MISC[-_]?\d{2,4}|[A-Z]{1,4}\d{3,4})\b"
    )
    seen: set[str] = set()
    for match in pattern.finditer(response_text):
        token = match.group(1)
        if token not in seen:
            seen.add(token)
            ids.append(token)

    return ids


# ---------------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------------

def _build_few_shot_examples(
    samples: list[dict],
    taxonomy: list[dict],
    n: int = 5,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Sample n diverse examples from the dataset to use as few-shot demonstrations.

    Tries to cover different misconception types and domains.

    Args:
        samples: All dataset samples.
        taxonomy: Full taxonomy (for type/domain lookup).
        n: Number of examples to select.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        List of example dicts with keys: problem, student_answer,
        correct_misconception_id, explanation.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    tax_by_id = {e["id"]: e for e in taxonomy}

    # Stratify by type
    by_type: dict[str, list[dict]] = {}
    for s in samples:
        mid = s.get("misconception_id", "")
        entry = tax_by_id.get(mid, {})
        etype = entry.get("type", "conceptual")
        by_type.setdefault(etype, []).append(s)

    selected: list[dict] = []
    types = list(by_type.keys())
    per_type = max(1, n // len(types)) if types else n

    for etype in types:
        pool = by_type[etype]
        k = min(per_type, len(pool))
        chosen = rng.choice(len(pool), size=k, replace=False)
        for idx in chosen:
            s = pool[idx]
            mid = s.get("misconception_id", "")
            entry = tax_by_id.get(mid, {})
            selected.append(
                {
                    "problem": s["problem"],
                    "student_answer": s["student_answer"],
                    "correct_misconception_id": mid,
                    "explanation": entry.get("description", ""),
                }
            )
        if len(selected) >= n:
            break

    # Top-up if needed
    if len(selected) < n:
        remaining = [s for s in samples if s not in selected]
        extra = min(n - len(selected), len(remaining))
        for i in rng.choice(len(remaining), size=extra, replace=False):
            s = remaining[i]
            mid = s.get("misconception_id", "")
            entry = tax_by_id.get(mid, {})
            selected.append(
                {
                    "problem": s["problem"],
                    "student_answer": s["student_answer"],
                    "correct_misconception_id": mid,
                    "explanation": entry.get("description", ""),
                }
            )

    return selected[:n]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_split_metrics(
    y_true: list[str],
    y_pred_ranked: list[list[str]],
    split_labels: list[str],
) -> dict[str, dict]:
    """Compute diagnosis metrics split by a label column.

    Args:
        y_true: Ground-truth misconception IDs.
        y_pred_ranked: Ranked prediction lists.
        split_labels: Label for each sample (e.g. error type or seen/unseen).

    Returns:
        Dict mapping label -> metric dict.
    """
    from collections import defaultdict

    groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for true, preds, label in zip(y_true, y_pred_ranked, split_labels):
        groups[label][0].append(true)
        groups[label][1].append(preds)

    return {
        label: compute_all_diagnosis_metrics(trues, preds_list)
        for label, (trues, preds_list) in groups.items()
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def _adjust_tokens(model_name: str, base: int) -> int:
    """Return token budget (think blocks are now skipped at model level)."""
    return base


def run_diagnosis_experiment(
    model_name: str,
    condition: str,
    output_dir: str | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Run the diagnosis experiment for one model / condition pair.

    Args:
        model_name: Model shorthand (e.g. "qwen-2.5-7b", "gpt-4o-mini").
        condition: One of VALID_CONDITIONS.
        output_dir: Directory for saving results JSON. Defaults to results/exp2/.
        max_samples: Cap on dataset size (useful for smoke tests).

    Returns:
        Results dict with metrics, predictions, and metadata.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Unknown condition '{condition}'. Valid: {VALID_CONDITIONS}"
        )

    if output_dir is None:
        output_dir = str(RESULTS_DIR / "exp2")
    ensure_dir(output_dir)

    logger.info("=== Exp2 Diagnosis | model=%s | condition=%s ===", model_name, condition)

    # --- Load data ---
    taxonomy = load_misconception_taxonomy()
    samples = load_eedi(max_samples=max_samples)

    if not samples:
        logger.error(
            "No samples loaded. Check dataset availability at %s", DATASETS_DIR
        )
        return {"error": "no_samples"}

    logger.info("Taxonomy entries: %d | Samples: %d", len(taxonomy), len(samples))

    # --- Build model ---
    logger.info("Loading model: %s", model_name)
    model = _build_model(model_name)

    # --- Condition-specific setup ---
    embed_model = None
    taxonomy_embeddings = None
    few_shot_examples: list[dict] = []
    rng = np.random.default_rng(42)

    if condition == "rag" or condition == "gen_retrieve_rerank":
        embed_model, taxonomy_embeddings = _build_rag_index(taxonomy)

    if condition == "few_shot_5":
        few_shot_examples = _build_few_shot_examples(samples, taxonomy, n=5, rng=rng)
        logger.info("Built %d few-shot examples", len(few_shot_examples))

    # --- Run inference ---
    y_true: list[str] = []
    y_pred_ranked: list[list[str]] = []
    type_labels: list[str] = []
    seen_labels: list[str] = []
    raw_outputs: list[dict] = []

    tax_by_id = {e["id"]: e for e in taxonomy}

    for i, sample in enumerate(samples):
        problem = sample["problem"]
        student_answer = sample["student_answer"]
        true_id = sample["misconception_id"]

        logger.debug("[%d/%d] Sample %s", i + 1, len(samples), sample.get("question_id", ""))

        try:
            if condition == "zero_shot":
                prompt = build_zero_shot_prompt(problem, student_answer, taxonomy)
                response = model.generate(prompt, max_tokens=_adjust_tokens(model_name, 512))
                preds = parse_diagnosis_response(response.text)

            elif condition == "few_shot_5":
                prompt = build_few_shot_prompt(
                    problem, student_answer, taxonomy, few_shot_examples, n_shots=5
                )
                response = model.generate(prompt, max_tokens=_adjust_tokens(model_name, 512))
                preds = parse_diagnosis_response(response.text)

            elif condition == "cot_rubric":
                prompt = build_cot_rubric_prompt(problem, student_answer, taxonomy)
                response = model.generate(prompt, max_tokens=_adjust_tokens(model_name, 768))
                preds = parse_diagnosis_response(response.text)

            elif condition == "rag":
                query = f"{problem} {student_answer}"
                retrieved = _rag_retrieve(
                    query, embed_model, taxonomy_embeddings, taxonomy, top_k=25
                )
                prompt = build_rag_prompt(problem, student_answer, retrieved)
                response = model.generate(prompt, max_tokens=_adjust_tokens(model_name, 512))
                preds = parse_diagnosis_response(response.text)
                # Fall back to retrieval order if parsing yields nothing
                if not preds:
                    preds = [r["id"] for r in retrieved]

            elif condition == "gen_retrieve_rerank":
                # Step 1: generate diagnosis
                gen_prompt = build_gen_retrieve_rerank_prompt(problem, student_answer)
                gen_response = model.generate(gen_prompt, max_tokens=_adjust_tokens(model_name, 256))
                candidate_diagnosis = gen_response.text.strip()

                # Step 2: retrieve using diagnosis as query
                retrieved = _rag_retrieve(
                    candidate_diagnosis,
                    embed_model,
                    taxonomy_embeddings,
                    taxonomy,
                    top_k=25,
                )

                # Step 3: rerank
                rerank_prompt = build_rerank_prompt(
                    problem, student_answer, candidate_diagnosis, retrieved
                )
                rerank_response = model.generate(rerank_prompt, max_tokens=_adjust_tokens(model_name, 512))
                preds = parse_diagnosis_response(rerank_response.text)
                # Fall back to retrieval order if parsing yields nothing
                if not preds:
                    preds = [r["id"] for r in retrieved]

                response = rerank_response  # use rerank response for logging

            else:
                raise ValueError(f"Unhandled condition: {condition}")

        except Exception as exc:
            logger.warning("Error on sample %s: %s", sample.get("question_id", i), exc)
            preds = []
            response = type("_R", (), {"text": "", "usage": {}})()

        # Ensure all taxonomy IDs are available as fallback at the end
        all_ids = [e["id"] for e in taxonomy]
        # Append any taxonomy IDs not already in preds (so list is complete for MAP@25)
        preds_set = set(preds)
        for tid in all_ids:
            if tid not in preds_set:
                preds.append(tid)

        y_true.append(true_id)
        y_pred_ranked.append(preds)

        # Labels for sub-group analysis
        entry = tax_by_id.get(true_id, {})
        type_labels.append(entry.get("type", "unknown"))
        seen_labels.append("seen" if sample.get("seen", True) else "unseen")

        raw_outputs.append(
            {
                "question_id": sample.get("question_id", str(i)),
                "problem": problem,
                "student_answer": student_answer,
                "true_misconception_id": true_id,
                "predicted_ranked": preds[:25],
                "response_text": getattr(response, "text", "")[:500],
            }
        )

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d/%d samples", i + 1, len(samples))

    # --- Compute metrics ---
    logger.info("Computing metrics...")
    overall_metrics = compute_all_diagnosis_metrics(y_true, y_pred_ranked)
    by_type = _compute_split_metrics(y_true, y_pred_ranked, type_labels)
    by_seen = _compute_split_metrics(y_true, y_pred_ranked, seen_labels)

    results = {
        "experiment": "exp2_diagnosis",
        "model": model_name,
        "condition": condition,
        "n_samples": len(samples),
        "n_taxonomy": len(taxonomy),
        "metrics": {
            "overall": overall_metrics,
            "by_type": by_type,
            "by_seen": by_seen,
        },
        "predictions": raw_outputs,
    }

    # --- Save ---
    safe_model = model_name.replace("/", "_").replace(".", "_")
    out_path = Path(output_dir) / f"{safe_model}_{condition}.json"
    save_json(results, out_path)
    logger.info("Results saved to %s", out_path)

    # Print summary
    m = overall_metrics
    logger.info(
        "RESULTS | acc@1=%.4f | acc@5=%.4f | MAP@25=%.4f | MRR=%.4f",
        m.get("acc_at_1", 0),
        m.get("acc_at_5", 0),
        m.get("map_at_25", 0),
        m.get("mrr", 0),
    )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: Misconception Diagnosis Accuracy"
    )
    parser.add_argument(
        "--model",
        default="qwen-2.5-7b",
        help="Model name (e.g. qwen-2.5-7b, gpt-4o-mini). Default: qwen-2.5-7b",
    )
    parser.add_argument(
        "--condition",
        default="zero_shot",
        choices=VALID_CONDITIONS,
        help=f"Experiment condition. One of: {', '.join(VALID_CONDITIONS)}",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap dataset at N samples (for development/smoke tests).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for result files. Default: results/exp2/",
    )
    parser.add_argument(
        "--all_conditions",
        action="store_true",
        help="Run all conditions sequentially for the given model.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    conditions_to_run = list(VALID_CONDITIONS) if args.all_conditions else [args.condition]

    for condition in conditions_to_run:
        logger.info("--- Running condition: %s ---", condition)
        run_diagnosis_experiment(
            model_name=args.model,
            condition=condition,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
