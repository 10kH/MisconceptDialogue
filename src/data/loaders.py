"""Dataset loaders for MisconceptDialogue experiments.

Loaders read each dataset from disk and return standardized TutoringInstance
objects (or plain dicts for non-dialogue datasets).

Dataset schemas (discovered from actual files):

Bridge (JSON list, 419 train items):
  keys: c_id, lesson_topic, c_h, c_r, c_r_, c_revision, e, z_what, z_why
  c_h / c_r: list of {id, text, time, user}
  c_revision: list of {user, text, is_revised}
  e: expert decision label ("guess", "diagnose", ...)
  z_what / z_why: strategy annotations

BEA2025 mrbench_v3 (JSON list, 300 dev / 191 test):
  keys: conversation_id, conversation_history, tutor_responses
  conversation_history: plain string "Tutor: ...\n Student: ..."
  tutor_responses: {model_name: {response, annotation: {Mistake_Identification,
                    Mistake_Location, Providing_Guidance, Actionability}}}

MathDial (HuggingFace DatasetDict at datasets/mathdial/hf_cache/):
  splits: train (2262), test (599)
  features: qid, scenario, question, ground_truth, student_incorrect_solution,
            student_profile, teacher_described_confusion, self-correctness,
            self-typical-confusion, self-typical-interactions, conversation
  conversation: pipe-delimited "Teacher: (type)text|EOM|Student: name text|EOM|..."

EEDI (optional, manual Kaggle download; see scripts/download_datasets.py):
  Kaggle CSVs in datasets/eedi/: train.csv, test.csv, misconception_mapping.csv

Misconceptions Benchmark (JSON list, 220 items, 55 unique misconception IDs):
  keys: Misconception, Misconception ID, Topic, Example Number, Question,
        Incorrect Answer, Correct Answer, Question image, Learner Answer image,
        Correct Answer image, Source, Explanation
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.common import DATASETS_DIR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standardized data format
# ---------------------------------------------------------------------------


@dataclass
class TutoringInstance:
    """Standardized representation of a math tutoring interaction."""

    id: str
    problem: str                  # math problem text
    student_answer: str           # student's (potentially wrong) answer
    correct_answer: str           # gold correct answer
    is_correct: bool              # whether student answer is correct
    conversation: list[dict]      # [{"role": "tutor"/"student", "content": "..."}]
    misconception_label: str      # specific misconception label (if available)
    misconception_type: str       # factual/procedural/conceptual (if available)
    error_description: str        # description of the error
    remediation_strategy: str     # strategy used (if available)
    metadata: dict = field(default_factory=dict)  # additional fields


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------


def _parse_bea2025_conversation(history: str) -> tuple[str, list[dict]]:
    """Parse a BEA2025 conversation_history string into (problem, turns).

    The string format is:
        Tutor: Hi, could you please provide a step-by-step solution for the
               question below? The question is: <problem text>
        Student: <student solution including numeric answer>
        Tutor: <follow-up question>
        Student: <response>
        ...

    Returns:
        problem: Extracted problem text (empty string if not found).
        turns: List of {"role": "tutor"/"student", "content": "..."} dicts.
    """
    turns: list[dict] = []
    problem = ""

    # Split on role-change boundaries: newline(s) followed by "Tutor:" or "Student:"
    raw_parts = re.split(r"\n\s*(?=Tutor:|Student:)", history.strip())

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("Tutor:"):
            content = part[len("Tutor:"):].strip()
            # First tutor turn contains the problem after "The question is:"
            if not problem:
                m = re.search(r"The question is:\s*(.+)", content, re.DOTALL)
                if m:
                    problem = m.group(1).strip()
            turns.append({"role": "tutor", "content": content})
        elif part.startswith("Student:"):
            content = part[len("Student:"):].strip()
            turns.append({"role": "student", "content": content})

    return problem, turns


def _parse_mathdial_conversation(raw: str) -> list[dict]:
    """Parse a MathDial pipe-delimited conversation string into turn dicts.

    Format: "Teacher: (type)text|EOM|Student name: text|EOM|Teacher: ..."
    Parenthetical type annotations like "(probing)" are stripped from content.
    """
    turns: list[dict] = []
    for seg in raw.split("|EOM|"):
        seg = seg.strip()
        if not seg:
            continue
        if seg.startswith("Teacher:"):
            content = seg[len("Teacher:"):].strip()
            content = re.sub(r"^\([^)]*\)\s*", "", content)
            turns.append({"role": "tutor", "content": content})
        elif ":" in seg:
            # Student turns may carry the student's name: "Steven: text"
            colon_idx = seg.index(":")
            content = seg[colon_idx + 1:].strip()
            turns.append({"role": "student", "content": content})
    return turns


def _bridge_msgs_to_turns(msgs: list[dict]) -> list[dict]:
    """Convert a Bridge message list (c_h, c_r, or c_revision) to standard turns."""
    turns: list[dict] = []
    for msg in msgs:
        user = msg.get("user", "")
        text = msg.get("text", "")
        if text:
            role = "tutor" if user == "tutor" else "student"
            turns.append({"role": role, "content": text})
    return turns


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_mathdial(split: str = "train") -> list[TutoringInstance]:
    """Load MathDial dataset from HuggingFace datasets disk cache.

    MathDial is stored as a DatasetDict with 'train' (2262 rows) and
    'test' (599 rows) splits.  Each row contains:
      qid, scenario, question, ground_truth, student_incorrect_solution,
      student_profile, teacher_described_confusion, self-correctness,
      self-typical-confusion, self-typical-interactions, conversation.

    The 'conversation' field is a pipe-delimited string:
      "Teacher: (type)text|EOM|Student name: text|EOM|..."

    Args:
        split: "train" or "test".

    Returns:
        List of TutoringInstance objects.

    Raises:
        FileNotFoundError: If the HF cache directory does not exist.
        ImportError: If the ``datasets`` library is not installed.
        ValueError: If the requested split is not present.
    """
    cache_dir = DATASETS_DIR / "mathdial" / "hf_cache"
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"MathDial HF cache not found at {cache_dir}. "
            "Run: python scripts/download_datasets.py --dataset mathdial"
        )

    try:
        from datasets import load_from_disk  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required to load MathDial. "
            "Install with: pip install datasets"
        ) from exc

    dataset_dict = load_from_disk(str(cache_dir))
    if split not in dataset_dict:
        raise ValueError(
            f"Split '{split}' not in MathDial. Available: {list(dataset_dict.keys())}"
        )

    instances: list[TutoringInstance] = []
    for row in dataset_dict[split]:
        conversation = _parse_mathdial_conversation(row.get("conversation", ""))
        # self-correctness is "Yes"/"No" string indicating if the student
        # self-corrects during the session (not whether the initial answer is right).
        # We set is_correct=False since we only load incorrect solution scenarios.
        is_correct = str(row.get("self-correctness", "No")).strip().lower() == "yes"
        instances.append(TutoringInstance(
            id=str(row["qid"]),
            problem=row.get("question", ""),
            student_answer=row.get("student_incorrect_solution", ""),
            correct_answer=row.get("ground_truth", ""),
            is_correct=is_correct,
            conversation=conversation,
            misconception_label="",
            misconception_type="",
            error_description=row.get("teacher_described_confusion", ""),
            remediation_strategy="",
            metadata={
                "scenario": row.get("scenario"),
                "student_profile": row.get("student_profile", ""),
                "self_typical_confusion": row.get("self-typical-confusion"),
                "self_typical_interactions": row.get("self-typical-interactions"),
                "dataset": "mathdial",
                "split": split,
            },
        ))

    logger.info("Loaded %d MathDial instances (split=%s)", len(instances), split)
    return instances


def load_bridge(split: str = "train") -> list[TutoringInstance]:
    """Load Bridge tutoring dataset from JSON files.

    Bridge contains real tutoring conversations with expert decision
    annotations at the moment a student gives an incorrect answer.

    Each JSON item has:
      c_id, lesson_topic, c_h (history before the error), c_r (actual
      tutor response), c_r_ (revised response head), c_revision (full
      revised conversation), e (expert decision label), z_what, z_why.

    Args:
        split: "train", "test", or "validation".

    Returns:
        List of TutoringInstance objects.

    Raises:
        FileNotFoundError: If the JSON split file does not exist.
        ValueError: If the split name is not recognised.
    """
    valid_splits = {"train", "test", "validation"}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Choose from: {valid_splits}")

    json_path = DATASETS_DIR / "bridge" / "repo" / "dataset" / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Bridge {split} split not found at {json_path}. "
            "Run: python scripts/download_datasets.py --dataset bridge"
        )

    with open(json_path, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    instances: list[TutoringInstance] = []
    for item in raw:
        # Prefer c_revision (full revised conversation) when available;
        # fall back to concatenating c_h + c_r.
        if item.get("c_revision"):
            conversation = _bridge_msgs_to_turns(item["c_revision"])
        else:
            conversation = _bridge_msgs_to_turns(
                item.get("c_h", []) + item.get("c_r", [])
            )

        # Student answer: last student turn in c_h (the error that triggered annotation)
        c_h_student = [m for m in item.get("c_h", []) if m.get("user") == "student"]
        student_answer = c_h_student[-1]["text"] if c_h_student else ""

        lesson_topic = item.get("lesson_topic", "")
        expert_decision = item.get("e", "")

        instances.append(TutoringInstance(
            id=item.get("c_id", ""),
            problem=lesson_topic,       # Bridge has no explicit problem text; use topic
            student_answer=student_answer,
            correct_answer="",          # Bridge does not provide gold answers
            is_correct=False,           # Bridge captures incorrect student responses
            conversation=conversation,
            misconception_label="",
            misconception_type="",
            error_description=expert_decision,
            remediation_strategy=item.get("z_what", ""),
            metadata={
                "lesson_topic": lesson_topic,
                "expert_decision": expert_decision,
                "z_why": item.get("z_why", ""),
                "dataset": "bridge",
                "split": split,
            },
        ))

    logger.info("Loaded %d Bridge instances (split=%s)", len(instances), split)
    return instances


def load_bea2025(split: str = "dev") -> list[TutoringInstance]:
    """Load BEA 2025 shared task data from mrbench_v3 JSON files.

    Each item has:
      conversation_id, conversation_history (plain string with interleaved
      Tutor/Student turns), tutor_responses (dict of model name -> response
      + annotation with Mistake_Identification, Mistake_Location,
      Providing_Guidance, Actionability).

    Args:
        split: "dev" (300 items) or "test" (191 items).

    Returns:
        List of TutoringInstance objects.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If the split name is not recognised.
    """
    split_map = {
        "dev": "mrbench_v3_devset.json",
        "test": "mrbench_v3_testset.json",
    }
    if split not in split_map:
        raise ValueError(f"Invalid split '{split}'. Choose from: {list(split_map)}")

    json_path = (
        DATASETS_DIR
        / "bea2025"
        / "repo"
        / "BEA_Shared_Task_2025_Datasets"
        / split_map[split]
    )
    if not json_path.exists():
        raise FileNotFoundError(
            f"BEA2025 {split} split not found at {json_path}. "
            "Run: python scripts/download_datasets.py --dataset bea2025"
        )

    with open(json_path, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    instances: list[TutoringInstance] = []
    for item in raw:
        history = item.get("conversation_history", "")
        problem, conversation = _parse_bea2025_conversation(history)

        # First student turn is typically the full solution attempt
        student_turns = [t for t in conversation if t["role"] == "student"]
        student_answer = student_turns[0]["content"] if student_turns else ""

        tutor_responses: dict[str, Any] = item.get("tutor_responses", {})

        instances.append(TutoringInstance(
            id=item.get("conversation_id", ""),
            problem=problem,
            student_answer=student_answer,
            correct_answer="",   # BEA2025 does not provide gold answers
            is_correct=False,    # BEA2025 focuses on tutor response quality
            conversation=conversation,
            misconception_label="",
            misconception_type="",
            error_description="",
            remediation_strategy="",
            metadata={
                "tutor_responses": tutor_responses,
                "raw_conversation_history": history,
                "dataset": "bea2025",
                "split": split,
            },
        ))

    logger.info("Loaded %d BEA2025 instances (split=%s)", len(instances), split)
    return instances


def load_eedi() -> list[dict]:
    """Load the optional EEDI Kaggle data as raw CSV rows.

    EEDI contains MCQ-based math questions with distractor-to-misconception mappings.
    It is distributed only through Kaggle, so it must be downloaded manually:
      https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data

    Expected location: datasets/eedi/{train,test}.csv (the Exp. 2 pipeline also reads
    datasets/eedi/misconception_mapping.csv).

    Returns:
        List of row dicts tagged with "split" and "dataset", or an empty list with a
        warning if the CSVs are absent.

    Note:
        Returns dicts rather than TutoringInstance because EEDI is MCQ-based
        and does not contain dialogue/conversation data.
    """
    import csv

    eedi_dir = DATASETS_DIR / "eedi"
    results: list[dict] = []
    for split_name in ("train", "test"):
        path = eedi_dir / f"{split_name}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append(dict(row) | {"split": split_name, "dataset": "eedi"})
    if not results:
        logger.warning(
            "EEDI not found under %s. It is optional; download train.csv/test.csv from "
            "https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data",
            eedi_dir,
        )
        return results
    logger.info("Loaded %d EEDI items", len(results))
    return results


def load_misconceptions_benchmark() -> list[dict]:
    """Load the 55 misconceptions benchmark with diagnostic examples.

    The benchmark has 220 items covering 55 unique misconception IDs
    (MaE01-MaE55).  Each item is a diagnostic example (question + incorrect
    answer) that illustrates a specific misconception pattern.

    Keys in each returned dict (snake_case):
      misconception_id, misconception, topic, example_number, question,
      incorrect_answer, correct_answer, question_image, learner_answer_image,
      correct_answer_image, source, explanation, dataset.

    Returns:
        List of dicts with misconception diagnostic examples.

    Raises:
        FileNotFoundError: If the benchmark data file does not exist.
    """
    data_path = (
        DATASETS_DIR / "misconceptions_benchmark" / "repo" / "data" / "data.json"
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"Misconceptions benchmark not found at {data_path}. "
            "Run: python scripts/download_datasets.py --dataset misconceptions_benchmark"
        )

    with open(data_path, encoding="utf-8") as f:
        raw: list[dict] = json.load(f)

    results: list[dict] = []
    for item in raw:
        results.append({
            "misconception_id": item.get("Misconception ID", ""),
            "misconception": item.get("Misconception", ""),
            "topic": item.get("Topic", ""),
            "example_number": item.get("Example Number"),
            "question": item.get("Question", ""),
            "incorrect_answer": item.get("Incorrect Answer", ""),
            "correct_answer": item.get("Correct Answer", ""),
            "question_image": item.get("Question image", ""),
            "learner_answer_image": item.get("Learner Answer image", ""),
            "correct_answer_image": item.get("Correct Answer image", ""),
            "source": item.get("Source", ""),
            "explanation": item.get("Explanation", ""),
            "dataset": "misconceptions_benchmark",
        })

    logger.info(
        "Loaded %d misconception benchmark items (%d unique IDs)",
        len(results),
        len({r["misconception_id"] for r in results}),
    )
    return results


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def get_dataset(name: str, split: str = "train") -> list:
    """Dispatcher that calls the appropriate loader by dataset name.

    Args:
        name: One of "mathdial", "bridge", "bea2025", "eedi",
              "misconceptions_benchmark".
        split: Split name passed through to the loader.  Ignored for
               "eedi" and "misconceptions_benchmark" which have no splits.

    Returns:
        List of TutoringInstance objects (for dialogue datasets) or
        list of dicts (for eedi and misconceptions_benchmark).

    Raises:
        ValueError: If the dataset name is not recognised.
    """
    loaders: dict[str, Any] = {
        "mathdial": lambda: load_mathdial(split),
        "bridge": lambda: load_bridge(split),
        "bea2025": lambda: load_bea2025(split),
        "eedi": load_eedi,
        "misconceptions_benchmark": load_misconceptions_benchmark,
    }
    if name not in loaders:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(loaders)}"
        )
    return loaders[name]()


# ---------------------------------------------------------------------------
# Experiment formatters
# ---------------------------------------------------------------------------


def create_detection_dataset(instances: list[TutoringInstance]) -> list[dict]:
    """Format TutoringInstances for Experiment 1 (misconception detection).

    Produces a binary classification dataset: given a conversation, predict
    whether a misconception is present (label=1) or not (label=0).

    Args:
        instances: List of TutoringInstance objects.

    Returns:
        List of dicts with keys:
          id, problem, student_answer, correct_answer, conversation_text,
          label, is_correct, misconception_label, error_description, metadata.
    """
    results: list[dict] = []
    for inst in instances:
        conv_text = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in inst.conversation
        )
        # Label: 1 = misconception present, 0 = correct
        label = 0 if inst.is_correct else 1
        results.append({
            "id": inst.id,
            "problem": inst.problem,
            "student_answer": inst.student_answer,
            "correct_answer": inst.correct_answer,
            "conversation_text": conv_text,
            "label": label,
            "is_correct": inst.is_correct,
            "misconception_label": inst.misconception_label,
            "error_description": inst.error_description,
            "metadata": inst.metadata,
        })
    return results


def create_diagnosis_dataset(
    instances: list[TutoringInstance],
    taxonomy: list[dict] | None = None,
) -> list[dict]:
    """Format TutoringInstances for Experiment 2 (misconception diagnosis).

    Produces a classification/generation dataset: given a conversation,
    identify the specific misconception type or label.

    Args:
        instances: List of TutoringInstance objects.
        taxonomy: Optional list of misconception dicts from
                  :func:`load_misconceptions_benchmark`.  When provided,
                  each output item includes a deduplicated
                  ``candidate_misconceptions`` list of
                  ``{id, label, topic}`` dicts.

    Returns:
        List of dicts with keys:
          id, problem, student_answer, correct_answer, conversation_text,
          misconception_label, misconception_type, error_description,
          candidate_misconceptions (if taxonomy provided), metadata.
    """
    candidates: list[dict] = []
    if taxonomy:
        seen: set[str] = set()
        for item in taxonomy:
            mid = item.get("misconception_id", "")
            if mid and mid not in seen:
                seen.add(mid)
                candidates.append({
                    "id": mid,
                    "label": item.get("misconception", ""),
                    "topic": item.get("topic", ""),
                })

    results: list[dict] = []
    for inst in instances:
        conv_text = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in inst.conversation
        )
        entry: dict = {
            "id": inst.id,
            "problem": inst.problem,
            "student_answer": inst.student_answer,
            "correct_answer": inst.correct_answer,
            "conversation_text": conv_text,
            "misconception_label": inst.misconception_label,
            "misconception_type": inst.misconception_type,
            "error_description": inst.error_description,
            "metadata": inst.metadata,
        }
        if candidates:
            entry["candidate_misconceptions"] = candidates
        results.append(entry)
    return results


def create_remediation_dataset(instances: list[TutoringInstance]) -> list[dict]:
    """Format TutoringInstances for Experiment 3 (remediation strategy).

    Produces a generation dataset: given the conversation up to and including
    the last student error turn, generate an appropriate remediation response.

    Args:
        instances: List of TutoringInstance objects.

    Returns:
        List of dicts with keys:
          id, problem, student_answer, correct_answer, conversation_text,
          context_turns, last_student_turn, remediation_strategy,
          error_description, misconception_label, metadata.
    """
    results: list[dict] = []
    for inst in instances:
        conv = inst.conversation

        # Find the last student turn to use as the context cutoff point
        last_student_idx = -1
        for i, turn in enumerate(conv):
            if turn["role"] == "student":
                last_student_idx = i

        context_turns = conv[: last_student_idx + 1] if last_student_idx >= 0 else conv
        last_student_turn = (
            conv[last_student_idx]["content"] if last_student_idx >= 0 else ""
        )

        conv_text = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in context_turns
        )

        results.append({
            "id": inst.id,
            "problem": inst.problem,
            "student_answer": inst.student_answer,
            "correct_answer": inst.correct_answer,
            "conversation_text": conv_text,
            "context_turns": context_turns,
            "last_student_turn": last_student_turn,
            "remediation_strategy": inst.remediation_strategy,
            "error_description": inst.error_description,
            "misconception_label": inst.misconception_label,
            "metadata": inst.metadata,
        })
    return results
