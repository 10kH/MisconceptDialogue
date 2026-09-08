#!/usr/bin/env python3
"""Comprehensive dialogue quality scoring for human eval sample selection.

Scores each paired dialogue on multiple dimensions:
  1. Tutor repetition (semantic overlap between consecutive turns)
  2. Student engagement (disengagement / refusal patterns)
  3. Tutor progress (whether tutor introduces new information each turn)
  4. Verification relevance (does verification answer address the original topic)
  5. Turn length (excessive length = truncation risk)
  6. Dialogue coherence (topic drift, nonsense)

Output: scored pairs ranked by quality, ready for stratified sampling.
"""
import json
import re
from pathlib import Path
from collections import Counter

# Reuse existing infrastructure
from scripts.extract_human_eval_samples import load_run, is_clean_dialogue, TUTOR_RUNS

SEED = 42
MAX_TURN_DISPLAY_LEN = 2000  # Truncation threshold for annotation display


def ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """Compute n-gram overlap ratio between two texts."""
    def get_ngrams(text, n):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    if not text1.strip() or not text2.strip():
        return 0.0
    ng1 = get_ngrams(text1, n)
    ng2 = get_ngrams(text2, n)
    if not ng1 or not ng2:
        return 0.0
    intersection = ng1 & ng2
    return len(intersection) / min(len(ng1), len(ng2))


def score_tutor_repetition(tutor_turns: list[str]) -> float:
    """Score tutor repetition. 0=no repetition (good), 1=fully repetitive (bad).

    Checks consecutive tutor turns for high n-gram overlap.
    """
    if len(tutor_turns) <= 1:
        return 0.0
    overlaps = []
    for i in range(1, len(tutor_turns)):
        overlap = ngram_overlap(tutor_turns[i], tutor_turns[i-1], n=3)
        overlaps.append(overlap)
    return max(overlaps) if overlaps else 0.0


def score_student_engagement(student_turns: list[str]) -> float:
    """Score student disengagement. 0=engaged (good), 1=disengaged (bad).

    Detects refusal, boredom, and disengagement patterns.
    """
    disengagement_patterns = [
        r"can we (just )?(move on|skip|go to)",
        r"i don'?t (care|know|see|understand)",
        r"let'?s just move on",
        r"whatever",
        r"\*sigh\*",
        r"\*shrug",
        r"\*looks away",
        r"i'?m (done|tired|bored)",
        r"this (doesn'?t|does not) make sense",
        r"can you just (tell|give|show) me",
    ]

    total_flags = 0
    for turn in student_turns:
        turn_lower = turn.lower()
        for pattern in disengagement_patterns:
            if re.search(pattern, turn_lower):
                total_flags += 1
                break  # Count at most once per turn

    return min(total_flags / max(len(student_turns), 1), 1.0)


def score_tutor_progress(tutor_turns: list[str]) -> float:
    """Score whether tutor introduces new content each turn. 0=good progress, 1=stalling.

    Detects stalling patterns like repeated "let's think about this".
    """
    stalling_phrases = [
        "let's think about this",
        "let's explore this further",
        "let's revisit",
        "you're making great progress",
        "you're on the right track",
        "that's a good point",
        "let's think about this further",
        "let's focus on",
        "you've made some great",
    ]

    stall_count = 0
    for turn in tutor_turns:
        turn_lower = turn.lower()
        for phrase in stalling_phrases:
            if phrase in turn_lower:
                stall_count += 1
                break

    # Also check if tutor introduces new unique content
    if len(tutor_turns) >= 3:
        # Compare first and last tutor turn — if too similar, tutor made no progress
        first_last_overlap = ngram_overlap(tutor_turns[0], tutor_turns[-1], n=4)
        if first_last_overlap > 0.5:
            stall_count += 1

    return min(stall_count / max(len(tutor_turns), 1), 1.0)


def score_verification_relevance(verification: str, tutor_turns: list[str],
                                  student_turns: list[str]) -> float:
    """Score whether verification answer is on-topic. 0=relevant (good), 1=off-topic (bad).

    Checks if verification answer shares vocabulary with the dialogue.
    """
    if not verification.strip():
        return 1.0  # Empty verification = bad

    # Combine all dialogue content
    dialogue_text = " ".join(tutor_turns + student_turns)

    # Check topic overlap via keyword intersection
    def get_content_words(text):
        words = re.findall(r'\b[a-z]+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
                     'neither', 'each', 'every', 'all', 'any', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'only', 'own', 'same',
                     'than', 'too', 'very', 'just', 'because', 'if', 'then',
                     'that', 'this', 'what', 'which', 'who', 'whom', 'how',
                     'when', 'where', 'why', 'i', 'me', 'my', 'we', 'you',
                     'your', 'he', 'she', 'it', 'they', 'them', 'its', 'our',
                     'their', 'about', 'also', 'think', 'let', 'know', 'like',
                     'right', 'well', 'now', 'still', 'said', 'say', 'get',
                     'make', 'one', 'two', 'three'}
        return set(w for w in words if w not in stopwords and len(w) > 2)

    dial_words = get_content_words(dialogue_text)
    verif_words = get_content_words(verification)

    if not verif_words:
        return 0.5  # Can't assess

    overlap = len(dial_words & verif_words) / len(verif_words)
    return max(0, 1.0 - overlap * 2)  # High overlap = good (low score)


def score_turn_length(result: dict) -> float:
    """Score based on max turn length. 0=good (short enough), 1=bad (needs truncation)."""
    max_len = 0
    for t in result.get("tutor_responses", []):
        max_len = max(max_len, len(t))
    for s in result.get("student_responses", []):
        max_len = max(max_len, len(s))
    max_len = max(max_len, len(result.get("verification_answer", "")))

    if max_len <= MAX_TURN_DISPLAY_LEN:
        return 0.0
    elif max_len <= 3000:
        return 0.3
    elif max_len <= 5000:
        return 0.6
    else:
        return 1.0


def score_dialogue(result: dict) -> dict:
    """Compute all quality scores for a single dialogue."""
    tutor = result.get("tutor_responses", [])
    student = result.get("student_responses", [])
    verif = result.get("verification_answer", "")

    return {
        "tutor_repetition": score_tutor_repetition(tutor),
        "student_engagement": score_student_engagement(student),
        "tutor_progress": score_tutor_progress(tutor),
        "verification_relevance": score_verification_relevance(verif, tutor, student),
        "turn_length": score_turn_length(result),
    }


def composite_score(scores_same: dict, scores_gpt5: dict) -> float:
    """Compute composite quality score for a pair. Lower = better."""
    weights = {
        "tutor_repetition": 2.0,
        "student_engagement": 1.5,
        "tutor_progress": 1.5,
        "verification_relevance": 1.0,
        "turn_length": 1.0,
    }
    total = 0
    for key, w in weights.items():
        # Take the worse of the two dialogues
        val = max(scores_same.get(key, 0), scores_gpt5.get(key, 0))
        total += val * w
    return total / sum(weights.values())


def main():
    all_scored = []

    for tutor, runs in TUTOR_RUNS.items():
        same = load_run(runs["same_scale"])
        gpt5 = load_run(runs["gpt5_mini"])

        for key in same:
            if key not in gpt5:
                continue
            s, g = same[key], gpt5[key]
            strategy, mtype, instance_id = key

            # Basic clean check
            clean_s = is_clean_dialogue(s)
            clean_g = is_clean_dialogue(g)

            # Category
            s_ok = s["remediation_success"]
            g_ok = g["remediation_success"]
            if s_ok and g_ok:
                cat = "concordant_success"
            elif not s_ok and not g_ok:
                cat = "concordant_failure"
            elif s_ok and not g_ok:
                cat = "discordant_same_yes_gpt_no"
            else:
                cat = "discordant_same_no_gpt_yes"

            # Quality scores
            scores_s = score_dialogue(s)
            scores_g = score_dialogue(g)
            composite = composite_score(scores_s, scores_g)

            all_scored.append({
                "tutor": tutor,
                "strategy": strategy,
                "misconception_type": mtype,
                "instance_id": instance_id,
                "category": cat,
                "basic_clean": clean_s and clean_g,
                "composite_score": composite,
                "scores_same": scores_s,
                "scores_gpt5": scores_g,
                "same_scale": s,
                "gpt5_mini": g,
            })

    # Sort by composite score (lower = better)
    all_scored.sort(key=lambda x: x["composite_score"])

    # Report distribution
    print("=" * 70)
    print("DIALOGUE QUALITY AUDIT — ALL 540 PAIRS")
    print("=" * 70)

    # Score distribution
    scores = [p["composite_score"] for p in all_scored]
    print(f"\nComposite score distribution (0=best, 1=worst):")
    for threshold in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        n = sum(1 for s in scores if s <= threshold)
        print(f"  score <= {threshold:.2f}: {n:>3} pairs")

    # By category at different quality thresholds
    print(f"\n{'='*70}")
    print("BY CATEGORY AT QUALITY THRESHOLDS")
    print(f"{'='*70}")
    for threshold in [0.15, 0.20, 0.25, 0.30]:
        eligible = [p for p in all_scored if p["composite_score"] <= threshold and p["basic_clean"]]
        cats = Counter(p["category"] for p in eligible)
        print(f"\n  threshold={threshold:.2f} (clean + quality):")
        for cat in sorted(cats):
            print(f"    {cat}: {cats[cat]}")
        print(f"    TOTAL: {sum(cats.values())}")

    # Top 100 selection preview
    print(f"\n{'='*70}")
    print("TOP 100 SELECTION PREVIEW")
    print(f"{'='*70}")

    # Select best clean pairs, stratified
    clean_scored = [p for p in all_scored if p["basic_clean"]]

    # Tier 1: best quality from rare categories
    tier1_cats = ["concordant_success", "discordant_same_yes_gpt_no", "discordant_same_no_gpt_yes"]
    tier1 = [p for p in clean_scored if p["category"] in tier1_cats]
    tier1.sort(key=lambda x: x["composite_score"])

    # Show tier 1 quality
    print(f"\nTier 1 (rare categories, sorted by quality):")
    for p in tier1:
        print(f"  {p['composite_score']:.3f}  {p['category']:30s}  {p['tutor']:15s}  {p['strategy']}_{p['misconception_type']}")

    # Suggested cutoff
    print(f"\nTier 1 candidates: {len(tier1)}")
    good_tier1 = [p for p in tier1 if p["composite_score"] <= 0.25]
    print(f"Tier 1 quality<=0.25: {len(good_tier1)}")
    cats_t1 = Counter(p["category"] for p in good_tier1)
    for cat, n in sorted(cats_t1.items()):
        print(f"  {cat}: {n}")

    # Save full scored data
    output = []
    for p in all_scored:
        output.append({
            "tutor": p["tutor"],
            "strategy": p["strategy"],
            "misconception_type": p["misconception_type"],
            "instance_id": p["instance_id"],
            "category": p["category"],
            "basic_clean": p["basic_clean"],
            "composite_score": round(p["composite_score"], 4),
            "scores_same": {k: round(v, 4) for k, v in p["scores_same"].items()},
            "scores_gpt5": {k: round(v, 4) for k, v in p["scores_gpt5"].items()},
        })

    out_path = Path("paper/human_eval/quality_scores.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull scores saved to {out_path}")


if __name__ == "__main__":
    main()
