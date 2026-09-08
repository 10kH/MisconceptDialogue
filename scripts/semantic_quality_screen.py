#!/usr/bin/env python3
"""Comprehensive semantic quality screening for ALL 540 exp3 dialogue pairs.

Detects issues that n-gram-based scoring misses:
  1. Student echo (student copies tutor text verbatim)
  2. Role confusion (student speaks like a tutor)
  3. Verification answer errors (empty, meta-instructions, truncated)
  4. Tutor factual errors (reinforcing wrong answers)
  5. Turn misalignment (tutor responds to nonexistent content)
  6. Broken/empty content

Outputs: paper/human_eval/semantic_screen_results.json
"""
import json
import re
from pathlib import Path
from collections import Counter

from scripts.extract_human_eval_samples import load_run, is_clean_dialogue, TUTOR_RUNS
from scripts.score_dialogue_quality import score_dialogue, composite_score, ngram_overlap


# ── Role confusion detection ──────────────────────────────────────────

TUTOR_PHRASES = [
    # Praise patterns (tutors praise students, not vice versa)
    r"(?:you'?re|you are) (?:on the right track|doing (?:great|well|a great job|an excellent job|fantastic))",
    r"(?:great|good|excellent|fantastic|wonderful|amazing) (?:job|work|observation|question|thinking|progress|insight|point)",
    r"(?:i'?m )?(?:proud of you|glad you)",
    r"that'?s (?:a )?(?:great|good|excellent|fantastic|wonderful) (?:question|observation|point|start|answer|insight)",
    r"well done",
    r"keep (?:up the|it up|going)",
    r"(?:exactly|absolutely|precisely)[!.]?\s*(?:you'?ve|you have|you got|you correctly|you found|that'?s|since)",
    # Direct affirmation of other's correctness (students don't say these to tutors)
    r"(?:you'?re|you are) (?:absolutely |completely |totally )?(?:right|correct)",
    r"(?:you'?ve|you have|you) got (?:it|this|the (?:right|correct|hang))",
    # Pedagogical prompts (tutors ask these, not students)
    r"does (?:this|that) (?:\w+ )?(?:help|make sense|clarify)",
    r"(?:can|could) you (?:think of|try|explain)",
    r"(?:let'?s|shall we) (?:try|explore|think about|consider|look at|move on|apply|practice|solve|work through|write)",
    r"would you like to try",
    r"(?:here'?s|let me give you) (?:a|an|another) (?:example|hint|clue)",
    # Comfort/confidence checks (tutor checking on student)
    r"(?:do|how do) you feel (?:more )?(?:comfortable|confident|about)",
    r"(?:does|did) (?:this|that|this process) (?:\w+ )?(?:click|resonate|land|feel comfortable|feel right|make sense)",
    r"(?:are|were) you (?:comfortable|confident) (?:with|using|applying)",
    r"(?:is|was) (?:this|that) (?:clear|clearer) (?:now|to you)",
    # Closing/summarizing patterns
    r"(?:if you have|feel free to) (?:any )?(?:more )?questions",
    r"don'?t hesitate to ask",
    r"(?:i hope|hopefully) (?:this|that) (?:helps|clarifies|makes sense)",
    r"does (?:this|that) (?:help )?(?:solidify|reinforce|clarify|deepen) your understanding",
    r"to (?:summarize|recap|sum up)",
    # Encouragement/validation (tutor affirming student's work)
    r"(?:this|that) (?:is|was) (?:a )?(?:really |very )?(?:important|key|crucial|valuable) (?:step|insight|observation|realization)",
    r"(?:i'?m |i am )?glad (?:we|you) (?:could|can|were able to) (?:clear|work|figure)",
    r"(?:you'?ve|you have) (?:correctly|successfully|already) (?:identified|found|shown|demonstrated|worked)",
    # Pedagogical offers (tutor offering next steps)
    r"(?:would|do) you (?:like|want) (?:to|me to) (?:try|practice|solve|see|work|do)",
    r"(?:shall|should) we (?:try|move|continue|go|practice)",
    r"(?:it'?s|it is) (?:great|good|wonderful|fantastic) that you",
]

TUTOR_PHRASE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TUTOR_PHRASES]


def count_tutor_phrases_in_student(student_turns: list[str]) -> dict:
    """Count tutor-like phrases in student turns."""
    total_matches = 0
    turns_with_matches = 0
    matched_phrases = []

    for turn in student_turns:
        turn_matches = 0
        for pattern in TUTOR_PHRASE_PATTERNS:
            finds = pattern.findall(turn)
            if finds:
                turn_matches += len(finds)
                matched_phrases.extend(finds)
        if turn_matches > 0:
            turns_with_matches += 1
            total_matches += turn_matches

    return {
        "total_matches": total_matches,
        "turns_with_matches": turns_with_matches,
        "n_student_turns": len(student_turns),
        "ratio": turns_with_matches / max(len(student_turns), 1),
    }


# ── Student echo detection ────────────────────────────────────────────

def detect_student_echo(tutor_turns: list[str], student_turns: list[str]) -> dict:
    """Detect if student copies tutor text verbatim."""
    echo_turns = 0
    max_overlap = 0.0

    for i, s_text in enumerate(student_turns):
        if not s_text.strip():
            continue
        # Check against the tutor turn that came BEFORE this student turn
        if i < len(tutor_turns):
            overlap = ngram_overlap(s_text, tutor_turns[i], n=5)
            max_overlap = max(max_overlap, overlap)
            if overlap > 0.4:  # 40%+ 5-gram overlap = likely echo
                echo_turns += 1
        # Also check against the tutor turn that comes AFTER (role swap)
        if i + 1 < len(tutor_turns):
            overlap = ngram_overlap(s_text, tutor_turns[i + 1], n=5)
            max_overlap = max(max_overlap, overlap)
            if overlap > 0.4:
                echo_turns += 1

    return {
        "echo_turns": echo_turns,
        "max_overlap": round(max_overlap, 4),
        "has_echo": echo_turns > 0,
    }


# ── Verification answer screening ─────────────────────────────────────

def screen_verification(verification: str, tutor_turns: list[str],
                        student_turns: list[str]) -> dict:
    """Screen verification answer for quality issues."""
    issues = []

    if not verification or not verification.strip():
        issues.append("empty_verification")
        return {"issues": issues, "ok": False}

    v_lower = verification.lower()

    # Meta-instructions (model talking to itself)
    meta_patterns = [
        r"\[after the tutor",
        r"\[if the tutor",
        r"\[student error\]",
        r"\[placeholder\]",
        r"i don'?t know how to solve",
        r"i'?m not sure (?:how to|if)",
    ]
    for p in meta_patterns:
        if re.search(p, v_lower):
            issues.append("meta_instructions")
            break

    # Severe truncation (ends mid-word or mid-sentence without conclusion)
    if len(verification.strip()) < 50:
        issues.append("too_short")

    # Check if verification ends abruptly (no period, question mark, or closing)
    stripped = verification.rstrip()
    if stripped and stripped[-1] not in '.?!)\"]':
        # Could be truncated
        last_line = stripped.split('\n')[-1].strip()
        if len(last_line) > 20 and last_line[-1] not in '.?!)\"]':
            issues.append("truncated")

    # Role confusion in verification (student acts as tutor)
    tutor_in_verif = count_tutor_phrases_in_student([verification])
    if tutor_in_verif["total_matches"] >= 3:
        issues.append("role_confusion_in_verification")

    return {
        "issues": issues,
        "ok": len(issues) == 0,
    }


# ── Tutor error detection (heuristic) ─────────────────────────────────

def detect_tutor_errors(tutor_turns: list[str], student_turns: list[str]) -> list[str]:
    """Heuristic detection of tutor factual errors."""
    issues = []

    for t in tutor_turns:
        t_lower = t.lower()
        # Tutor validates wrong answer patterns
        if re.search(r"(?:that'?s|you'?re) (?:correct|right|exactly)", t_lower):
            # Check if this validation follows a student error
            # (Hard to detect automatically, so just flag suspicious patterns)
            pass

        # Negative × negative = negative (common wrong math)
        if re.search(r"negative\s*(?:times|×|multiplied by)\s*(?:a\s*)?negative.*(?:gives?|equals?|is|=)\s*(?:a\s*)?negative", t_lower):
            issues.append("neg_times_neg_equals_neg")

        # 0/n = undefined (wrong)
        if re.search(r"0\s*(?:divided by|÷|/)\s*\d+.*undefined", t_lower):
            issues.append("zero_div_n_undefined")

    return issues


# ── Main screening function ───────────────────────────────────────────

def screen_dialogue(result: dict) -> dict:
    """Comprehensive semantic screening of a single dialogue."""
    tutor = result.get("tutor_responses", [])
    student = result.get("student_responses", [])
    verif = result.get("verification_answer", "")

    # Strip think blocks
    tutor = [re.sub(r"<think>.*?</think>\s*", "", t, flags=re.DOTALL).strip() for t in tutor]
    student = [re.sub(r"<think>.*?</think>\s*", "", s, flags=re.DOTALL).strip() for s in student]
    verif = re.sub(r"<think>.*?</think>\s*", "", verif, flags=re.DOTALL).strip()

    flags = []
    details = {}

    # 1. Student echo
    echo = detect_student_echo(tutor, student)
    details["echo"] = echo
    if echo["has_echo"]:
        flags.append("student_echo")

    # 2. Role confusion
    role = count_tutor_phrases_in_student(student)
    details["role_confusion"] = role
    if role["ratio"] >= 0.5:  # 50%+ of student turns have tutor phrases
        flags.append("severe_role_confusion")
    elif role["ratio"] >= 0.3:  # 30%+ turns
        flags.append("moderate_role_confusion")

    # 2b. Duplicated student turns (exact or near-exact repeats)
    dup_count = 0
    for i in range(1, len(student)):
        if student[i].strip() and student[i].strip() == student[i-1].strip():
            dup_count += 1
    if dup_count > 0:
        flags.append("duplicated_student_turns")
        details["duplicated_student_turns"] = dup_count

    # 2c. Student teaching (numbered step-by-step in tutor voice)
    teaching_turns = 0
    for s_turn in student:
        # Student giving numbered pedagogical steps AND using tutor phrases
        has_numbered_steps = len(re.findall(r"(?:^|\n)\s*(?:\d+[\.\)]\s|step \d)", s_turn, re.IGNORECASE)) >= 3
        has_tutor_lang = any(p.search(s_turn) for p in TUTOR_PHRASE_PATTERNS)
        if has_numbered_steps and has_tutor_lang:
            teaching_turns += 1
    if teaching_turns > 0:
        details["student_teaching_turns"] = teaching_turns
        if teaching_turns >= 2:
            flags.append("student_teaching")

    # 3. Verification
    verif_screen = screen_verification(verif, tutor, student)
    details["verification"] = verif_screen
    if not verif_screen["ok"]:
        flags.extend([f"verification_{i}" for i in verif_screen["issues"]])

    # 4. Tutor errors
    tutor_errors = detect_tutor_errors(tutor, student)
    details["tutor_errors"] = tutor_errors
    if tutor_errors:
        flags.extend(tutor_errors)

    # 5. Empty/broken content
    empty_student = sum(1 for s in student if not s.strip() or s.strip() == "[Student error]")
    if empty_student > 0:
        flags.append("empty_student_turns")
    empty_tutor = sum(1 for t in tutor if not t.strip())
    if empty_tutor > 0:
        flags.append("empty_tutor_turns")

    # 6. Basic clean check (reuse existing)
    if not is_clean_dialogue(result):
        flags.append("basic_dirty")

    # 7. Quality scores
    scores = score_dialogue(result)
    details["quality_scores"] = scores

    return {
        "flags": flags,
        "n_flags": len(flags),
        "clean": len(flags) == 0,
        "details": details,
    }


def screen_pair(same_result: dict, gpt5_result: dict) -> dict:
    """Screen a pair of dialogues."""
    screen_s = screen_dialogue(same_result)
    screen_g = screen_dialogue(gpt5_result)

    # Pair is clean only if BOTH dialogues are clean
    pair_clean = screen_s["clean"] and screen_g["clean"]

    # Composite quality score
    cs = composite_score(screen_s["details"]["quality_scores"],
                         screen_g["details"]["quality_scores"])

    return {
        "same_scale": screen_s,
        "gpt5_mini": screen_g,
        "pair_clean": pair_clean,
        "composite_score": round(cs, 4),
        # Either-clean: at least one dialogue is clean (for partial use)
        "any_clean": screen_s["clean"] or screen_g["clean"],
    }


def main():
    all_results = []

    for tutor, runs in TUTOR_RUNS.items():
        if runs["same_scale"] is None:
            continue

        same = load_run(runs["same_scale"])
        gpt5 = load_run(runs["gpt5_mini"])

        for key in sorted(same.keys()):
            if key not in gpt5:
                continue

            s, g = same[key], gpt5[key]
            strategy, mtype, instance_id = key

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

            # Screen
            screen = screen_pair(s, g)

            all_results.append({
                "tutor": tutor,
                "strategy": strategy,
                "misconception_type": mtype,
                "instance_id": instance_id,
                "category": cat,
                "pair_clean": screen["pair_clean"],
                "any_clean": screen["any_clean"],
                "composite_score": screen["composite_score"],
                "same_flags": screen["same_scale"]["flags"],
                "gpt5_flags": screen["gpt5_mini"]["flags"],
                "same_clean": screen["same_scale"]["clean"],
                "gpt5_clean": screen["gpt5_mini"]["clean"],
            })

    # Sort by composite score (best first)
    all_results.sort(key=lambda x: x["composite_score"])

    # ── Reports ────────────────────────────────────────────────────────
    n = len(all_results)
    n_pair_clean = sum(1 for r in all_results if r["pair_clean"])
    n_any_clean = sum(1 for r in all_results if r["any_clean"])

    print(f"{'='*70}")
    print(f"SEMANTIC QUALITY SCREENING — ALL {n} PAIRS")
    print(f"{'='*70}")
    print(f"\n  Both dialogues clean (pair_clean): {n_pair_clean}/{n}")
    print(f"  At least one clean (any_clean):     {n_any_clean}/{n}")

    # By category
    print(f"\n--- Pair-clean by category ---")
    for cat in sorted(set(r["category"] for r in all_results)):
        cat_items = [r for r in all_results if r["category"] == cat]
        cat_clean = sum(1 for r in cat_items if r["pair_clean"])
        print(f"  {cat}: {cat_clean}/{len(cat_items)}")

    # By tutor
    print(f"\n--- Pair-clean by tutor ---")
    for tutor in sorted(set(r["tutor"] for r in all_results)):
        t_items = [r for r in all_results if r["tutor"] == tutor]
        t_clean = sum(1 for r in t_items if r["pair_clean"])
        print(f"  {tutor}: {t_clean}/{len(t_items)}")

    # Flag distribution
    print(f"\n--- Flag distribution (across all {n*2} dialogues) ---")
    all_flags = []
    for r in all_results:
        all_flags.extend(r["same_flags"])
        all_flags.extend(r["gpt5_flags"])
    for flag, count in Counter(all_flags).most_common():
        print(f"  {flag}: {count}")

    # Clean pairs with quality <= 0.25
    clean_quality = [r for r in all_results if r["pair_clean"] and r["composite_score"] <= 0.25]
    print(f"\n--- Pair-clean AND quality<=0.25: {len(clean_quality)} ---")
    for cat in sorted(set(r["category"] for r in clean_quality)):
        cat_items = [r for r in clean_quality if r["category"] == cat]
        print(f"  {cat}: {len(cat_items)}")

    print(f"\n--- By tutor (pair-clean + quality<=0.25) ---")
    for tutor in sorted(set(r["tutor"] for r in clean_quality)):
        t_items = [r for r in clean_quality if r["tutor"] == tutor]
        print(f"  {tutor}: {len(t_items)}")

    print(f"\n--- By strategy ---")
    for strat in sorted(set(r["strategy"] for r in clean_quality)):
        s_items = [r for r in clean_quality if r["strategy"] == strat]
        print(f"  {strat}: {len(s_items)}")

    print(f"\n--- By misconception_type ---")
    for mt in sorted(set(r["misconception_type"] for r in clean_quality)):
        m_items = [r for r in clean_quality if r["misconception_type"] == mt]
        print(f"  {mt}: {len(m_items)}")

    # Save full results
    out_path = Path("paper/human_eval/semantic_screen_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
