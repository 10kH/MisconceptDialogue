"""Diagnosis-emphasis robustness variant (camera-ready).

Re-runs the wrong-vs-gold diagnosis contrast under a STRONGER diagnosis-emphasis
tutor prompt (diagnosis-first ordering + explicit "address this misconception
directly" instruction). Tests whether the wrong-diagnosis NULL from
scripts/run_wrong_diagnosis_control.py is sensitive to prompt emphasis.

Two arms (both use the emphasis prompt from exp4v2.prompts_emph):
  - gold_emph : tutor sees the ORIGINAL (gold) misconception   -> results/exp4v2/condition_C2_emph
  - wrong_emph: tutor sees the WRONG (mis-diagnosed) label      -> results/exp4v2/condition_C_wrong_emph
In BOTH arms the judge always evaluates against the ORIGINAL misconception,
exactly as in the original control.

Everything else is held identical to run_wrong_diagnosis_control.py:
  student/judge = gpt-5-mini, temps (tutor 0.2 / student 0.7 / verify 0.3),
  5 tutor turns, C2 student profiles (loaded from the original exp4v2 dirs),
  original verification problem/answer, judge cache, checkpoint/resume.

Usage:
  python scripts/run_diagnosis_emphasis_variant.py --dry-run
  python scripts/run_diagnosis_emphasis_variant.py --pilot --model qwen-2.5-7b --arm both
  python scripts/run_diagnosis_emphasis_variant.py --model qwen-2.5-7b --arm both        # 1-model fallback
  python scripts/run_diagnosis_emphasis_variant.py --all --arm both                      # full 5-model run
  python scripts/run_diagnosis_emphasis_variant.py --all --arm wrong                     # single arm only
"""

import json
import os
import sys
import glob
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

SCENARIO_BANK_PATH = "data/exp3v2_scenario_bank.json"
WRONG_MAPPING_PATH = "data/wrong_diagnosis_mapping.json"
RESULTS_BASE = "results/exp4v2"
MODEL_NAMES = ["qwen-2.5-7b", "mistral-7b", "llama-3.1-8b", "phi-4", "qwen3.5-9b"]

# Arm -> (output dir, which misconception label the TUTOR sees)
ARM_CONFIG = {
    "gold_emph": {
        "output_dir": "results/exp4v2/condition_C2_emph",
        "tutor_label": "gold",   # original_misc
    },
    "wrong_emph": {
        "output_dir": "results/exp4v2/condition_C_wrong_emph",
        "tutor_label": "wrong",  # wrong_misc
    },
}


def load_scenario_bank():
    with open(SCENARIO_BANK_PATH) as f:
        return json.load(f)


def load_wrong_mapping():
    with open(WRONG_MAPPING_PATH) as f:
        return json.load(f)


def find_model_dir(model_name):
    for d in sorted(glob.glob(f"{RESULTS_BASE}/2026*")):
        if d.endswith(f"_{model_name}"):
            return d
    return None


def load_c2_profiles(model_dir):
    with open(f"{model_dir}/condition_C2.json") as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get("results", [])
    return {r["scenario_id"]: (r["profile_knowledge"], r["profile_cooperation"])
            for r in results}


def load_checkpoint(output_dir):
    cp = f"{output_dir}/checkpoint.jsonl"
    completed = {}
    if os.path.exists(cp):
        with open(cp) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed[f"{r['model']}___{r['scenario_id']}"] = r
    return completed


def append_checkpoint(output_dir, result):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/checkpoint.jsonl", "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_one(arm, tutor_model, student_model, judge_model, scenario_data,
            tutor_misc, original_misc, profile_kc, judge_cache):
    """One emphasis-variant dialogue.

    tutor prompt uses the EMPHASIS builder with tutor_misc (gold for gold_emph,
    wrong for wrong_emph). Judge always evaluates against original_misc.
    """
    from src.evaluation.simulated_student import (
        StudentProfile, build_student_system_prompt,
    )
    from src.experiments.exp4v2.prompts import (
        build_tutor_system_prompt, build_verification_prompt_v2,
    )
    from src.experiments.exp4v2.prompts_emph import build_tutor_prompt_emph
    from src.experiments.exp3v2.evaluator import evaluate_remediation
    from src.experiments.exp3v2.run import _check_answer_revealed, _adjust_tokens
    from src.experiments.exp3v2.scenario_bank import Scenario

    sc = Scenario(
        id=scenario_data["id"],
        problem=scenario_data["problem"],
        student_answer=scenario_data.get("student_answer", ""),
        correct_answer=scenario_data["correct_answer"],
        misconception_label=original_misc,   # ORIGINAL for judge/verification
        misconception_type=scenario_data["misconception_type"],
        difficulty=scenario_data.get("difficulty", "medium"),
        indicators=scenario_data.get("indicators", []),
        verification_problem=scenario_data["verification_problem"],
        verification_answer=scenario_data["verification_answer"],
        topic=scenario_data.get("topic", ""),
        source=scenario_data.get("source", ""),
        source_id=scenario_data.get("source_id", ""),
        variation_of=scenario_data.get("variation_of"),
    )

    knowledge, cooperation = profile_kc
    profile = StudentProfile(
        knowledge_level=knowledge,
        cooperation=cooperation,
        misconception_type=scenario_data["misconception_type"],
        misconception_description="",
        topic=scenario_data.get("topic", ""),
    )

    student_system = build_student_system_prompt(profile)
    tutor_system = build_tutor_system_prompt()

    conversation_history = []
    tutor_responses = []
    student_responses = []

    initial_msg = (
        f"I solved this problem: {sc.problem}\n\n"
        f"My answer is: {sc.student_answer}"
    )
    conversation_history.append({"role": "user", "content": initial_msg})
    student_responses.append(initial_msg)

    max_turns = 5
    for turn in range(max_turns):
        # --- Tutor turn (EMPHASIS prompt; tutor_misc = gold or wrong per arm) ---
        tutor_prompt = build_tutor_prompt_emph(
            problem=sc.problem,
            student_answer=sc.student_answer,
            conversation_history=conversation_history,
            misconception_label=tutor_misc,
            turn=turn,
        )
        try:
            tutor_resp = tutor_model.generate(
                prompt=tutor_prompt,
                system_prompt=tutor_system,
                temperature=0.2,
                max_tokens=_adjust_tokens(tutor_model, 2048),
            )
            tutor_text = tutor_resp.text.strip() or "[Tutor empty]"
        except Exception as e:
            logger.warning(f"Tutor error turn {turn}: {e}")
            tutor_text = "[Tutor error]"

        tutor_responses.append(tutor_text)
        conversation_history.append({"role": "assistant", "content": tutor_text})

        if turn < max_turns - 1:
            student_messages = [{"role": "system", "content": student_system}]
            student_messages.extend(conversation_history)
            try:
                student_resp = student_model.generate_with_history(
                    messages=student_messages,
                    temperature=0.7,
                    max_tokens=_adjust_tokens(student_model, 2048),
                )
                student_text = student_resp.text.strip() or "[Student empty]"
            except Exception as e:
                logger.warning(f"Student error turn {turn}: {e}")
                student_text = "[Student error]"

            student_responses.append(student_text)
            conversation_history.append({"role": "user", "content": student_text})

    # --- Verification ---
    verification_prompt_text = build_verification_prompt_v2(
        verification_problem=sc.verification_problem,
        misconception_type=sc.misconception_type,
    )
    verification_messages = [
        {"role": "system", "content": student_system},
        *conversation_history,
        {"role": "user", "content": verification_prompt_text},
    ]
    try:
        verif_resp = student_model.generate_with_history(
            messages=verification_messages,
            temperature=0.3,
            max_tokens=_adjust_tokens(student_model, 2048),
        )
        verification_answer = verif_resp.text.strip() or ""
    except Exception as e:
        logger.warning(f"Verification error: {e}")
        verification_answer = ""

    # --- Evaluate (ORIGINAL misconception for judge) ---
    evaluation = evaluate_remediation(
        student_response=verification_answer,
        correct_answer=sc.verification_answer,
        problem=sc.verification_problem,
        misconception=original_misc,
        misconception_indicators=sc.indicators,
        judge_model=judge_model,
        cache=judge_cache,
    )

    answer_revealed = _check_answer_revealed(tutor_responses, sc.correct_answer)

    return {
        "scenario_id": sc.id,
        "condition": arm,
        "misconception_type": sc.misconception_type,
        "difficulty": sc.difficulty,
        "profile_knowledge": knowledge,
        "profile_cooperation": cooperation,
        "original_misconception": original_misc,
        "tutor_misconception": tutor_misc,
        "num_turns": len(tutor_responses),
        "tutor_responses": tutor_responses,
        "student_responses": student_responses,
        "verification_answer": verification_answer,
        "remediation_success": evaluation["official_correct"],
        "tier1_correct": evaluation["tier1_correct"],
        "tier2_correct": evaluation["tier2_correct"],
        "judge_reasoning": evaluation.get("reasoning", ""),
        "judge_confidence": evaluation.get("confidence"),
        "misconception_resolved": evaluation.get("misconception_resolved"),
        "answer_revealed": answer_revealed,
    }


def run_arm(arm, models, scenarios, bank_by_id, mapping,
            student_model, judge_model, judge_cache, completed_by_arm):
    cfg = ARM_CONFIG[arm]
    output_dir = cfg["output_dir"]
    completed = completed_by_arm[arm]

    for model_name in models:
        logger.info(f"[{arm}] Loading tutor: {model_name}")
        from src.models.hf_model import HFModel
        tutor_model = HFModel(model_name=model_name)

        model_dir = find_model_dir(model_name)
        if not model_dir:
            logger.error(f"No result dir for {model_name} — cannot load C2 profiles")
            del tutor_model
            import torch; torch.cuda.empty_cache()
            continue
        profiles = load_c2_profiles(model_dir)

        for sc_data in scenarios:
            sid = sc_data["id"]
            key = f"{model_name}___{sid}"
            if key in completed:
                continue
            if sid not in mapping:
                continue

            original_misc = sc_data["misconception_label"]
            wrong_misc = mapping[sid]["wrong_misconception"]
            tutor_misc = original_misc if cfg["tutor_label"] == "gold" else wrong_misc
            profile_kc = profiles.get(sid, ("medium", "cooperative"))

            try:
                result = run_one(
                    arm, tutor_model, student_model, judge_model,
                    sc_data, tutor_misc, original_misc, profile_kc, judge_cache,
                )
                result["model"] = model_name
                result["arm"] = arm
                if cfg["tutor_label"] == "wrong":
                    result["wrong_scenario_id"] = mapping[sid].get("wrong_scenario_id", "")
                    result["mapping_method"] = "same_type_nonduplicate"
                append_checkpoint(output_dir, result)
                completed[key] = result

                t2 = result.get("tier2_correct")
                logger.info(f"[{arm}] {model_name}/{sid} t2={t2} "
                            f"({len(completed)} done in arm)")
            except Exception as e:
                logger.error(f"Error {arm}/{key}: {e}")
                err = {"model": model_name, "scenario_id": sid,
                       "arm": arm, "condition": arm, "error": str(e)}
                append_checkpoint(output_dir, err)
                completed[key] = err

            time.sleep(0.05)

        del tutor_model
        import torch; torch.cuda.empty_cache()

    valid = [v for v in completed.values() if not v.get("error")]
    errors = [v for v in completed.values() if v.get("error")]
    t2 = sum(1 for v in valid if v.get("tier2_correct"))
    logger.info(f"\n{'='*60}\n[{arm}] {len(valid)} valid, {len(errors)} errors")
    if valid:
        logger.info(f"[{arm}] RSR: {t2}/{len(valid)} = {t2/len(valid)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="5 scenarios only")
    parser.add_argument("--model", type=str, help="single model")
    parser.add_argument("--all", action="store_true", help="all 5 models")
    parser.add_argument("--arm", choices=["gold", "wrong", "both"], default="both",
                        help="which emphasis arm(s) to run")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--student-model", default="gpt-5-mini")
    args = parser.parse_args()

    bank = load_scenario_bank()
    mapping = load_wrong_mapping()
    bank_by_id = {s["id"]: s for s in bank}

    if args.dry_run:
        arms = ["gold_emph", "wrong_emph"] if args.arm == "both" \
            else [f"{args.arm}_emph"]
        print(f"Arms: {arms}")
        print(f"Scenario bank: {len(bank)}, wrong mapping: {len(mapping)}")
        from src.experiments.exp4v2.prompts_emph import build_tutor_prompt_emph
        sid = bank[0]["id"]
        demo = build_tutor_prompt_emph(
            problem=bank[0]["problem"],
            student_answer=bank[0].get("student_answer", ""),
            conversation_history=[],
            misconception_label=bank[0]["misconception_label"],
            turn=0,
        )
        print(f"\n--- turn-0 emphasis prompt for {sid} ---\n{demo}\n")
        return

    if args.model:
        models = [args.model]
    elif args.all:
        models = MODEL_NAMES
    else:
        print("Specify --dry-run, --pilot --model M, --model M, or --all")
        return

    if args.arm == "both":
        arms = ["gold_emph", "wrong_emph"]
    else:
        arms = [f"{args.arm}_emph"]

    scenarios = bank[:5] if args.pilot else bank
    if args.pilot:
        logger.info(f"PILOT MODE: {len(scenarios)} scenarios only")

    from src.models.openai_model import OpenAIModel
    student_model = OpenAIModel(model_name=args.student_model)
    judge_model = OpenAIModel(model_name=args.judge_model)
    judge_cache = {}

    completed_by_arm = {arm: load_checkpoint(ARM_CONFIG[arm]["output_dir"])
                        for arm in arms}
    total = len(arms) * len(models) * len(scenarios)
    done = sum(len(c) for c in completed_by_arm.values())
    logger.info(f"Target: {total} dialogues, already done: {done}")

    for arm in arms:
        run_arm(arm, models, scenarios, bank_by_id, mapping,
                student_model, judge_model, judge_cache, completed_by_arm)


if __name__ == "__main__":
    main()
