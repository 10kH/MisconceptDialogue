"""
Wrong-diagnosis control experiment (C_wrong condition).
Tutor에게는 틀린 diagnosis를, judge에게는 원래 diagnosis를 제공하여
pipeline diagnosis 정보가 실제로 활용되는지 검증.

핵심 설계:
- tutor prompt: wrong misconception (의도적 오진)
- judge evaluation: original misconception (정확한 평가 유지)
- verification problem/answer: original (변경 없음)
- student profile: C2와 동일 (기존 결과에서 로드)

사용법:
  python scripts/run_wrong_diagnosis_control.py --dry-run
  python scripts/run_wrong_diagnosis_control.py --pilot --model qwen-2.5-7b
  python scripts/run_wrong_diagnosis_control.py --model qwen-2.5-7b
  python scripts/run_wrong_diagnosis_control.py --all
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
OUTPUT_DIR = "results/exp4v2/condition_C_wrong"
MODEL_NAMES = ["qwen-2.5-7b", "mistral-7b", "llama-3.1-8b", "phi-4", "qwen3.5-9b"]


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


def load_checkpoint():
    cp = f"{OUTPUT_DIR}/checkpoint.jsonl"
    completed = {}
    if os.path.exists(cp):
        with open(cp) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed[f"{r['model']}___{r['scenario_id']}"] = r
    return completed


def append_checkpoint(result):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(f"{OUTPUT_DIR}/checkpoint.jsonl", "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def run_one(tutor_model, student_model, judge_model, scenario_data,
            wrong_misc, original_misc, profile_kc, judge_cache):
    """
    C_wrong 대화 1건 실행.
    tutor prompt에만 wrong_misc를 넣고, evaluate_remediation에는 original_misc를 넣는다.
    run_pipeline_dialogue()를 재사용하지 않고 대화 루프를 직접 구현하여
    tutor/judge misconception을 분리한다.
    """
    from src.evaluation.simulated_student import (
        StudentProfile, build_student_system_prompt,
    )
    from src.experiments.exp4v2.prompts import (
        build_tutor_prompt, build_tutor_system_prompt,
        build_verification_prompt_v2,
    )
    from src.experiments.exp3v2.evaluator import evaluate_remediation
    from src.experiments.exp3v2.run import _check_answer_revealed, _adjust_tokens
    from src.experiments.exp3v2.scenario_bank import Scenario

    # Scenario 객체 (원본 — verification/evaluation용)
    sc = Scenario(
        id=scenario_data["id"],
        problem=scenario_data["problem"],
        student_answer=scenario_data.get("student_answer", ""),
        correct_answer=scenario_data["correct_answer"],
        misconception_label=original_misc,  # ORIGINAL for judge
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

    # System prompts
    student_system = build_student_system_prompt(profile)
    tutor_system = build_tutor_system_prompt()

    conversation_history = []
    tutor_responses = []
    student_responses = []

    # Initial student message
    initial_msg = (
        f"I solved this problem: {sc.problem}\n\n"
        f"My answer is: {sc.student_answer}"
    )
    conversation_history.append({"role": "user", "content": initial_msg})
    student_responses.append(initial_msg)

    # 기존 Exp4와 동일한 turn 구조: 5 tutor turns, 4 student turns
    max_turns = 5
    for turn in range(max_turns):
        # --- Tutor turn (WRONG misconception in prompt) ---
        tutor_prompt = build_tutor_prompt(
            condition="C2",
            problem=sc.problem,
            student_answer=sc.student_answer,
            conversation_history=conversation_history,
            misconception_label=wrong_misc,  # ← WRONG for tutor
            misconception_type=sc.misconception_type,
            model_diagnosis=None,
            model_diagnosis_type=None,
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

        # Student turn (except after last tutor turn)
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
        correct_answer=sc.verification_answer,       # original gold
        problem=sc.verification_problem,              # original problem
        misconception=original_misc,                  # ← ORIGINAL for judge
        misconception_indicators=sc.indicators,
        judge_model=judge_model,
        cache=judge_cache,
    )

    answer_revealed = _check_answer_revealed(tutor_responses, sc.correct_answer)

    return {
        "scenario_id": sc.id,
        "condition": "C_wrong",
        "misconception_type": sc.misconception_type,
        "difficulty": sc.difficulty,
        "profile_knowledge": knowledge,
        "profile_cooperation": cooperation,
        "original_misconception": original_misc,
        "wrong_misconception": wrong_misc,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="5 scenarios only")
    parser.add_argument("--model", type=str, help="단일 모델")
    parser.add_argument("--all", action="store_true", help="전체 5 모델")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--student-model", default="gpt-5-mini")
    args = parser.parse_args()

    bank = load_scenario_bank()
    mapping = load_wrong_mapping()

    if args.dry_run:
        dupes = sum(1 for v in mapping.values()
                    if v["original_misconception"] == v["wrong_misconception"])
        print(f"Mapping: {len(mapping)} entries, duplicates: {dupes}")
        for sid in list(mapping.keys())[:5]:
            m = mapping[sid]
            print(f"  {sid}: {m['original_misconception'][:50]}...")
            print(f"       → {m['wrong_misconception'][:50]}...")
        return

    if args.model:
        models = [args.model]
    elif args.all:
        models = MODEL_NAMES
    else:
        print("Specify --dry-run, --pilot --model MODEL, --model MODEL, or --all")
        return

    scenarios = bank
    if args.pilot:
        scenarios = bank[:5]
        logger.info(f"PILOT MODE: {len(scenarios)} scenarios only")

    # Load models
    from src.models.openai_model import OpenAIModel
    from src.models.hf_model import HFModel

    student_model = OpenAIModel(model_name=args.student_model)
    judge_model = OpenAIModel(model_name=args.judge_model)
    judge_cache = {}

    completed = load_checkpoint()
    total = len(models) * len(scenarios)
    logger.info(f"Target: {total} dialogues, Already done: {len(completed)}")

    for model_name in models:
        logger.info(f"Loading tutor: {model_name}")
        tutor_model = HFModel(model_name=model_name)

        model_dir = find_model_dir(model_name)
        if not model_dir:
            logger.error(f"No result dir for {model_name}")
            continue
        profiles = load_c2_profiles(model_dir)

        for sc_data in scenarios:
            sid = sc_data["id"]
            key = f"{model_name}___{sid}"
            if key in completed:
                continue
            if sid not in mapping:
                continue

            wrong_misc = mapping[sid]["wrong_misconception"]
            original_misc = sc_data["misconception_label"]
            profile_kc = profiles.get(sid, ("medium", "cooperative"))

            try:
                result = run_one(
                    tutor_model, student_model, judge_model,
                    sc_data, wrong_misc, original_misc, profile_kc, judge_cache,
                )
                result["model"] = model_name
                result["wrong_scenario_id"] = mapping[sid].get("wrong_scenario_id", "")
                result["mapping_method"] = "same_type_nonduplicate"
                append_checkpoint(result)
                completed[key] = result

                t2 = result.get("tier2_correct")
                logger.info(f"[{len(completed)}/{total}] {model_name}/{sid} t2={t2}")
            except Exception as e:
                logger.error(f"Error {key}: {e}")
                err = {"model": model_name, "scenario_id": sid,
                       "error": str(e), "condition": "C_wrong"}
                append_checkpoint(err)
                completed[key] = err

            time.sleep(0.05)

        del tutor_model
        import torch; torch.cuda.empty_cache()

    # Summary
    valid = [v for v in completed.values() if not v.get("error")]
    errors = [v for v in completed.values() if v.get("error")]
    t2_correct = sum(1 for v in valid if v.get("tier2_correct"))
    logger.info(f"\n{'='*60}")
    logger.info(f"C_wrong: {len(valid)} valid, {len(errors)} errors")
    if valid:
        logger.info(f"RSR: {t2_correct}/{len(valid)} = {t2_correct/len(valid)*100:.1f}%")
    logger.info(f"Compare: C2=76.6%, B1=78.5%")


if __name__ == "__main__":
    main()
