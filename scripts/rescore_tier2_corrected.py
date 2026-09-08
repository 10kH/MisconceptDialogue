"""
Tier2 재채점 스크립트: verification_answer/verification_problem을 올바른 gold로 사용하여
GPT-5-mini judge에게 Exp3v2 2,250건 + Exp4v2 전 condition을 재채점한다.

배경: 이전 run.py에서 scenario.correct_answer(원문 정답)를 judge 프롬프트에 넘겨
잘못된 문맥으로 채점했음. 이 스크립트는 올바른 verification 필드로 재채점한다.

사용법:
  python scripts/rescore_tier2_corrected.py             # Exp3v2만
  python scripts/rescore_tier2_corrected.py --exp4       # Exp4v2만
  python scripts/rescore_tier2_corrected.py --all        # 둘 다
"""

import json
import glob
import os
import sys
import time
import argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# evaluator.py와 동일한 judge 프롬프트 (변경 없음)
JUDGE_SYSTEM_PROMPT = """You are an expert mathematics education evaluator. Your job is to assess whether a student has demonstrated CORRECT UNDERSTANDING of a mathematical concept after a remediation session.

IMPORTANT DISTINCTION:
- "Correct understanding" means the student grasps the underlying concept/procedure/fact.
- "Correct answer" means the student arrived at the right numerical/symbolic answer.
- Understanding matters MORE than the final answer. A student who shows correct reasoning but makes an arithmetic slip has demonstrated understanding. A student who states the right answer but shows wrong reasoning has NOT.

EVALUATION RUBRIC:
1. correct (bool): Does the student demonstrate correct understanding of the concept?
   - true: Student shows they understand the concept, even with minor arithmetic errors
   - false: Student does NOT demonstrate understanding, even if they happen to state the right answer
2. reasoning (str): Brief explanation of your judgment (1-2 sentences)
3. confidence (float): Your confidence in this assessment (0.0-1.0)
4. misconception_resolved (bool): Is the original misconception no longer present in the student's reasoning?

Respond ONLY with a JSON object. No markdown, no explanation outside the JSON."""

JUDGE_MODEL = "gpt-5-mini"  # 원래 실험과 동일한 judge 모델
CHECKPOINT_INTERVAL = 10  # gpt-5-mini 호출이 느려 자주 저장


def build_judge_prompt(student_response, correct_answer, problem, misconception):
    """evaluator.py의 _build_judge_prompt와 동일."""
    return f"""Evaluate this student's response after a remediation session:

ORIGINAL PROBLEM: {problem}
CORRECT ANSWER: {correct_answer}
MISCONCEPTION BEING ADDRESSED: {misconception}
STUDENT'S RESPONSE: {student_response}

Assess whether the student demonstrates correct understanding. Respond with JSON:
{{"correct": bool, "reasoning": "...", "confidence": float, "misconception_resolved": bool}}"""


def call_judge(student_response, correct_answer, problem, misconception):
    """GPT-5-mini judge 호출. 최대 3회 재시도."""
    prompt = build_judge_prompt(student_response, correct_answer, problem, misconception)
    backoff = [1, 2, 4]

    for attempt in range(3):
        try:
            # gpt-5-mini는 temperature=0.0 미지원 → 생략 (기본값 1.0 사용)
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=8192,  # gpt-5-mini는 reasoning token 소모 → 충분히 크게
                response_format={"type": "json_object"},
            )
            result = json.loads(r.choices[0].message.content)
            return {
                "correct": bool(result.get("correct", False)),
                "reasoning": str(result.get("reasoning", "")),
                "confidence": float(result.get("confidence", 0.5)),
                "misconception_resolved": bool(result.get("misconception_resolved", False)),
                "judge_error": None,
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(backoff[attempt])
            else:
                return {
                    "correct": False,
                    "reasoning": f"Judge failed: {e}",
                    "confidence": 0.0,
                    "misconception_resolved": False,
                    "judge_error": str(e),
                }


def load_scenario_bank():
    with open("data/exp3v2_scenario_bank.json") as f:
        return {s["id"]: s for s in json.load(f)}


def rescore_exp3v2(bank):
    """Exp3v2 2,250건 재채점."""
    print("=" * 60)
    print("EXP3V2 TIER2 RE-SCORING (GPT-5-mini)")
    print("=" * 60)

    out_dir = "results/exp3v2/tier2_corrected"
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")

    # 체크포인트 로드
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            completed = json.load(f)
        print(f"Checkpoint loaded: {len(completed)} already done")

    # 전체 대화 수집
    all_items = []
    for d in sorted(glob.glob("results/exp3v2/2026*")):
        if not os.path.isdir(d):
            continue
        model = d.split("_")[-1]
        for cf in sorted(glob.glob(f"{d}/cell_*.json")):
            with open(cf) as f:
                cell = json.load(f)
            for idx, r in enumerate(cell["results"]):
                key = f"{model}_{os.path.basename(cf)}_{idx}"
                all_items.append({
                    "key": key,
                    "model": model,
                    "scenario_id": r["scenario_id"],
                    "strategy": r.get("strategy", ""),
                    "misconception_type": r.get("misconception_type", ""),
                    "student_response": r.get("verification_answer", ""),
                    "original_tier2": r.get("tier2_correct", r.get("remediation_success")),
                })

    print(f"Total dialogues: {len(all_items)}")
    remaining = [item for item in all_items if item["key"] not in completed]
    print(f"Remaining: {len(remaining)}")

    for i, item in enumerate(remaining):
        sid = item["scenario_id"]
        sc = bank.get(sid, {})

        # [핵심] verification 필드 사용 (수정된 gold)
        correct_answer = str(sc.get("verification_answer", ""))
        problem = str(sc.get("verification_problem", ""))
        misconception = str(sc.get("misconception_label", ""))

        student = item["student_response"]
        if not student:
            completed[item["key"]] = {
                "correct": False,
                "reasoning": "Empty student response",
                "confidence": 1.0,
                "misconception_resolved": False,
                "judge_error": None,
                "scenario_id": sid,
                "model": item["model"],
                "strategy": item["strategy"],
                "misconception_type": item["misconception_type"],
                "original_tier2": item["original_tier2"],
            }
            continue

        result = call_judge(student, correct_answer, problem, misconception)
        result["scenario_id"] = sid
        result["model"] = item["model"]
        result["strategy"] = item["strategy"]
        result["misconception_type"] = item["misconception_type"]
        result["original_tier2"] = item["original_tier2"]
        completed[item["key"]] = result

        # 체크포인트 저장
        if (len(completed)) % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, ensure_ascii=False)
            done = len(completed)
            total = len(all_items)
            corr = sum(1 for v in completed.values() if v["correct"])
            orig = sum(1 for v in completed.values() if v.get("original_tier2"))
            print(f"  [{done}/{total}] Corrected={corr/done*100:.1f}% Original={orig/done*100:.1f}%")

        time.sleep(0.05)  # rate limit

    # 최종 저장
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, ensure_ascii=False)

    # 결과 요약
    results_list = list(completed.values())
    n = len(results_list)
    errors = sum(1 for r in results_list if r.get("judge_error"))
    corr = sum(1 for r in results_list if r["correct"])
    orig = sum(1 for r in results_list if r.get("original_tier2"))

    print(f"\n--- FINAL SUMMARY ---")
    print(f"  Total: {n}")
    print(f"  Judge errors: {errors}")
    print(f"  Original Tier2 (wrong gold): {orig}/{n} = {orig/n*100:.1f}%")
    print(f"  Corrected Tier2 (right gold): {corr}/{n} = {corr/n*100:.1f}%")
    print(f"  Delta: {(corr-orig)/n*100:+.1f}pp")

    # Per-model
    print(f"\n--- PER MODEL ---")
    models = sorted(set(r["model"] for r in results_list))
    for m in models:
        mr = [r for r in results_list if r["model"] == m]
        mn = len(mr)
        mo = sum(1 for r in mr if r.get("original_tier2"))
        mc = sum(1 for r in mr if r["correct"])
        print(f"  {m:<16} Orig={mo/mn*100:.1f}% Corr={mc/mn*100:.1f}% Δ={((mc-mo)/mn*100):+.1f}pp (n={mn})")

    # Per-type
    print(f"\n--- PER TYPE ---")
    types = sorted(set(r["misconception_type"] for r in results_list if r.get("misconception_type")))
    for t in types:
        tr = [r for r in results_list if r.get("misconception_type") == t]
        tn = len(tr)
        to_ = sum(1 for r in tr if r.get("original_tier2"))
        tc = sum(1 for r in tr if r["correct"])
        print(f"  {t:<12} Orig={to_/tn*100:.1f}% Corr={tc/tn*100:.1f}% Δ={((tc-to_)/tn*100):+.1f}pp (n={tn})")

    # 최종 결과 파일
    final_path = os.path.join(out_dir, "tier2_recomputed.json")
    with open(final_path, "w") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {final_path}")

    return results_list


def rescore_exp4v2(bank):
    """Exp4v2 전 condition 재채점."""
    print("\n" + "=" * 60)
    print("EXP4V2 TIER2 RE-SCORING (GPT-5-mini)")
    print("=" * 60)

    out_dir = "results/exp4v2/tier2_corrected"
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")

    # 체크포인트 로드
    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            completed = json.load(f)
        print(f"Checkpoint loaded: {len(completed)} already done")

    # 전체 결과 수집
    all_items = []
    for d in sorted(glob.glob("results/exp4v2/2026*")):
        if not os.path.isdir(d):
            continue
        model = d.split("_")[-1]
        for cf in sorted(glob.glob(f"{d}/condition_*.json")):
            condition = os.path.basename(cf).replace(".json", "").replace("condition_", "")
            with open(cf) as f:
                data = json.load(f)
            results = data if isinstance(data, list) else data.get("results", [])
            for idx, r in enumerate(results):
                key = f"{model}_{condition}_{idx}"
                all_items.append({
                    "key": key,
                    "model": model,
                    "condition": condition,
                    "scenario_id": r.get("scenario_id", ""),
                    "student_response": r.get("verification_answer", ""),
                    "original_tier2": r.get("tier2_correct", r.get("remediation_success")),
                })

    print(f"Total results: {len(all_items)}")
    remaining = [item for item in all_items if item["key"] not in completed]
    print(f"Remaining: {len(remaining)}")

    for i, item in enumerate(remaining):
        sid = item["scenario_id"]
        sc = bank.get(sid, {})

        correct_answer = str(sc.get("verification_answer", ""))
        problem = str(sc.get("verification_problem", ""))
        misconception = str(sc.get("misconception_label", ""))

        student = item["student_response"]
        if not student:
            completed[item["key"]] = {
                "correct": False,
                "reasoning": "Empty student response",
                "confidence": 1.0,
                "misconception_resolved": False,
                "judge_error": None,
                "scenario_id": sid,
                "model": item["model"],
                "condition": item["condition"],
                "original_tier2": item["original_tier2"],
            }
            continue

        result = call_judge(student, correct_answer, problem, misconception)
        result["scenario_id"] = sid
        result["model"] = item["model"]
        result["condition"] = item["condition"]
        result["original_tier2"] = item["original_tier2"]
        completed[item["key"]] = result

        if (len(completed)) % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(completed, f, ensure_ascii=False)
            done = len(completed)
            total = len(all_items)
            corr = sum(1 for v in completed.values() if v["correct"])
            orig = sum(1 for v in completed.values() if v.get("original_tier2"))
            print(f"  [{done}/{total}] Corrected={corr/done*100:.1f}% Original={orig/done*100:.1f}%")

        time.sleep(0.05)

    # 최종 저장
    with open(checkpoint_path, "w") as f:
        json.dump(completed, f, ensure_ascii=False)

    results_list = list(completed.values())
    n = len(results_list)
    if n == 0:
        print("No Exp4v2 results.")
        return results_list

    errors = sum(1 for r in results_list if r.get("judge_error"))
    corr = sum(1 for r in results_list if r["correct"])
    orig = sum(1 for r in results_list if r.get("original_tier2"))

    print(f"\n--- FINAL SUMMARY ---")
    print(f"  Total: {n}, Judge errors: {errors}")
    print(f"  Original Tier2: {orig}/{n} = {orig/n*100:.1f}%")
    print(f"  Corrected Tier2: {corr}/{n} = {corr/n*100:.1f}%")
    print(f"  Delta: {(corr-orig)/n*100:+.1f}pp")

    # Per-condition
    print(f"\n--- PER CONDITION ---")
    conditions = sorted(set(r["condition"] for r in results_list if r.get("condition")))
    for c in conditions:
        cr = [r for r in results_list if r.get("condition") == c]
        cn = len(cr)
        co = sum(1 for r in cr if r.get("original_tier2"))
        cc = sum(1 for r in cr if r["correct"])
        print(f"  {c:<10} Orig={co/cn*100:.1f}% Corr={cc/cn*100:.1f}% Δ={((cc-co)/cn*100):+.1f}pp (n={cn})")

    final_path = os.path.join(out_dir, "tier2_recomputed.json")
    with open(final_path, "w") as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {final_path}")

    return results_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp4", action="store_true", help="Only Exp4v2")
    parser.add_argument("--all", action="store_true", help="Both Exp3v2 + Exp4v2")
    args = parser.parse_args()

    bank = load_scenario_bank()
    print(f"Loaded {len(bank)} scenarios")

    if args.exp4:
        rescore_exp4v2(bank)
    elif args.all:
        rescore_exp3v2(bank)
        rescore_exp4v2(bank)
    else:
        rescore_exp3v2(bank)
