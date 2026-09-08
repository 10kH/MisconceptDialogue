"""
Within-construct judge sensitivity experiment:
Apply strict and lenient LLM-as-judge prompts to the same 2,250 Exp3v2 dialogues.
Standard (Tier 2) already exists. This adds strict + lenient for comparison.
"""
import json, os, glob, time, sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

OUTPUT_FILE = "results/exp3v2/judge_sensitivity_results.json"

STRICT_PROMPT = """You are a STRICT mathematics education evaluator. A student was tutored on a misconception and then given a verification problem.

**Misconception**: {misconception}
**Verification problem**: {verification_problem}
**Correct answer**: {correct_answer}
**Student's response**: {student_response}

Judge whether the student demonstrates CORRECT understanding. To count as correct, the student must:
1. Arrive at the numerically correct answer
2. Show clear, step-by-step reasoning
3. Demonstrate explicit understanding of WHY the original misconception was wrong

If ANY of these criteria is missing, judge as INCORRECT.

Respond in JSON: {{"correct": true/false, "reasoning": "one sentence"}}"""

LENIENT_PROMPT = """You are a LENIENT mathematics education evaluator. A student was tutored on a misconception and then given a verification problem.

**Misconception**: {misconception}
**Verification problem**: {verification_problem}
**Correct answer**: {correct_answer}
**Student's response**: {student_response}

Judge whether the student shows ANY sign of improved understanding compared to their original misconception. Even partial understanding, approximate answers, or correct reasoning with minor computational errors should count as correct. The key question is: has the student moved AWAY from their misconception toward a better understanding?

Respond in JSON: {{"correct": true/false, "reasoning": "one sentence"}}"""


def evaluate(prompt_template, student_response, misconception, verif_problem, correct_answer):
    prompt = prompt_template.format(
        misconception=misconception[:300],
        verification_problem=verif_problem[:300],
        correct_answer=correct_answer[:200],
        student_response=student_response[:1000]
    )
    try:
        r = client.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=150,
            response_format={"type": "json_object"}
        )
        result = json.loads(r.choices[0].message.content)
        return result.get("correct", False)
    except Exception as e:
        return None


def main():
    # [FIX] scenario bank에서 verification 필드를 직접 로드.
    # 이전: scenario_source(문자열)를 dict로 가정하여 빈 gold로 평가했음.
    BANK_PATH = "data/exp3v2_scenario_bank.json"
    with open(BANK_PATH) as f:
        scenario_bank = {s["id"]: s for s in json.load(f)}
    print(f"Loaded scenario bank: {len(scenario_bank)} scenarios")

    # Load all dialogues
    all_dialogues = []
    for d in sorted(glob.glob('results/exp3v2/2026*')):
        if not os.path.isdir(d):
            continue
        model = d.split('_')[-1]
        for cf in sorted(glob.glob(f'{d}/cell_*.json')):
            with open(cf) as f:
                cell = json.load(f)
            for idx, r in enumerate(cell['results']):
                r['_model'] = model
                r['_cell'] = os.path.basename(cf)
                r['_idx'] = idx
                all_dialogues.append(r)

    print(f"Loaded {len(all_dialogues)} dialogues")

    # Load checkpoint
    results = []
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        completed = {r['key'] for r in results}
        print(f"Checkpoint: {len(results)} already done")

    for i, dial in enumerate(all_dialogues):
        key = f"{dial['_model']}_{dial['_cell']}_{dial['_idx']}"
        if key in completed:
            continue

        # [FIX] scenario bank에서 verification 필드를 직접 조회.
        # 이전: scenario_source(문자열 "gpt5.4_generated" 등)를 dict로 가정 → 빈 gold.
        sid = dial.get('scenario_id', '')
        sc = scenario_bank.get(sid, {})
        misconception = sc.get('misconception_label', str(sid))
        verif_problem = sc.get('verification_problem', '')
        correct_answer = sc.get('verification_answer', '')

        student_response = dial.get('verification_answer', '')
        if not student_response:
            continue

        # Evaluate with both prompts
        strict = evaluate(STRICT_PROMPT, student_response, misconception, verif_problem, correct_answer)
        lenient = evaluate(LENIENT_PROMPT, student_response, misconception, verif_problem, correct_answer)

        entry = {
            'key': key,
            'model': dial['_model'],
            'scenario_id': dial.get('scenario_id', ''),
            'strategy': dial.get('strategy', ''),
            'misconception_type': dial.get('misconception_type', ''),
            'tier1': dial.get('tier1_correct', False),
            'tier2_standard': dial.get('tier2_correct', False),
            'tier2_strict': strict,
            'tier2_lenient': lenient,
        }
        results.append(entry)
        completed.add(key)

        if (len(results)) % 50 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, ensure_ascii=False)
            # Print progress
            done_strict = [r for r in results if r['tier2_strict'] is not None]
            done_lenient = [r for r in results if r['tier2_lenient'] is not None]
            strict_rsr = sum(1 for r in done_strict if r['tier2_strict']) / len(done_strict) if done_strict else 0
            lenient_rsr = sum(1 for r in done_lenient if r['tier2_lenient']) / len(done_lenient) if done_lenient else 0
            std_rsr = sum(1 for r in results if r['tier2_standard']) / len(results) if results else 0
            print(f"  [{len(results)}/{len(all_dialogues)}] Strict={strict_rsr:.1%} Standard={std_rsr:.1%} Lenient={lenient_rsr:.1%}")

        time.sleep(0.05)  # Rate limit

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    valid = [r for r in results if r['tier2_strict'] is not None and r['tier2_lenient'] is not None]
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (n={len(valid)})")
    print(f"{'='*60}")

    tier1_rsr = sum(1 for r in valid if r['tier1']) / len(valid)
    strict_rsr = sum(1 for r in valid if r['tier2_strict']) / len(valid)
    std_rsr = sum(1 for r in valid if r['tier2_standard']) / len(valid)
    lenient_rsr = sum(1 for r in valid if r['tier2_lenient']) / len(valid)

    print(f"  Tier 1 (regex):      {tier1_rsr:.1%}")
    print(f"  Tier 2 STRICT:       {strict_rsr:.1%}")
    print(f"  Tier 2 STANDARD:     {std_rsr:.1%}")
    print(f"  Tier 2 LENIENT:      {lenient_rsr:.1%}")
    print(f"  Strict-Lenient gap:  {(lenient_rsr - strict_rsr)*100:.1f}pp")
    print(f"  Standard-Strict gap: {(std_rsr - strict_rsr)*100:.1f}pp")
    print(f"\n  → Within-construct (same LLM judge, different prompt) gap: {(lenient_rsr - strict_rsr)*100:.1f}pp")
    print(f"  → Cross-construct (Tier1 vs Tier2 standard) gap: {(std_rsr - tier1_rsr)*100:.1f}pp")


if __name__ == '__main__':
    main()
