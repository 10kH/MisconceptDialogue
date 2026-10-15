"""
Screen 440 'borderline dirty' dialogues using GPT-5.4 for human evaluation suitability.
Asks: "Is this dialogue evaluable by human annotators despite its quality issues?"
Outputs a scored list for selecting 100 'meaningfully dirty' dialogues.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)
import json, os, sys, time, glob
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
client = OpenAI()

CHECKPOINT_FILE = "results/exp3v2/gpt54_screen/dirty_screen_checkpoint.json"
OUTPUT_FILE = "results/exp3v2/gpt54_screen/dirty_screen_results.json"

SCREEN_PROMPT = """You are evaluating whether a tutoring dialogue is suitable for HUMAN ANNOTATION in a research study.

The dialogue has known quality issues (it scored 2.0-2.99 out of 5.0 on automated quality screening). Your job is to determine whether, DESPITE these issues, a human annotator can still meaningfully evaluate:
1. Whether the student demonstrates correct understanding after tutoring
2. The quality of the tutor's pedagogical approach

## Dialogue

**Misconception**: {misconception}
**Strategy**: {strategy}
**Model**: {model}

### Tutoring Dialogue:
{dialogue_text}

### Verification Response:
{verification}

## Evaluation

Rate this dialogue on a 1-3 scale:
- **1 = UNUSABLE**: Dialogue is incoherent, roles are completely confused, or the conversation is so broken that no meaningful evaluation is possible.
- **2 = BORDERLINE**: Dialogue has clear issues (repetition, partial role confusion, odd responses) but a human annotator can still judge whether the student understood the concept.
- **3 = USABLE**: Dialogue has minor issues but is clearly evaluable.

Respond in JSON format:
{{"score": <1-3>, "reason": "<one sentence explaining your rating>"}}
"""


def load_safe_pool():
    """Load the 440 safe dialogues (score 2.0-2.99, no answer_leaked, RI>=2)."""
    with open('results/exp3v2/gpt54_screen/checkpoint.json') as f:
        gpt54 = json.load(f)

    safe = []
    for item in gpt54:
        scores = item['scores']
        vals = [v for v in scores.values() if isinstance(v, (int, float))]
        mean_score = sum(vals) / len(vals) if vals else 0
        item['mean_quality'] = mean_score
        if 2.0 <= mean_score < 3.0:
            if 'answer_leaked' not in scores.get('issues', []):
                if scores.get('role_integrity', 0) >= 2:
                    safe.append(item)
    return safe


def load_dialogue_content(item):
    """Load actual dialogue text from the cell file."""
    cell_file = item['cell_file']
    idx = item['idx_in_cell']
    model = item['model']

    # Find the right directory
    for d in sorted(os.listdir('results/exp3v2')):
        if d.startswith('2026') and model.replace('-', '') in d.replace('-', ''):
            cf = f'results/exp3v2/{d}/{cell_file}'
            if os.path.exists(cf):
                with open(cf) as f:
                    cell_data = json.load(f)
                results = cell_data.get('results', [])
                if idx < len(results):
                    return results[idx]

    # Try all directories
    for d in sorted(os.listdir('results/exp3v2')):
        full = f'results/exp3v2/{d}'
        if os.path.isdir(full) and d.startswith('2026'):
            cf = os.path.join(full, cell_file)
            if os.path.exists(cf):
                with open(cf) as f:
                    cell_data = json.load(f)
                results = cell_data.get('results', [])
                if idx < len(results):
                    r = results[idx]
                    if r.get('scenario_id') == item.get('scenario_id'):
                        return r
    return None


def format_dialogue(dialogue_data):
    """Format dialogue for the prompt."""
    tutor = dialogue_data.get('tutor_responses', [])
    student = dialogue_data.get('student_responses', [])

    text = ""
    for i in range(max(len(tutor), len(student))):
        if i < len(tutor):
            text += f"**Tutor**: {tutor[i]}\n\n"
        if i < len(student):
            text += f"**Student**: {student[i]}\n\n"
    return text


def screen_dialogue(item, dialogue_data):
    """Screen a single dialogue with GPT-5.4."""
    scenario = dialogue_data.get('scenario_source', {})
    if isinstance(scenario, str):
        misconception = scenario[:200]
    elif isinstance(scenario, dict):
        misconception = scenario.get('misconception_text', scenario.get('misconception_id', 'unknown'))
    else:
        misconception = str(dialogue_data.get('scenario_id', 'unknown'))

    dialogue_text = format_dialogue(dialogue_data)
    verification = dialogue_data.get('verification_answer', 'N/A')

    prompt = SCREEN_PROMPT.format(
        misconception=misconception,
        strategy=item['strategy'],
        model=item['model'],
        dialogue_text=dialogue_text[:3000],  # Truncate if very long
        verification=str(verification)[:500]
    )

    try:
        response = client.chat.completions.create(
            model="gpt-5.4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_completion_tokens=200,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"score": 0, "reason": f"API error: {str(e)[:50]}"}


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    safe_pool = load_safe_pool()
    print(f"Safe pool: {len(safe_pool)} dialogues")

    # Load checkpoint if exists
    completed = {}
    if os.path.exists(CHECKPOINT_FILE.replace('checkpoint', 'dirty_screen_checkpoint')):
        with open(CHECKPOINT_FILE.replace('checkpoint', 'dirty_screen_checkpoint')) as f:
            completed = {r['key']: r for r in json.load(f)}

    results = list(completed.values())
    print(f"Already screened: {len(results)}")

    for i, item in enumerate(safe_pool):
        key = f"{item['model']}_{item['cell_file']}_{item['idx_in_cell']}"
        if key in completed:
            continue

        # Load dialogue content
        dialogue_data = load_dialogue_content(item)
        if not dialogue_data:
            print(f"  [{i+1}/{len(safe_pool)}] SKIP: dialogue not found for {key}")
            continue

        # Screen with GPT-5.4
        result = screen_dialogue(item, dialogue_data)

        entry = {
            'key': key,
            'model': item['model'],
            'strategy': item['strategy'],
            'misconception_type': item['misconception_type'],
            'scenario_id': item['scenario_id'],
            'cell_file': item['cell_file'],
            'idx_in_cell': item['idx_in_cell'],
            'tier1_correct': item['tier1_correct'],
            'tier2_correct': item['tier2_correct'],
            'mean_quality': item['mean_quality'],
            'screen_score': result.get('score', 0),
            'screen_reason': result.get('reason', ''),
        }
        results.append(entry)
        completed[key] = entry

        score = result.get('score', 0)
        print(f"  [{i+1}/{len(safe_pool)}] {item['model']:15s} score={score} {result.get('reason', '')[:60]}")

        # Checkpoint every 20
        if len(results) % 20 == 0:
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  --- Checkpoint: {len(results)} screened ---")

        time.sleep(0.1)  # Rate limiting

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    scores = [r['screen_score'] for r in results]
    from collections import Counter
    score_dist = Counter(scores)
    print(f"\n=== Final Results ===")
    print(f"Total screened: {len(results)}")
    print(f"Score distribution: {dict(sorted(score_dist.items()))}")

    usable = [r for r in results if r['screen_score'] >= 2]
    print(f"Usable (score >= 2): {len(usable)}")
    print(f"  Tier2 RSR: {sum(1 for r in usable if r['tier2_correct'])/len(usable):.1%}")


if __name__ == '__main__':
    main()
