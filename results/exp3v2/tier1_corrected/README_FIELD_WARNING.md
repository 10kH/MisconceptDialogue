# Field warning: `tier1_recomputed.json` carries a stale embedded `tier2_correct`

**Do not join against this file's own `tier2_correct` field.** It is a leftover
pre-fix (pre verification-gold-mismatch-correction) copy of the LLM-judge verdict,
computed under the old, incorrect pipeline. It was not deleted from this artifact, but
it must never be used.

This file was **not mutated** to fix this — the embedded field is left in place
(deleting or silently regenerating it risked masking exactly the kind of stale-data bug
this note exists to prevent). Read this file before writing any code that reads
`tier1_recomputed.json`.

## What to use instead

| Need | Use |
|---|---|
| Strict-match (Tier 1) verdict | **`corrected_tier1`** field in this file (`tier1_recomputed.json`) |
| LLM-judge (Tier 2) verdict | **`correct`** field in the standalone `../tier2_corrected/tier2_recomputed.json`, joined on `(scenario_id, model, strategy)` — a unique key in both files (2,250 records each, 0 duplicates, 0 missing joins) |
| ~~LLM-judge verdict via this file's `tier2_correct`~~ | **Never.** Stale — see cross-tab below. |

```python
import json

with open("results/exp3v2/tier1_corrected/tier1_recomputed.json") as f:
    tier1 = json.load(f)
with open("results/exp3v2/tier2_corrected/tier2_recomputed.json") as f:
    tier2 = json.load(f)

tier2_by_key = {(r["scenario_id"], r["model"], r["strategy"]): r["correct"] for r in tier2}

for r in tier1:
    key = (r["scenario_id"], r["model"], r["strategy"])
    strict_verdict = r["corrected_tier1"]        # correct
    judge_verdict = tier2_by_key[key]             # correct — from the standalone file
    # judge_verdict_WRONG = r["tier2_correct"]    # DO NOT USE — stale
```

## Proof: wrong-vs-right cross-tabs

Both tables below use the same `corrected_tier1` column (the strict-match verdict is
identical either way — this field is not affected by the bug). They differ only in
which judge-verdict field is joined against. n = 2,250 in both cases.

### Correct — `corrected_tier1` × standalone `tier2_corrected/tier2_recomputed.json`'s `correct`

| | judge pass | judge fail |
|---|---|---|
| **strict pass** | 881 | 67 |
| **strict fail** | 949 | 353 |

- Strict RSR = (881+67)/2250 = **42.13% → 42.1%** (matches the paper, Table 3 mean)
- Judge RSR = (881+949)/2250 = **81.33% → 81.3%** (matches the paper, Table 3 mean)
- McNemar discordant pairs: 949 strict-fail/judge-pass vs. 67 strict-pass/judge-fail
- χ² = (949−67)²/(949+67) = **765.67 → 765.7** (matches the paper, §4.4)

Three numbers printed in the paper are reproduced simultaneously from a join that was not
tuned to hit them; this is the join behind the McNemar 949-vs-67 discordant pair.

### Wrong — `corrected_tier1` × this file's own embedded (stale) `tier2_correct`

| | judge pass | judge fail |
|---|---|---|
| **strict pass** | 571 | 377 |
| **strict fail** | 710 | 592 |

- Strict RSR = (571+377)/2250 = 42.13% → 42.1% (same — strict side is unaffected)
- Judge RSR = (571+710)/2250 = **56.93% → 56.9%** (WRONG — does not match the paper's 81.3%; this is the known pre-fix number recorded in project memory, not the paper)
- McNemar discordant pairs: 710 strict-fail/judge-pass vs. 377 strict-pass/judge-fail
- χ² = (710−377)²/(710+377) = **102.0** (WRONG — does not match the paper's 765.7)

## Bottom line

Cross-tabulating with the embedded field instead of the standalone file silently
produces a completely different, wrong table (710/377 discordant instead of 949/67;
χ²=102.0 instead of 765.7; judge RSR=56.9% instead of 81.3%) — with no error or crash to
signal the mistake. Anyone recomputing Exp.3 contingency tables, McNemar tests, or
strict-vs-judge RSR comparisons from these artifacts in the future **must** join
`tier1_recomputed.json`'s `corrected_tier1` against the *standalone*
`tier2_corrected/tier2_recomputed.json` file's own `correct` field, never against the
bundled copy inside this file.

Both cross-tabs above can be reproduced directly from the two shipped files with the
snippet at the top of this note.
