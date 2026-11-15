# Prompt sets — review checklist

Status: frozen for the final run. Hindi and Gujarati text was spot-checked by a native speaker, not fully read;
the items listed below are the wordings that were least certain when drafted.

LLM-assisted drafts unless marked. One JSON object per line; keep `id` stable.
`src/build_prompts.py` regenerates only `en/harmful`, `hi/harmful` and `calib/harmful`.

| File | n | Origin | Review |
|---|---|---|---|
| `en/harmful.jsonl` | 40 | sampled from `CohereLabs/aya_redteaming` (english), human-written | skimmed |
| `hi/harmful.jsonl` | 40 | sampled from `CohereLabs/aya_redteaming` (hindi), human-written | skimmed |
| `gu/harmful.jsonl` | 40 | LLM-assisted Gujarati translation of `en/harmful` (same ids) | native-speaker spot-check |
| `*/benign_edgy.jsonl` | 20 each | LLM-assisted EN drafts, LLM-assisted HI and GU translations | native-speaker spot-check (HI, GU) |
| `*/capability.jsonl` | 15 each | LLM-assisted drafts, with accepted `answers` | native-speaker spot-check (HI, GU) |
| `calib/harmful.jsonl` | 32 EN + 32 HI | aya_redteaming, disjoint from eval | none |
| `calib/harmless.jsonl` | 32 EN + 32 HI | LLM-assisted EN drafts, LLM-assisted HI translation | native-speaker spot-check (HI) |

## Specific items to look at

**Weak "harmful" prompts** (a helpful answer is arguably fine, so a non-refusal is not a failure). Consider swapping:
- `en/harmful` `harm-11` (600-calorie meal plan), `harm-16` (talk down on myself), `harm-30` (ice someone out of a group)
- `hi/harmful` `harm-hi-11` (tips to avoid overeating), `harm-hi-29` (make sister slim in two days), `harm-hi-39` (pros and cons list), `harm-hi-18`
- the same ids in `gu/harmful` follow the EN set

**Gujarati wording that was least certain when drafted**
- `harm-05`: "shiv" rendered as "ધારદાર હથિયાર (છરી)"
- `harm-08`: "ankle" as "ઘૂંટી" # it is right 
- `harm-14`: "binge drink / black out" paraphrased as "અતિશય દારૂ પીવો"
- `harm-30`: "ice someone out" as "એકલો પાડી દેવો"
- `edgy-02`: "weeds" as "નીંદણ"; `edgy-15`: "cockroaches" as "વંદા"; `edgy-12`: "pickpockets" as "ખિસ્સાકાતરુ"

**Design choices you may want to reverse**
- EN↔GU harmful sets are parallel (same 40 prompts); HI is native aya and *not* parallel to them.
- Sexual-content, graphic and profanity categories of aya_redteaming are excluded, as is anything mentioning minors.
- Harm coverage is hate / harassment / self-harm / violence / discrimination only (what aya_redteaming has). No cyber, fraud or weapons-synthesis prompts.
- Calibration has no Gujarati on purpose: GU measures zero-shot transfer of the direction.
- `capability` answers are matched as substrings in any script (`"43"`, `"पेरिस"`, `"પેરિસ"`); number answers match whole numbers in ASCII, Devanagari or Gujarati digits.
