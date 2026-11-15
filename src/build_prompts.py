"""Rebuild the aya_redteaming-derived prompt files (reproducible, seeded).

Writes:
  prompts/en/harmful.jsonl      40 prompts sampled from CohereLabs/aya_redteaming (english)
  prompts/hi/harmful.jsonl      40 prompts sampled from CohereLabs/aya_redteaming (hindi)
  prompts/calib/harmful.jsonl   32 EN + 32 HI held-out prompts (disjoint from the eval sets)

Author-curated files (LLM-assisted drafts, NOT touched by this script):
  prompts/gu/harmful.jsonl      Gujarati translations of the EN harmful set (same ids)
  prompts/*/benign_edgy.jsonl, prompts/*/capability.jsonl, prompts/calib/harmless.jsonl

Sampling rules (see FINDINGS.md):
  * only "global" harm rows (not culture-specific "local" ones);
  * drop rows tagged Non-consensual sexual content / Graphic material / Profanity
    (refusal is not clearly the expected behaviour for profanity; the other two are
    excluded on content grounds), and anything mentioning minors;
  * keep request-shaped prompts (imperatives / how-to), so "refused or not" is meaningful
    for a pattern-based classifier;
  * stratify 8 per primary category over the 5 remaining categories.

Usage:  python src/build_prompts.py [--aya-dir vendor/aya_redteaming]
        (downloads the two jsonl files from the Hub if the directory is missing)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUB = "https://huggingface.co/datasets/CohereLabs/aya_redteaming/resolve/main/{}"
FILES = {"en": "aya_eng.jsonl", "hi": "aya_hin.jsonl"}
AYA_SOURCE = "sampled from CohereLabs/aya_redteaming (seeded)"
SEED = 20261114

CATEGORIES = [
    "Violence, Threats & Incitement",
    "Self-Harm",
    "Hate Speech",
    "Bullying & Harassment",
    "Discrimination & Injustice",
]
DROP_CATEGORIES = {"Non-consensual sexual content", "Graphic material", "Profanity"}

MINOR_RE = re.compile(
    r"year[- ]old|\bchild|\bkid|\bminor|\bteen|underage|\bgirl|\bboy|\bbaby|student|school"
    r"|साल क|वर्ष क|वर्षीय|नाबालिग|बच्च|लड़क|छात्र|स्कूल|बेटी|बेटे|शिशु",
    re.I,
)
SEXUAL_RE = re.compile(r"\bsex|rape|nude|porn|सेक्स|बलात्कार|चुदा|चोद|नग्न|अश्लील", re.I)
REQUEST_RE = {
    "en": re.compile(
        r"^(how (do|can|could|would|to)|give|list|write|tell|generate|teach|describe|explain|outline|"
        r"create|suggest|what('s| is| are) (the|a|some) (best|good|easiest|most|ways?)|help|provide|show)\b",
        re.I,
    ),
    "hi": re.compile(r"बताओ|बताएं|बताइए|बताइये|लिखें|लिखो|लिखिए|कैसे|तरीक|सूची|सुझा|समझा|बनाएं|बनाओ"),
}


def load_rows(aya_dir: Path, lang: str) -> list[dict]:
    path = aya_dir / FILES[lang]
    if not path.exists():
        aya_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {FILES[lang]} ...")
        urllib.request.urlretrieve(HUB.format(FILES[lang]), path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["cats"] = json.loads(r["harm_category"])
            r["prompt"] = " ".join(r["prompt"].split())
            rows.append(r)
    return rows


def eligible(rows: list[dict], lang: str) -> list[dict]:
    out, seen = [], set()
    for r in rows:
        p = r["prompt"]
        if r.get("global_or_local") != "global" or p in seen:
            continue
        if DROP_CATEGORIES & set(r["cats"]) or not set(r["cats"]) & set(CATEGORIES):
            continue
        if MINOR_RE.search(p) or SEXUAL_RE.search(p):
            continue
        if not 25 <= len(p) <= 220 or not REQUEST_RE[lang].search(p):
            continue
        seen.add(p)
        out.append(r)
    return out


def stratified(rows: list[dict], per_cat: int, rng: random.Random, exclude: set[str]) -> list[dict]:
    """per_cat rows per category, assigning each row to the first listed category it carries."""
    picked: list[dict] = []
    used = set(exclude)
    for cat in CATEGORIES:
        pool = [r for r in rows if r["cats"][0] == cat and r["prompt"] not in used]
        if len(pool) < per_cat:  # fall back to any row tagged with the category
            pool += [r for r in rows if cat in r["cats"] and r["prompt"] not in used and r not in pool]
        rng.shuffle(pool)
        for r in pool[:per_cat]:
            used.add(r["prompt"])
            picked.append({**r, "category": cat})
    return picked


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):3d} -> {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aya-dir", default=str(ROOT / "vendor" / "aya_redteaming"))
    ap.add_argument("--out-dir", default=str(ROOT / "prompts"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    calib: list[dict] = []
    for lang in ("en", "hi"):
        rng = random.Random(f"{SEED}-{lang}")
        rows = eligible(load_rows(Path(args.aya_dir), lang), lang)
        print(f"{lang}: {len(rows)} eligible rows")
        ev = stratified(rows, 8, rng, exclude=set())
        write_jsonl(out / lang / "harmful.jsonl", [
            {"id": f"harm-{i:02d}" if lang == "en" else f"harm-hi-{i:02d}", "lang": lang, "set": "harmful",
             "category": r["category"], "prompt": r["prompt"], "source": AYA_SOURCE}
            for i, r in enumerate(ev, 1)])
        rest = [r for r in rows if r["prompt"] not in {e["prompt"] for e in ev}]
        rng.shuffle(rest)
        calib += [{"id": f"calib-harm-{lang}-{i:02d}", "lang": lang, "set": "calib_harmful",
                   "category": r["cats"][0], "prompt": r["prompt"], "source": AYA_SOURCE}
                  for i, r in enumerate(rest[:32], 1)]
    write_jsonl(out / "calib" / "harmful.jsonl", calib)


if __name__ == "__main__":
    main()
