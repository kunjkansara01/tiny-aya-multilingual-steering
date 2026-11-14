"""Small shared helpers: JSONL I/O and prompt-set loading. Standard library only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SETS = ("harmful", "benign_edgy", "capability")
LANGS = ("en", "hi", "gu")


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict) -> None:
    """Append one row and flush, so a Colab disconnect loses at most the current batch."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def parse_csv(value: str | None, cast=str) -> list:
    if not value:
        return []
    return [cast(x.strip()) for x in str(value).replace(" ", ",").split(",") if x.strip()]


def load_eval_prompts(prompt_dir: str | Path, langs: list[str], limit: int | None = None) -> list[dict]:
    """All eval rows for ``langs``; ``limit`` keeps the first N rows of every (lang, set)."""
    rows: list[dict] = []
    for lang in langs:
        for name in SETS:
            path = Path(prompt_dir) / lang / f"{name}.jsonl"
            part = read_jsonl(path)
            if not part:
                raise FileNotFoundError(f"missing or empty prompt file: {path}")
            for r in part[:limit] if limit else part:
                rows.append({**r, "lang": lang, "set": name, "uid": f"{lang}/{name}/{r['id']}"})
    return rows


def load_calibration(prompt_dir: str | Path, langs: list[str]) -> tuple[list[dict], list[dict]]:
    base = Path(prompt_dir) / "calib"
    harmful = [r for r in read_jsonl(base / "harmful.jsonl") if r["lang"] in langs]
    harmless = [r for r in read_jsonl(base / "harmless.jsonl") if r["lang"] in langs]
    if not harmful or not harmless:
        raise FileNotFoundError(f"no calibration prompts for langs={langs} under {base}")
    return harmful, harmless
