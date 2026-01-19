#!/usr/bin/env python3
"""
fewshot_noise_generator_openai.py

Few-shot, OpenAI-API-backed noise generator for OSWorld task JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ----------------------------- IO helpers -----------------------------

def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj: Dict[str, Any], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")

def iter_inputs(inp: Path) -> List[Path]:
    if inp.is_dir():
        return sorted([p for p in inp.rglob("*.json") if p.is_file()])
    return [inp]

def json_minify(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# ----------------------------- constraint helpers -----------------------------

def extract_target_keywords(task: Dict[str, Any]) -> List[str]:
    """
    If evaluator.expected.rules.type == 'keywords', treat those keywords as "targets"
    that noise should not introduce (or increase occurrences of), to avoid breaking deletion checks.
    """
    rules = (((task.get("evaluator") or {}).get("expected") or {}).get("rules") or {})
    if isinstance(rules, dict) and rules.get("type") == "keywords":
        kw = rules.get("keywords")
        if isinstance(kw, list):
            return [str(x).lower() for x in kw]
    return []

def validate_constraints(clean: Dict[str, Any], noisy: Dict[str, Any]) -> List[str]:
    """
    Returns a list of violations; empty means OK.
    """
    v: List[str] = []

    if noisy.get("instruction") != clean.get("instruction"):
        v.append("instruction_changed")

    if (noisy.get("evaluator") or {}).get("func") != (clean.get("evaluator") or {}).get("func"):
        v.append("evaluator_func_changed")

    if (noisy.get("evaluator") or {}).get("expected") != (clean.get("evaluator") or {}).get("expected"):
        v.append("evaluator_expected_changed")

    # config subsequence check
    clean_cfg = clean.get("config", [])
    noisy_cfg = noisy.get("config", [])
    if not isinstance(clean_cfg, list) or not isinstance(noisy_cfg, list):
        v.append("config_not_list")
        return v

    j = 0
    for step in clean_cfg:
        found = False
        while j < len(noisy_cfg):
            if noisy_cfg[j] == step:
                found = True
                j += 1
                break
            j += 1
        if not found:
            v.append("clean_config_step_missing_or_reordered")
            break

    # avoid introducing more target keywords
    target_kw = extract_target_keywords(clean)
    if target_kw:
        noisy_str = json_minify(noisy).lower()
        clean_str = json_minify(clean).lower()
        for k in target_kw:
            if noisy_str.count(k) > clean_str.count(k):
                v.append(f"introduced_target_keyword:{k}")

    return v


# ----------------------------- OpenAI backend -----------------------------

@dataclass
class OpenAIConfig:
    api_key: str
    model: str
    effort: str  # "low" | "medium" | "high" (or model-dependent)
    temperature: float
    base_url: str | None = None
    store: bool = False


def call_openai_noisy_json(prompt_messages: List[Dict[str, Any]], cfg: OpenAIConfig) -> Dict[str, Any]:
    """
    Calls OpenAI Chat Completions API with JSON mode and returns parsed JSON object.
    """
    from openai import OpenAI

    client_kwargs: Dict[str, Any] = {"api_key": cfg.api_key}
    if cfg.base_url:
        client_kwargs["base_url"] = cfg.base_url
    client = OpenAI(**client_kwargs)

    # Use Chat Completions API with JSON mode
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=prompt_messages,
        temperature=cfg.temperature,
        response_format={"type": "json_object"}
    )
    
    # Extract the output text from chat completion
    out_text = (resp.choices[0].message.content or "").strip()
    if not out_text:
        raise ValueError("Empty model output.")

    try:
        return json.loads(out_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output was not valid JSON: {e}\n---\n{out_text[:1000]}")


# ----------------------------- prompting -----------------------------

def build_messages(
    clean_task: Dict[str, Any],
    examples: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    noise_level: str,
) -> List[Dict[str, Any]]:
    """
    Few-shot messages: show examples, then request transformation.
    IMPORTANT: We ask for JSON-only output; schema enforcement is also used in the API call.
    """
    target_kw = extract_target_keywords(clean_task)

    constraints = [
        "Return ONLY a single JSON object. No markdown. No commentary.",
        "Keep the 'instruction' field EXACTLY the same as input.",
        "Do NOT change evaluator.func or evaluator.expected.",
        "Preserve the original config steps as a subsequence in the same order (you may insert/append noise).",
        "Add only light GUI-realism noise: a few distractor pages/tabs, ad/tracker-like benign URLs, extra history items, or extra irrelevant files, as appropriate.",
        "Do NOT introduce accounts/credentials or new external dependencies.",
        "Do NOT add anything unsafe; use benign placeholder domains like example.com / ads.example.com.",
    ]
    if target_kw:
        constraints.append(
            f"Avoid introducing MORE occurrences of these target keywords anywhere in the JSON: {target_kw}."
        )

    # Build a compact few-shot payload
    parts: List[str] = []
    parts.append("You transform OSWorld task JSONs by adding realistic GUI noise, imitating the examples.")
    parts.append(f"NOISE LEVEL: {noise_level}")
    parts.append("CONSTRAINTS:\n- " + "\n- ".join(constraints))

    for i, (c, n) in enumerate(examples, 1):
        parts.append(f"\nEXAMPLE {i} CLEAN:\n{json.dumps(c, indent=2, ensure_ascii=False)}")
        parts.append(f"\nEXAMPLE {i} NOISY:\n{json.dumps(n, indent=2, ensure_ascii=False)}")

    parts.append("\nINPUT CLEAN TASK:\n" + json.dumps(clean_task, indent=2, ensure_ascii=False))
    parts.append("\nOUTPUT NOISY JSON (single JSON object only):")

    user_content = "\n".join(parts)

    return [
        {"role": "system", "content": "You are a careful data generator. Output ONLY a valid JSON object."},
        {"role": "user", "content": user_content},
    ]


# ----------------------------- CLI -----------------------------

def parse_example_pairs(pairs: List[str]) -> List[Tuple[Path, Path]]:
    out: List[Tuple[Path, Path]] = []
    for s in pairs:
        if ":" not in s:
            raise ValueError(f"Bad --examples entry (expected clean:noisy): {s}")
        a, b = s.split(":", 1)
        out.append((Path(a), Path(b)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="osworld_examples/tasks", help="Input JSON file or directory")
    ap.add_argument("--out", default="osworld_examples/noisy_tasks", help="Output directory")
    ap.add_argument("--examples", nargs="+", required=True, help="Few-shot pairs clean.json:noisy.json")

    ap.add_argument("--model", default="gpt-4o", help="Model name, e.g., gpt-4o")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"], help="Reasoning effort (not used with chat completions)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--noise_level", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--suffix", default="_noise", help="Output filename suffix (default: _noise)")
    ap.add_argument("--base_url", default=None, help="Optional base_url (leave blank unless needed)")
    ap.add_argument("--store", action="store_true", help="Store responses (default false)")

    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY. Set it like:\n"
            "  export OPENAI_API_KEY='YOUR_KEY'\n"
        )

    cfg = OpenAIConfig(
        api_key=api_key,
        model=args.model,
        effort=args.effort,
        temperature=args.temperature,
        base_url=args.base_url,
        store=bool(args.store),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load example pairs
    examples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for clean_p, noisy_p in parse_example_pairs(args.examples):
        examples.append((load_json(clean_p), load_json(noisy_p)))

    # Process inputs
    inputs = iter_inputs(Path(args.inputs))
    if not inputs:
        raise SystemExit(f"No input JSON files found at: {args.inputs}")

    for p in inputs:
        clean_task = load_json(p)
        messages = build_messages(clean_task, examples, noise_level=args.noise_level)
        noisy_task = call_openai_noisy_json(messages, cfg)

        violations = validate_constraints(clean_task, noisy_task)
        noisy_task.setdefault("noise_meta", {})
        noisy_task["noise_meta"].update({
            "generated_by": "fewshot_noise_generator_openai.py",
            "noise_level": args.noise_level,
            "model": args.model,
            "effort": args.effort,
            "temperature": args.temperature,
        })
        if violations:
            noisy_task["noise_meta"]["validation_violations"] = violations

        out_name = f"{p.stem}{args.suffix}.json"  # <original>_noise.json
        dump_json(noisy_task, out_dir / out_name)

    print(f"Done. Wrote {len(inputs)} noisy JSON(s) to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()