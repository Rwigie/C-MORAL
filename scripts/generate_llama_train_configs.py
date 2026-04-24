#!/usr/bin/env python3
"""Batch-generate Llama GRPO configs from existing Mistral task configs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def replace_model_block(lines: list[str]) -> list[str]:
    start = None
    for i, line in enumerate(lines):
        if line.startswith("model:"):
            start = i
            break
    if start is None:
        raise ValueError("Missing top-level 'model:' block")

    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
            end = i
            break

    new_header = ["include:", "  - model/llama.yaml", ""]
    return lines[:start] + new_header + lines[end:]


def patch_runs_dir(line: str) -> str:
    # e.g. runs_dir: "Molo/runs/bpq/grpo" -> "Molo/runs/bpq/llama/grpo"
    m = re.match(r'^(\s*runs_dir:\s*["\']?Molo/runs/[^/"\']+)(/grpo["\']?\s*)$', line)
    if not m:
        return line
    return f"{m.group(1)}/llama{m.group(2)}"


def patch_wandb_project(line: str) -> str:
    m = re.match(r'^(\s*wandb_project:\s*)(["\']?)([^"\'\s]+)(["\']?\s*)$', line)
    if not m:
        return line

    prefix, q1, project, q2 = m.groups()
    if project.startswith("llama_"):
        return line
    new_project = f"llama_{project}"
    quote = q1 if q1 else q2
    if quote:
        return f"{prefix}{quote}{new_project}{quote}"
    return f"{prefix}{new_project}"


def transform_content(text: str) -> str:
    lines = text.splitlines()
    lines = replace_model_block(lines)
    out: list[str] = []
    for line in lines:
        line = patch_runs_dir(line)
        line = patch_wandb_project(line)
        out.append(line)
    return "\n".join(out) + "\n"


def generate(source: Path, target: Path, overwrite: bool) -> str:
    if target.exists() and not overwrite:
        return "skipped"

    src_text = source.read_text(encoding="utf-8")
    tgt_text = transform_content(src_text)
    target.write_text(tgt_text, encoding="utf-8")
    return "written"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Llama train configs from Mistral configs")
    parser.add_argument("--configs-dir", default="configs", help="Directory containing train_config_*.yaml")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target files")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    sources = sorted(configs_dir.glob("train_config_mistral_*.yaml"))
    if not sources:
        raise FileNotFoundError(f"No source configs found under: {configs_dir}")

    written = 0
    skipped = 0
    for source in sources:
        suffix = source.stem.removeprefix("train_config_mistral_")
        target = configs_dir / f"train_config_llama_{suffix}.yaml"
        status = generate(source, target, overwrite=args.overwrite)
        print(f"[{status}] {target}")
        if status == "written":
            written += 1
        else:
            skipped += 1

    print(f"Done. written={written}, skipped={skipped}")


if __name__ == "__main__":
    main()
