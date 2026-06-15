#!/usr/bin/env python3
"""Collect top LLM models from Hugging Face Hub, filter derivatives, export JSON."""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_PATH = DATA_DIR / "models.json"

PIPELINE_TAGS = ["text-generation", "text2text-generation"]
FETCH_LIMIT = 5000
FINAL_LIMIT = 1000

EXPAND_FIELDS = [
    "downloadsAllTime",
    "safetensors",
    "tags",
    "createdAt",
    "lastModified",
    "likes",
    "author",
]

DERIVATIVE_RELATIONS = {"quantized", "adapter", "merge"}

QUANT_ID_PATTERN = re.compile(
    r"[-_\.](GGUF|GPTQ|AWQ|EXL2|GGML|bnb|4bit|8bit|fp16)(?:[-_\.]|$)",
    re.IGNORECASE,
)


def fetch_models(api: HfApi) -> list:
    """Fetch models for all target pipeline tags, deduplicate by ID."""
    seen = set()
    models = []
    for tag in PIPELINE_TAGS:
        print(f"Fetching up to {FETCH_LIMIT} models for pipeline_tag={tag}...")
        for m in api.list_models(
            pipeline_tag=tag,
            sort="downloads",
            limit=FETCH_LIMIT,
            expand=EXPAND_FIELDS,
        ):
            if m.id not in seen:
                seen.add(m.id)
                models.append(m)
    print(f"Fetched {len(models)} unique models across {len(PIPELINE_TAGS)} pipeline tags.")
    return models


def is_gguf_library(model) -> bool:
    return getattr(model, "library_name", None) == "gguf"


def has_derivative_tag(model) -> bool:
    for tag in getattr(model, "tags", None) or []:
        if tag.startswith("base_model:"):
            parts = tag.split(":")
            if len(parts) >= 3 and parts[1] in DERIVATIVE_RELATIONS:
                return True
    return False


def has_finetune_from_different_org(model) -> bool:
    author = getattr(model, "author", None) or ""
    for tag in getattr(model, "tags", None) or []:
        if tag.startswith("base_model:finetune:"):
            parts = tag.split(":")
            if len(parts) >= 3:
                base_id = parts[2]
                base_org = base_id.split("/")[0] if "/" in base_id else ""
                if base_org and base_org != author:
                    return True
    return False


def has_quant_id(model) -> bool:
    model_name = model.id.split("/")[-1] if "/" in model.id else model.id
    return bool(QUANT_ID_PATTERN.search(model_name))


def filter_models(models: list) -> list:
    """Apply the filtering pipeline to keep only base/official LLMs."""
    before = len(models)

    filtered = []
    for m in models:
        if is_gguf_library(m):
            continue
        if has_derivative_tag(m):
            continue
        if has_quant_id(m):
            continue
        if has_finetune_from_different_org(m):
            continue
        filtered.append(m)

    print(f"Filtered {before} → {len(filtered)} models.")
    return filtered


def get_param_count(model) -> int | None:
    st = getattr(model, "safetensors", None)
    if st is None:
        return None
    pc = getattr(st, "parameter_count", None) or getattr(st, "parameters", None)
    if isinstance(pc, dict) and pc:
        return sum(pc.values())
    return None


def size_bucket(params: int | None) -> str:
    if params is None:
        return "Unknown"
    b = params / 1e9
    if b < 1:
        return "<1B"
    if b < 3:
        return "1-3B"
    if b < 7:
        return "3-7B"
    if b < 13:
        return "7-13B"
    if b < 70:
        return "13-70B"
    return "70B+"


def iso_or_none(dt) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def serialize_model(model) -> dict:
    param_count = get_param_count(model)
    downloads_all_time = getattr(model, "downloads_all_time", None)
    if downloads_all_time is None:
        downloads_all_time = getattr(model, "downloads", 0) or 0
    return {
        "id": model.id,
        "author": getattr(model, "author", None) or (model.id.split("/")[0] if "/" in model.id else ""),
        "downloads": getattr(model, "downloads", 0) or 0,
        "downloads_all_time": downloads_all_time,
        "likes": getattr(model, "likes", 0) or 0,
        "pipeline_tag": getattr(model, "pipeline_tag", None),
        "param_count": param_count,
        "size_bucket": size_bucket(param_count),
        "tags": list(getattr(model, "tags", None) or []),
        "created_at": iso_or_none(getattr(model, "created_at", None)),
        "last_modified": iso_or_none(getattr(model, "last_modified", None)),
    }


def main():
    api = HfApi()

    models = fetch_models(api)
    models = filter_models(models)

    models.sort(key=lambda m: getattr(m, "downloads_all_time", None) or getattr(m, "downloads", 0) or 0, reverse=True)
    models = models[:FINAL_LIMIT]

    serialized = [serialize_model(m) for m in models]

    output = {
        "meta": {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "total_after_filter": len(serialized),
            "pipeline_tags": PIPELINE_TAGS,
        },
        "models": serialized,
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Wrote {len(serialized)} models to {OUTPUT_PATH}")

    top5 = serialized[:5]
    for m in top5:
        dl = m["downloads_all_time"]
        print(f"  {m['id']:50s}  {dl:>15,}  {m['size_bucket']:>8s}")


if __name__ == "__main__":
    main()
