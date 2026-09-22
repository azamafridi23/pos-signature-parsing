#!/usr/bin/env python3
"""Generate prompt-source snapshots and verify every archived prompt.

This script is intentionally offline. It checks the prompt text preserved in all
12 final response archives against the six source prompt files and the exact
message-building rules used during collection.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
MANIFEST_PATH = OUT_DIR / "prompt_provenance.json"
SNAPSHOT_PATH = OUT_DIR / "prompt_source_snapshot.md"

SOURCE_BY_CONDITION = {
    "zero_shot": "prompts/zero_shot.txt",
    "few_shot_fixed": "prompts/few_shot_fixed.txt",
    "few_shot_retrieval": "prompts/few_shot_retrieval.txt",
    "semantic_retrieval": "prompts/semantic_retrieval.txt",
    "cot_few_shot": "prompts/cot_few_shot.txt",
    "critique_refine": "prompts/critique_refine.txt",
}
MODELS = ("gpt_oss_120b", "qwen25_72b")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_first_pass_user(text: str, gold: list[dict[str, Any]]) -> str:
    token_lines = "\n".join(f"{token['id']}\t{token['form']}" for token in gold)
    return f"# text = {text}\n{token_lines}"


def build_critique_user(
    text: str,
    gold: list[dict[str, Any]],
    llm_parse: list[dict[str, Any]],
) -> str:
    token_lines = "\n".join(f"{token['id']}\t{token['form']}" for token in gold)
    existing_parse = json.dumps({"tokens": llm_parse}, indent=2)
    return (
        f"Sentence: {text}\n\n"
        f"Tokens:\n{token_lines}\n\n"
        f"Existing parse:\n{existing_parse}"
    )


def build_short_critique_user(
    text: str,
    llm_parse: list[dict[str, Any]],
) -> str:
    existing_parse = json.dumps({"tokens": llm_parse}, indent=2)
    return f"# text = {text}\n\nExisting parse:\n{existing_parse}"


def validate_retrieval_system(system: str, base: str) -> bool:
    prefix = base + "\n\n"
    if not system.startswith(prefix):
        return False
    suffix = system[len(prefix) :]
    blocks = suffix.split("\n\nInput:\n")
    if not blocks or not blocks[0].startswith("Input:\n"):
        return False
    rendered_blocks = [blocks[0]]
    rendered_blocks.extend("Input:\n" + block for block in blocks[1:])
    # The collector requested exactly three demonstrations for every archived query.
    if len(rendered_blocks) != 3:
        return False
    for block in rendered_blocks:
        if "\n\nOutput:\n" not in block:
            return False
        input_part, output_part = block.split("\n\nOutput:\n", 1)
        if not input_part.startswith("Input:\n# text = "):
            return False
        try:
            parsed = json.loads(output_part)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict) or not isinstance(parsed.get("tokens"), list):
            return False
    return True


def source_summary() -> tuple[dict[str, str], list[dict[str, Any]]]:
    source_texts: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    for condition, relative_path in SOURCE_BY_CONDITION.items():
        path = ROOT / relative_path
        raw = path.read_bytes()
        collector_text = raw.decode("utf-8").strip()
        source_texts[condition] = collector_text
        summaries.append(
            {
                "condition": condition,
                "path": relative_path,
                "bytes": len(raw),
                "lines": len(raw.decode("utf-8").splitlines()),
                "file_sha256": sha256_bytes(raw),
                "collector_text_sha256": sha256_text(collector_text),
            }
        )
    return source_texts, summaries


def load_source_parses(model: str) -> dict[str, list[dict[str, Any]]]:
    path = ROOT / "outputs" / model / "cot_few_shot" / "model_evaluation.json.gz"
    records = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    return {
        record["sent_id"]: record["llm_parse"]
        for record in records
        if isinstance(record.get("llm_parse"), list)
    }


def audit_archive(
    model: str,
    condition: str,
    source_texts: dict[str, str],
) -> dict[str, Any]:
    relative_path = f"outputs/{model}/{condition}/model_responses.json.gz"
    path = ROOT / relative_path
    records = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    source_system = source_texts[condition]
    source_parses = load_source_parses(model) if condition == "critique_refine" else {}

    static_system_matches = 0
    retrieval_template_matches = 0
    regenerated_user_matches = 0
    standard_critique_user_matches = 0
    short_critique_user_matches = 0
    system_hashes: set[str] = set()
    user_hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    prompt_records: list[dict[str, str | None]] = []

    for record in records:
        prompt = record["prompt"]
        system = prompt["system"]
        user = prompt["user"]
        system_hash = sha256_text(system)
        user_hash = sha256_text(user)
        prompt_hash = sha256_text(system + "\0" + user)
        system_hashes.add(system_hash)
        user_hashes.add(user_hash)
        prompt_hashes.add(prompt_hash)

        if condition in {"few_shot_retrieval", "semantic_retrieval"}:
            if validate_retrieval_system(system, source_system):
                retrieval_template_matches += 1
        elif system == source_system:
            static_system_matches += 1

        user_template: str | None = None
        if condition == "critique_refine":
            llm_parse = source_parses.get(record["sent_id"])
            if llm_parse is not None:
                standard_user = build_critique_user(
                    record["text"],
                    record["gold"],
                    llm_parse,
                )
                short_user = build_short_critique_user(
                    record["text"],
                    llm_parse,
                )
                if user == standard_user:
                    regenerated_user_matches += 1
                    standard_critique_user_matches += 1
                    user_template = "standard_critique"
                elif user == short_user:
                    regenerated_user_matches += 1
                    short_critique_user_matches += 1
                    user_template = "short_critique"
        else:
            expected_user = build_first_pass_user(record["text"], record["gold"])
            if user == expected_user:
                regenerated_user_matches += 1
                user_template = "first_pass"

        prompt_records.append(
            {
                "sent_id": record["sent_id"],
                "system_sha256": system_hash,
                "user_sha256": user_hash,
                "prompt_pair_sha256": prompt_hash,
                "user_template": user_template,
            }
        )

    total = len(records)
    system_verified = (
        retrieval_template_matches == total
        if condition in {"few_shot_retrieval", "semantic_retrieval"}
        else static_system_matches == total
    )
    return {
        "model": model,
        "condition": condition,
        "archive_path": relative_path,
        "archive_sha256": file_sha256(path),
        "record_count": total,
        "source_system_path": SOURCE_BY_CONDITION[condition],
        "source_system_verified_for_all_records": system_verified,
        "regenerated_user_verified_for_all_records": regenerated_user_matches == total,
        "static_system_matches": static_system_matches,
        "retrieval_template_matches": retrieval_template_matches,
        "regenerated_user_matches": regenerated_user_matches,
        "standard_critique_user_matches": standard_critique_user_matches,
        "short_critique_user_matches": short_critique_user_matches,
        "unique_system_prompts": len(system_hashes),
        "unique_user_messages": len(user_hashes),
        "unique_prompt_pairs": len(prompt_hashes),
        "records": prompt_records,
    }


def write_snapshot(source_summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Prompt Source Snapshot",
        "",
        "This snapshot contains the exact UTF-8 prompt "
        "source files used by the collectors. SHA-256 digests are computed over "
        "the source-file bytes. Query-specific effective prompts are preserved in "
        "`outputs/<model>/<condition>/model_responses.json.gz`; "
        "`docs/prompt_provenance.json` provides a digest for every "
        "archived system/user pair.",
        "",
        "## Source files",
        "",
        "| Condition | Artifact path | Bytes | Lines | SHA-256 |",
        "|---|---|---:|---:|---|",
    ]
    for item in source_summaries:
        lines.append(
            f"| `{item['condition']}` | `{item['path']}` | {item['bytes']} | "
            f"{item['lines']} | `{item['file_sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Exact dynamic message construction",
            "",
            "For the five first-pass conditions, the user message is exactly:",
            "",
            "````text",
            "# text = {sentence_text}",
            "{token_id_1}\\t{token_form_1}",
            "...",
            "{token_id_n}\\t{token_form_n}",
            "````",
            "",
            "For POS-signature and semantic retrieval, the corresponding source "
            "file is followed by two newline characters and three blocks rendered "
            "exactly as follows, with two newline characters between blocks:",
            "",
            "````text",
            "Input:",
            "# text = {retrieved_training_sentence}",
            "{token_id_1}\\t{token_form_1}",
            "...",
            "",
            "Output:",
            "{",
            '  "tokens": [',
            "    {",
            '      "id": {integer},',
            '      "form": "{exact_form}",',
            '      "upos": "{gold_upos}",',
            '      "head": {gold_head},',
            '      "deprel": "{gold_deprel}"',
            "    }",
            "  ]",
            "}",
            "````",
            "",
            "For 4,146 Critique-Refine calls, the user message is exactly:",
            "",
            "````text",
            "Sentence: {sentence_text}",
            "",
            "Tokens:",
            "{token_id_1}\\t{token_form_1}",
            "...",
            "{token_id_n}\\t{token_form_n}",
            "",
            "Existing parse:",
            "{",
            '  "tokens": {extracted_CoT_token_array_serialized_with_indent_2}',
            "}",
            "````",
            "",
            "Five gpt-oss Critique-Refine calls used the following shorter template:",
            "",
            "````text",
            "# text = {sentence_text}",
            "",
            "Existing parse:",
            "{",
            '  "tokens": {extracted_CoT_token_array_serialized_with_indent_2}',
            "}",
            "````",
            "",
            "The five short-template sentence IDs are:",
            "",
            "- `email-enronsent04_02-0013`",
            "- `newsgroup-groups.google.com_jokecity_0566f0ba3b5f748f_ENG_20051125_240500-0003`",
            "- `weblog-blogspot.com_aggressivevoicedaily_20060629164800_ENG_20060629_164800-0003`",
            "- `weblog-juancole.com_juancole_20040722101300_ENG_20040722_101300-0012`",
            "- `weblog-juancole.com_juancole_20041109060653_ENG_20041109_060653-0010`",
            "",
            "The braces above denote documented dynamic fields; they were not sent "
            "literally. The manifest records `user_template` for every call. The "
            "archived `prompt.system` and `prompt.user` strings are the authoritative "
            "per-query records.",
            "",
            "## Exact fixed source content",
            "",
        ]
    )

    for condition, relative_path in SOURCE_BY_CONDITION.items():
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        lines.extend(
            [
                f"### `{condition}` — `{relative_path}`",
                "",
                "````text",
                content,
                "````",
                "",
            ]
        )
    SNAPSHOT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    source_texts, summaries = source_summary()
    archives = [
        audit_archive(model, condition, source_texts)
        for model in MODELS
        for condition in SOURCE_BY_CONDITION
    ]
    manifest = {
        "schema_version": 1,
        "hash_algorithm": "SHA-256",
        "prompt_pair_hash_definition": "sha256(UTF8(system) || NUL || UTF8(user))",
        "source_prompts": summaries,
        "archives": archives,
        "all_archives_verified": all(
            item["source_system_verified_for_all_records"]
            and item["regenerated_user_verified_for_all_records"]
            for item in archives
        ),
    }
    if not manifest["all_archives_verified"]:
        raise SystemExit("Prompt reconstruction failed; refusing to generate provenance.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_snapshot(summaries)
    print(f"audited {sum(item['record_count'] for item in archives)} prompt pairs")
    print(f"all archives verified: {manifest['all_archives_verified']}")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"wrote {SNAPSHOT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
