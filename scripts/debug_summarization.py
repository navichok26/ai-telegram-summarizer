#!/usr/bin/env python3
"""Run summarization pipeline against mocked Telegram updates with step-by-step trace output."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import summarize  # noqa: E402


def configure_utf8_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug summarization pipeline locally using Telegram update mocks."
    )
    parser.add_argument(
        "--updates-file",
        required=True,
        help="Path to JSON with mocked Telegram updates (list or {'updates': [...]}).",
    )
    parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="Specific chat id to summarize (default: auto-pick chat with most messages).",
    )
    parser.add_argument("--min-chars", type=int, default=12)
    parser.add_argument("--max-buffer-messages", type=int, default=800)
    parser.add_argument("--merge-window-seconds", type=int, default=180)
    parser.add_argument("--max-merge-messages", type=int, default=3)
    parser.add_argument("--max-merge-span-seconds", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-topics", type=int, default=12)
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument(
        "--max-prompt-chars",
        type=int,
        default=2400,
        help="How many prompt characters to print (0 prints full prompt).",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON: {path}") from exc


def extract_updates(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        updates = payload.get("updates")
        if not isinstance(updates, list):
            raise RuntimeError("Updates payload object must contain list field 'updates'.")
        items = updates
    else:
        raise RuntimeError("Updates payload must be JSON list or object with 'updates'.")

    return [item for item in items if isinstance(item, dict)]


def detect_filter_reason(message: dict[str, Any], min_chars: int) -> str:
    if message.get("via_bot") is not None or message.get("via_bot_id") is not None:
        return "via_bot"

    text = summarize.extract_text_from_live_message(message)
    if text == "" and summarize.is_media_message(message):
        return "media_without_text"
    if len(text) < min_chars:
        return f"short_text<{min_chars}"

    message_id = message.get("message_id")
    date_ts = message.get("date")
    if not isinstance(message_id, int) or not isinstance(date_ts, int):
        return "invalid_message_id_or_date"

    try:
        datetime.fromtimestamp(date_ts, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return "invalid_timestamp"

    return "filtered_unknown"


def build_update_trace(updates: list[dict[str, Any]], runtime_args: SimpleNamespace) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    state = {"offset": 0, "chats": {}}
    chats: dict[str, Any] = state["chats"]

    for update in updates:
        update_id = update.get("update_id")
        message = update.get("message")
        row: dict[str, Any] = {
            "update_id": update_id,
            "status": "skipped",
            "reason": "unknown",
        }

        if not isinstance(message, dict):
            row["reason"] = "non_message_update"
            trace.append(row)
            continue

        row["message_id"] = message.get("message_id")
        chat = message.get("chat")
        if not isinstance(chat, dict):
            row["reason"] = "missing_chat"
            trace.append(row)
            continue

        chat_id = chat.get("id")
        chat_type = chat.get("type")
        row["chat_id"] = chat_id
        row["chat_type"] = chat_type
        if chat_type not in {"group", "supergroup"}:
            row["reason"] = f"non_group:{chat_type}"
            trace.append(row)
            continue

        sender = message.get("from")
        if isinstance(sender, dict) and sender.get("is_bot") is True:
            row["reason"] = "bot_sender"
            trace.append(row)
            continue

        chat_state = summarize.ensure_chat_state(chats, chat)
        normalized = summarize.normalize_live_message(message, runtime_args.min_chars)
        if normalized is None:
            row["reason"] = detect_filter_reason(message, runtime_args.min_chars)
            trace.append(row)
            continue

        seen_ids = chat_state.get("seen_ids")
        was_duplicate = isinstance(seen_ids, list) and normalized.id in seen_ids
        added = summarize.append_message_to_chat_state(
            chat_state=chat_state,
            normalized=normalized,
            max_buffer_messages=runtime_args.max_buffer_messages,
        )
        if added:
            row["status"] = "accepted"
            row["reason"] = "added"
            row["from_name"] = normalized.from_name
            row["text"] = normalized.text
        elif was_duplicate:
            row["reason"] = "duplicate_message_id"
        else:
            row["reason"] = "append_rejected"

        trace.append(row)

    return trace


class LLMProvider:
    def __init__(self, ollama_url: str, model: str) -> None:
        self._ollama_url = ollama_url
        self._model = model

    def call_batch(self, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        parsed, raw = summarize.call_ollama_generate(
            ollama_url=self._ollama_url,
            model=self._model,
            prompt=prompt,
            schema=schema,
        )
        return parsed, raw, "live_ollama"

    def call_semantic_merge(self, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        parsed, raw = summarize.call_ollama_generate(
            ollama_url=self._ollama_url,
            model=self._model,
            prompt=prompt,
            schema=schema,
        )
        return parsed, raw, "live_ollama"


def apply_semantic_merge_result(
    raw_merged_object: dict[str, Any],
    batch_topics: list[dict[str, Any]],
    global_id_to_index: dict[int, int],
    chat_public_id: str,
) -> list[dict[str, Any]]:
    raw_merged_topics = raw_merged_object.get("merged_topics")
    if not isinstance(raw_merged_topics, list):
        raise RuntimeError("Semantic merge response does not contain 'merged_topics' array.")

    indexed_batch_topics = {idx: topic for idx, topic in enumerate(batch_topics, start=1)}
    merged_intermediate: list[dict[str, Any]] = []

    for item in raw_merged_topics:
        if not isinstance(item, dict):
            continue

        title = item.get("title")
        summary = item.get("summary")
        source_topic_indices = item.get("source_topic_indices")
        if not isinstance(title, str) or not isinstance(summary, str) or not isinstance(source_topic_indices, list):
            continue

        title = " ".join(title.split())
        summary = " ".join(summary.split())
        if not title or not summary:
            continue

        valid_indices: list[int] = []
        for source_idx in source_topic_indices:
            if isinstance(source_idx, int) and source_idx in indexed_batch_topics and source_idx not in valid_indices:
                valid_indices.append(source_idx)
        if not valid_indices:
            continue

        evidence_ids: list[int] = []
        participants: list[str] = []
        for source_idx in valid_indices:
            topic = indexed_batch_topics[source_idx]
            for evidence_id in topic["evidence_message_ids"]:
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)
            for participant in topic["participants"]:
                if participant not in participants:
                    participants.append(participant)

        valid_evidence = [evidence_id for evidence_id in evidence_ids if evidence_id in global_id_to_index]
        if not valid_evidence:
            continue

        valid_evidence.sort(key=lambda message_id: global_id_to_index[message_id])
        first_message_id = valid_evidence[0]
        merged_intermediate.append(
            {
                "title": title,
                "summary": summary,
                "evidence_message_ids": valid_evidence,
                "participants": participants,
                "first_message_id": first_message_id,
                "first_message_link": f"https://t.me/c/{chat_public_id}/{first_message_id}",
            }
        )

    if not merged_intermediate:
        raise RuntimeError("Semantic merge returned no valid topics.")

    return summarize.merge_topics_across_batches_by_title(
        batch_topics=merged_intermediate,
        global_id_to_index=global_id_to_index,
        chat_public_id=chat_public_id,
    )


def print_title(text: str) -> None:
    print(f"\n=== {text} ===")


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def maybe_trim(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n... [trimmed {len(text) - max_chars} chars]"


def choose_chat_id(chats: dict[str, Any], preferred_chat_id: int | None) -> int:
    if preferred_chat_id is not None:
        key = str(preferred_chat_id)
        if key not in chats:
            raise RuntimeError(f"Chat {preferred_chat_id} not found after processing updates.")
        return preferred_chat_id

    best_chat_id: int | None = None
    best_count = -1
    for key, chat_state in chats.items():
        if not isinstance(chat_state, dict):
            continue
        messages = chat_state.get("messages")
        count = len(messages) if isinstance(messages, list) else 0
        try:
            chat_id = int(key)
        except ValueError:
            continue
        if count > best_count:
            best_chat_id = chat_id
            best_count = count

    if best_chat_id is None:
        raise RuntimeError("No chats available in processed state.")
    return best_chat_id


def run_debug() -> int:
    configure_utf8_output()
    args = parse_args()

    updates_payload = load_json(Path(args.updates_file))
    updates = extract_updates(updates_payload)

    runtime_args = SimpleNamespace(
        min_chars=args.min_chars,
        max_buffer_messages=args.max_buffer_messages,
        merge_window_seconds=args.merge_window_seconds,
        max_merge_messages=args.max_merge_messages,
        max_merge_span_seconds=args.max_merge_span_seconds,
        batch_size=args.batch_size,
        max_topics=args.max_topics,
        ollama_url=args.ollama_url,
        model=args.model,
    )

    print_title("Input")
    print(f"updates_file: {Path(args.updates_file).resolve()}")
    print("llm_mode: live_ollama")
    print(f"ollama_url: {args.ollama_url}")
    print(f"model: {args.model}")
    print(f"updates_loaded: {len(updates)}")

    print_title("Filter Trace")
    update_trace = build_update_trace(updates, runtime_args)
    print_json(update_trace)

    state: dict[str, Any] = {"offset": 0, "chats": {}}
    changed, stats = summarize.process_updates(state, updates, runtime_args)
    print_title("process_updates Stats")
    print(f"state_changed: {changed}")
    print_json(vars(stats))
    print(f"offset_after_processing: {state.get('offset')}")

    chats = state.get("chats")
    if not isinstance(chats, dict) or not chats:
        print_title("Result")
        print("No chats with accepted messages; nothing to summarize.")
        return 0

    chat_id = choose_chat_id(chats, args.chat_id)
    chat_state = chats[str(chat_id)]
    chat_title = chat_state.get("chat_title", str(chat_id))

    print_title("Selected Chat")
    print(f"chat_id: {chat_id}")
    print(f"chat_title: {chat_title}")

    messages = summarize.deserialize_chat_messages(chat_state)
    print_title("Normalized Messages")
    print_json(
        [
            {
                "id": msg.id,
                "from": msg.from_name,
                "date": msg.date.isoformat(),
                "text": msg.text,
            }
            for msg in messages
        ]
    )

    if not messages:
        print_title("Result")
        print("No normalized messages in selected chat.")
        return 0

    merged_messages = summarize.merge_consecutive_messages(
        messages=messages,
        merge_window_seconds=args.merge_window_seconds,
        max_merge_messages=args.max_merge_messages,
        max_merge_span_seconds=args.max_merge_span_seconds,
    )
    print_title("Merged Chunks")
    print_json(
        [
            {
                "first_message_id": chunk.first_message_id,
                "message_ids": chunk.message_ids,
                "from": chunk.from_name,
                "text": chunk.text,
            }
            for chunk in merged_messages
        ]
    )

    batches = summarize.chunk_merged_messages(merged_messages, args.batch_size)
    print_title("Batches")
    print_json(
        [
            {
                "batch": idx,
                "size": len(batch),
                "message_ids": [item.first_message_id for item in batch],
            }
            for idx, batch in enumerate(batches, start=1)
        ]
    )

    llm_provider = LLMProvider(ollama_url=args.ollama_url, model=args.model)
    chat_public_id = summarize.to_public_chat_id(chat_id)
    global_id_to_index = {chunk.first_message_id: idx for idx, chunk in enumerate(merged_messages)}
    schema = summarize.build_schema(args.max_topics)

    all_batch_topics: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches, start=1):
        allowed_ids = [chunk.first_message_id for chunk in batch]
        message_id_to_author = {chunk.first_message_id: chunk.from_name for chunk in batch}
        llm_chat_payload = summarize.build_llm_chat_payload(batch)
        prompt = summarize.build_prompt(chat_payload=llm_chat_payload, max_topics=args.max_topics)

        print_title(f"Batch {batch_index} Prompt")
        print(maybe_trim(prompt, args.max_prompt_chars))

        raw_topics_object, raw_llm_output, source = llm_provider.call_batch(prompt=prompt, schema=schema)
        print_title(f"Batch {batch_index} LLM Output ({source})")
        print(raw_llm_output)

        batch_topics = summarize.validate_and_enrich_topics(
            raw_topics_object=raw_topics_object,
            allowed_ids=allowed_ids,
            chat_public_id=chat_public_id,
            message_id_to_author=message_id_to_author,
        )
        print_title(f"Batch {batch_index} Validated Topics")
        print_json(batch_topics)
        all_batch_topics.extend(batch_topics)

    if not all_batch_topics:
        print_title("Result")
        print("No valid topics produced by batch extraction.")
        return 0

    merge_prompt = summarize.build_topic_merge_prompt(all_batch_topics)
    print_title("Semantic Merge Prompt")
    print(maybe_trim(merge_prompt, args.max_prompt_chars))

    merge_schema = summarize.build_topic_merge_schema()
    raw_merged_object, raw_semantic_output, source = llm_provider.call_semantic_merge(
        prompt=merge_prompt,
        schema=merge_schema,
    )
    print_title(f"Semantic Merge LLM Output ({source})")
    print(raw_semantic_output)

    merge_method = "semantic_llm"
    try:
        topics = apply_semantic_merge_result(
            raw_merged_object=raw_merged_object,
            batch_topics=all_batch_topics,
            global_id_to_index=global_id_to_index,
            chat_public_id=chat_public_id,
        )
    except RuntimeError as exc:
        merge_method = "title_fallback"
        print_title("Semantic Merge Error")
        print(str(exc))
        topics = summarize.merge_topics_across_batches_by_title(
            batch_topics=all_batch_topics,
            global_id_to_index=global_id_to_index,
            chat_public_id=chat_public_id,
        )

    topics = summarize.rebalance_first_message_links(
        topics=topics,
        global_id_to_index=global_id_to_index,
        chat_public_id=chat_public_id,
    )
    print_title(f"Final Topics ({merge_method})")
    print_json(topics)

    formatted = summarize.format_topics_text(chat_title, topics, args.max_topics)
    print_title("Formatted Digest")
    print(formatted)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run_debug())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
