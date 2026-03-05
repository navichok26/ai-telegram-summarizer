#!/usr/bin/env python3
"""Poll Telegram getUpdates, summarize group chats with Ollama, and post topics back."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

LOGGER = logging.getLogger("tg_summarizer")


@dataclass
class NormalizedMessage:
    id: int
    from_name: str
    date: datetime
    text: str


@dataclass
class MergedChunk:
    message_ids: list[int]
    first_message_id: int
    from_name: str
    text: str


@dataclass
class ProcessUpdatesStats:
    total_updates: int = 0
    message_updates: int = 0
    skipped_non_message: int = 0
    skipped_non_group: int = 0
    skipped_bot_sender: int = 0
    normalized_added: int = 0
    duplicates: int = 0
    filtered_out: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Telegram group topic summarizer via Ollama (polling getUpdates)."
    )
    parser.add_argument(
        "--bot-token",
        default=None,
        help="Telegram bot token (default: BOT_TOKEN env or .env)",
    )
    parser.add_argument(
        "--state-file",
        default="bot_state.json",
        help="Path to runtime state with offset and chat buffers (default: bot_state.json)",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=2.0,
        help="Interval between getUpdates calls with timeout=0 (default: 2)",
    )
    parser.add_argument(
        "--min-new-messages-for-summary",
        type=int,
        default=25,
        help="Auto-trigger summary when this many new messages accumulated (default: 25)",
    )
    parser.add_argument(
        "--summary-cooldown-seconds",
        type=int,
        default=300,
        help="Minimum seconds between auto summaries per chat (default: 300)",
    )
    parser.add_argument(
        "--max-buffer-messages",
        type=int,
        default=800,
        help="How many normalized messages to keep per chat in memory/state (default: 800)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=12,
        help="Ignore messages shorter than this many characters after cleanup (default: 12)",
    )
    parser.add_argument(
        "--merge-window-seconds",
        type=int,
        default=180,
        help=(
            "Merge consecutive messages from the same author if gap is within this many seconds "
            "(default: 180)"
        ),
    )
    parser.add_argument(
        "--max-merge-messages",
        type=int,
        default=3,
        help="Maximum source messages to merge into one chunk (default: 3)",
    )
    parser.add_argument(
        "--max-merge-span-seconds",
        type=int,
        default=90,
        help="Maximum timespan of one merged chunk in seconds (default: 90)",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:9b",
        help="Ollama model name (default: qwen3.5:9b)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for Ollama HTTP API (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=12,
        help="Maximum number of topics per LLM response (default: 12)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="How many merged messages to send to Ollama per request (default: 100)",
    )
    parser.add_argument(
        "--max-send-chars",
        type=int,
        default=3800,
        help="Max chars per Telegram outbound message chunk (default: 3800)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    squashed = "\n".join(line for line in lines if line)
    return " ".join(squashed.split()) if "\n" not in squashed else squashed


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"offset": 0, "chats": {}}

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid state JSON: {state_path}") from exc

    if not isinstance(state, dict):
        return {"offset": 0, "chats": {}}

    offset = state.get("offset")
    chats = state.get("chats")
    if not isinstance(offset, int):
        offset = 0
    if not isinstance(chats, dict):
        chats = {}

    return {"offset": offset, "chats": chats}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    started = time.perf_counter()
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(state_path)
    LOGGER.debug("state_saved path=%s elapsed_ms=%.1f", state_path, (time.perf_counter() - started) * 1000)


def telegram_api(bot_token: str, method: str, payload: dict[str, Any]) -> Any:
    started = time.perf_counter()
    url = f"https://api.telegram.org/bot{bot_token}/{method}"
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram HTTP error {exc.code}: {error_body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Telegram API unavailable: {exc.reason}") from exc

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Telegram API returned non-JSON response.") from exc

    if not isinstance(envelope, dict):
        raise RuntimeError("Telegram API response is not a JSON object.")

    if not envelope.get("ok"):
        raise RuntimeError(f"Telegram API error: {envelope}")

    # LOGGER.debug(
    #     "telegram_api method=%s elapsed_ms=%.1f",
    #     method,
    #     (time.perf_counter() - started) * 1000,
    # )
    return envelope.get("result")


def get_updates(bot_token: str, offset: int) -> list[dict[str, Any]]:
    started = time.perf_counter()
    result = telegram_api(
        bot_token,
        "getUpdates",
        {
            "offset": offset,
            "limit": 100,
            "timeout": 0,
            "allowed_updates": ["message"],
        },
    )
    if not isinstance(result, list):
        # LOGGER.debug(
        #     "get_updates offset=%s updates=0 elapsed_ms=%.1f",
        #     offset,
        #     (time.perf_counter() - started) * 1000,
        # )
        return []
    updates = [item for item in result if isinstance(item, dict)]
    # LOGGER.debug(
    #     "get_updates offset=%s updates=%s elapsed_ms=%.1f",
    #     offset,
    #     len(updates),
    #     (time.perf_counter() - started) * 1000,
    # )
    return updates


def split_for_telegram(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        split_idx = remaining.rfind("\n", 0, max_chars)
        if split_idx < max_chars // 2:
            split_idx = max_chars
        chunks.append(remaining[:split_idx].rstrip())
        remaining = remaining[split_idx:].lstrip("\n")

    if remaining:
        chunks.append(remaining)
    return chunks


def send_message(bot_token: str, chat_id: int, text: str, max_send_chars: int) -> int | None:
    chunks = split_for_telegram(text, max_send_chars)
    started = time.perf_counter()
    first_message_id: int | None = None
    for chunk in chunks:
        result = telegram_api(
            bot_token,
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
                "parse_mode": "HTML",
            },
        )
        if first_message_id is None and isinstance(result, dict):
            message_id = result.get("message_id")
            if isinstance(message_id, int):
                first_message_id = message_id

    LOGGER.info(
        "send_message chat_id=%s chunks=%s chars=%s elapsed_ms=%.1f",
        chat_id,
        len(chunks),
        len(text),
        (time.perf_counter() - started) * 1000,
    )
    return first_message_id


def pin_chat_message_silent(bot_token: str, chat_id: int, message_id: int) -> None:
    try:
        telegram_api(
            bot_token,
            "pinChatMessage",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "disable_notification": True,
            },
        )
        LOGGER.info("pin_message chat_id=%s message_id=%s status=ok", chat_id, message_id)
    except RuntimeError as exc:
        # Pinning is best-effort: for missing rights or any Telegram-side failure we continue normally.
        LOGGER.debug(
            "pin_message chat_id=%s message_id=%s status=skipped error=%s",
            chat_id,
            message_id,
            exc,
        )


def extract_sender_name(message: dict[str, Any]) -> str:
    sender = message.get("from")
    if not isinstance(sender, dict):
        return "Unknown"

    first_name = sender.get("first_name")
    last_name = sender.get("last_name")
    username = sender.get("username")

    parts: list[str] = []
    if isinstance(first_name, str) and first_name.strip():
        parts.append(first_name.strip())
    if isinstance(last_name, str) and last_name.strip():
        parts.append(last_name.strip())

    if parts:
        return " ".join(parts)
    if isinstance(username, str) and username.strip():
        return f"@{username.strip()}"
    return "Unknown"


def extract_text_from_live_message(message: dict[str, Any]) -> str:
    text = message.get("text")
    if isinstance(text, str):
        return normalize_whitespace(text)

    caption = message.get("caption")
    if isinstance(caption, str):
        return normalize_whitespace(caption)

    return ""


def is_media_message(message: dict[str, Any]) -> bool:
    media_keys = (
        "photo",
        "video",
        "document",
        "sticker",
        "animation",
        "voice",
        "audio",
        "video_note",
        "contact",
        "location",
        "venue",
        "poll",
    )
    return any(key in message for key in media_keys)


def normalize_live_message(message: dict[str, Any], min_chars: int) -> NormalizedMessage | None:
    if message.get("via_bot") is not None or message.get("via_bot_id") is not None:
        return None

    text = extract_text_from_live_message(message)
    if text == "" and is_media_message(message):
        return None
    if len(text) < min_chars:
        return None

    message_id = message.get("message_id")
    date_ts = message.get("date")
    if not isinstance(message_id, int) or not isinstance(date_ts, int):
        return None

    try:
        date = datetime.fromtimestamp(date_ts, tz=timezone.utc)
    except (OSError, OverflowError, ValueError):
        return None

    return NormalizedMessage(
        id=message_id,
        from_name=extract_sender_name(message),
        date=date,
        text=text,
    )


def merge_consecutive_messages(
    messages: list[NormalizedMessage],
    merge_window_seconds: int,
    max_merge_messages: int,
    max_merge_span_seconds: int,
) -> list[MergedChunk]:
    if not messages:
        return []

    if max_merge_messages < 1:
        raise RuntimeError("--max-merge-messages must be >= 1")
    if max_merge_span_seconds < 0:
        raise RuntimeError("--max-merge-span-seconds must be >= 0")

    ordered = sorted(messages, key=lambda item: item.date)
    merged: list[MergedChunk] = []

    current = MergedChunk(
        message_ids=[ordered[0].id],
        first_message_id=ordered[0].id,
        from_name=ordered[0].from_name,
        text=ordered[0].text,
    )
    current_start_dt = ordered[0].date
    last_dt = ordered[0].date

    for msg in ordered[1:]:
        same_author = msg.from_name == current.from_name
        seconds_since_last = (msg.date - last_dt).total_seconds()
        close_enough = seconds_since_last <= merge_window_seconds
        within_count_limit = len(current.message_ids) < max_merge_messages
        span_seconds = (msg.date - current_start_dt).total_seconds()
        within_span_limit = span_seconds <= max_merge_span_seconds

        if same_author and close_enough and within_count_limit and within_span_limit:
            current.message_ids.append(msg.id)
            current.text += f"\n{msg.text}"
        else:
            merged.append(current)
            current = MergedChunk(
                message_ids=[msg.id],
                first_message_id=msg.id,
                from_name=msg.from_name,
                text=msg.text,
            )
            current_start_dt = msg.date

        last_dt = msg.date

    merged.append(current)
    return merged


def build_llm_chat_payload(merged_messages: list[MergedChunk]) -> dict[str, Any]:
    return {
        "messages": [
            {
                "id": item.first_message_id,
                "from": item.from_name,
                "text": item.text,
            }
            for item in merged_messages
        ]
    }


def chunk_merged_messages(merged_messages: list[MergedChunk], batch_size: int) -> list[list[MergedChunk]]:
    if batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")
    return [merged_messages[i : i + batch_size] for i in range(0, len(merged_messages), batch_size)]


def build_schema(max_topics: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "maxItems": max_topics,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "evidence_message_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": 10,
                        },
                    },
                    "required": ["title", "summary", "evidence_message_ids"],
                },
            }
        },
        "required": ["topics"],
    }


def build_prompt(chat_payload: dict[str, Any], max_topics: int) -> str:
    min_topics = min(5, max_topics)
    chat_json = json.dumps(chat_payload, ensure_ascii=False, indent=2)
    return (
        "Ты анализируешь JSON с сообщениями из Telegram-чата и выделяешь персонализированные темы про людей.\n"
        "Стиль формулировок: очень простой разговорный русский, ироничный, вайбовый, молодёжный и прикольный.\n"
        "Пиши легко и по-человечески: короткие фразы, без канцелярита и тяжёлых конструкций.\n"
        "Допустима лёгкая ирония и сленг, но без перегиба и без кринжа.\n"
        f"Верни {min_topics}-{max_topics} тем, без микротем и дублей.\n"
        "Каждая тема должна быть вокруг конкретных участников из messages[].from: кто что предложил, сделал, спросил или отстаивал.\n"
        "Заголовок должен явно содержать имя/ник человека и его действие; избегай абстракций вроде 'Обсуждение проекта' или 'Планы на будущее'.\n"
        "В summary обязательно называй людей и их позиции/действия, а не пересказывай безлично.\n"
        "Добавляй контекста словам, какой проект обсуждался, какая проблема была решена.\n"
        "Если контекста в обсуждении нет, то описывай примерный контекст на свое усмотрение в котором эти слова могли звучать.\n"
        "Объединяй одинаковые и очень близкие по смыслу темы.\n"
        "По возможности выбирай темы, где участвовали 2+ человека.\n"
        "КРИТИЧНО: evidence_message_ids должны содержать только id из messages[].id.\n"
        "Отвечай строго JSON по заданной схеме.\n\n"
        "Входной JSON:\n"
        f"{chat_json}"
    )


def call_ollama_generate(
    ollama_url: str,
    model: str,
    prompt: str,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    started = time.perf_counter()
    api_url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1},
        "format": schema,
    }
    request = urllib_request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(request, timeout=600) as response:
            response_body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP error {exc.code}: {body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {api_url}: {exc.reason}") from exc

    try:
        envelope = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned non-JSON envelope.") from exc

    if not isinstance(envelope, dict):
        raise RuntimeError("Ollama envelope is not a JSON object.")
    if envelope.get("error"):
        raise RuntimeError(f"Ollama error: {envelope['error']}")

    raw_response = envelope.get("response")
    if not isinstance(raw_response, str):
        raise RuntimeError("Ollama response is missing string field 'response'.")

    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Model response is not valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Model JSON response root must be an object.")

    LOGGER.debug(
        "ollama_generate model=%s prompt_chars=%s response_chars=%s elapsed_ms=%.1f",
        model,
        len(prompt),
        len(raw_response),
        (time.perf_counter() - started) * 1000,
    )
    return parsed, raw_response


def normalize_topic_key(title: str) -> str:
    return " ".join(title.split()).casefold()


def to_public_chat_id(chat_id: int) -> str:
    raw = str(chat_id)
    if raw.startswith("-100"):
        raw = raw[4:]
    elif raw.startswith("-"):
        raw = raw[1:]

    if not raw.isdigit():
        raise RuntimeError(f"Cannot derive public chat id from '{chat_id}'.")
    return raw


def validate_and_enrich_topics(
    raw_topics_object: dict[str, Any],
    allowed_ids: list[int],
    chat_public_id: str,
    message_id_to_author: dict[int, str],
) -> list[dict[str, Any]]:
    raw_topics = raw_topics_object.get("topics")
    if not isinstance(raw_topics, list):
        raise RuntimeError("Model JSON does not contain array field 'topics'.")

    allowed_set = set(allowed_ids)
    id_to_index = {message_id: idx for idx, message_id in enumerate(allowed_ids)}
    seen_titles: set[str] = set()
    topics: list[dict[str, Any]] = []

    for item in raw_topics:
        if not isinstance(item, dict):
            continue

        title = item.get("title")
        summary = item.get("summary")
        evidence_ids = item.get("evidence_message_ids")
        if not isinstance(title, str) or not isinstance(summary, str) or not isinstance(evidence_ids, list):
            continue

        title = " ".join(title.split())
        summary = " ".join(summary.split())
        if not title or not summary:
            continue

        topic_key = normalize_topic_key(title)
        if topic_key in seen_titles:
            continue

        filtered_evidence: list[int] = []
        for evidence_id in evidence_ids:
            if isinstance(evidence_id, int) and evidence_id in allowed_set and evidence_id not in filtered_evidence:
                filtered_evidence.append(evidence_id)

        if not filtered_evidence:
            continue

        participants: list[str] = []
        for evidence_id in sorted(filtered_evidence, key=lambda message_id: id_to_index[message_id]):
            author = message_id_to_author.get(evidence_id)
            if author and author not in participants:
                participants.append(author)

        first_message_id = min(filtered_evidence, key=lambda message_id: id_to_index[message_id])
        topics.append(
            {
                "title": title,
                "summary": summary,
                "evidence_message_ids": filtered_evidence,
                "participants": participants,
                "first_message_id": first_message_id,
                "first_message_link": f"https://t.me/c/{chat_public_id}/{first_message_id}",
            }
        )
        seen_titles.add(topic_key)

    topics.sort(key=lambda topic: id_to_index[topic["first_message_id"]])
    return topics


def merge_topics_across_batches_by_title(
    batch_topics: list[dict[str, Any]],
    global_id_to_index: dict[int, int],
    chat_public_id: str,
) -> list[dict[str, Any]]:
    merged_by_key: dict[str, dict[str, Any]] = {}

    for topic in batch_topics:
        key = normalize_topic_key(topic["title"])
        if key not in merged_by_key:
            merged_by_key[key] = {
                "title": topic["title"],
                "summary": topic["summary"],
                "evidence_message_ids": list(topic["evidence_message_ids"]),
                "participants": list(topic["participants"]),
            }
            continue

        current = merged_by_key[key]
        if len(topic["summary"]) > len(current["summary"]):
            current["summary"] = topic["summary"]

        for evidence_id in topic["evidence_message_ids"]:
            if evidence_id not in current["evidence_message_ids"]:
                current["evidence_message_ids"].append(evidence_id)

        for participant in topic["participants"]:
            if participant not in current["participants"]:
                current["participants"].append(participant)

    merged_topics: list[dict[str, Any]] = []
    for topic in merged_by_key.values():
        valid_evidence = [
            evidence_id
            for evidence_id in topic["evidence_message_ids"]
            if evidence_id in global_id_to_index
        ]
        if not valid_evidence:
            continue

        valid_evidence.sort(key=lambda message_id: global_id_to_index[message_id])
        first_message_id = valid_evidence[0]
        merged_topics.append(
            {
                "title": topic["title"],
                "summary": topic["summary"],
                "evidence_message_ids": valid_evidence,
                "participants": topic["participants"],
                "first_message_id": first_message_id,
                "first_message_link": f"https://t.me/c/{chat_public_id}/{first_message_id}",
            }
        )

    merged_topics.sort(key=lambda topic: global_id_to_index[topic["first_message_id"]])
    return merged_topics


def rebalance_first_message_links(
    topics: list[dict[str, Any]],
    global_id_to_index: dict[int, int],
    chat_public_id: str,
) -> list[dict[str, Any]]:
    used_first_ids: set[int] = set()

    for topic in topics:
        evidence_ids = topic.get("evidence_message_ids")
        if not isinstance(evidence_ids, list):
            continue

        candidates = [
            evidence_id
            for evidence_id in evidence_ids
            if isinstance(evidence_id, int) and evidence_id in global_id_to_index
        ]
        if not candidates:
            continue

        candidates.sort(key=lambda message_id: global_id_to_index[message_id])
        chosen = candidates[0]
        for candidate in candidates:
            if candidate not in used_first_ids:
                chosen = candidate
                break

        used_first_ids.add(chosen)
        topic["first_message_id"] = chosen
        topic["first_message_link"] = f"https://t.me/c/{chat_public_id}/{chosen}"

    topics.sort(key=lambda topic: global_id_to_index.get(topic.get("first_message_id"), 10**12))
    return topics


def build_topic_merge_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "merged_topics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "source_topic_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                        },
                    },
                    "required": ["title", "summary", "source_topic_indices"],
                },
            }
        },
        "required": ["merged_topics"],
    }


def build_topic_merge_prompt(batch_topics: list[dict[str, Any]]) -> str:
    indexed_topics: list[dict[str, Any]] = []
    for idx, topic in enumerate(batch_topics, start=1):
        indexed_topics.append(
            {
                "index": idx,
                "title": topic["title"],
                "summary": topic["summary"],
                "participants": topic["participants"],
                "first_message_id": topic["first_message_id"],
            }
        )

    topics_json = json.dumps({"topics": indexed_topics}, ensure_ascii=False, indent=2)
    return (
        "Слей кандидаты тем из разных батчей в единый список по смыслу.\n"
        "Стиль формулировок: простой разговорный русский, ироничный, вайбовый и молодёжный, но читабельный.\n"
        "Правила:\n"
        "- Сохраняй персонализацию: итоговые title и summary должны явно называть людей и их действия.\n"
        "- Объединяй дубли и почти одинаковые темы, даже если формулировки разные.\n"
        "- Не склеивай темы только по общей абстрактной теме, если в них разные ключевые люди или разные роли людей.\n"
        "- Разные по смыслу темы не склеивай.\n"
        "- source_topic_indices должны содержать только индексы из входного списка.\n"
        "- Отвечай строго JSON по заданной схеме.\n\n"
        "Входные кандидаты:\n"
        f"{topics_json}"
    )


def merge_topics_across_batches_semantic(
    batch_topics: list[dict[str, Any]],
    global_id_to_index: dict[int, int],
    chat_public_id: str,
    ollama_url: str,
    model: str,
) -> list[dict[str, Any]]:
    if not batch_topics:
        return []

    raw_merged_object, _ = call_ollama_generate(
        ollama_url=ollama_url,
        model=model,
        prompt=build_topic_merge_prompt(batch_topics),
        schema=build_topic_merge_schema(),
    )
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

        valid_evidence = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id in global_id_to_index
        ]
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

    return merge_topics_across_batches_by_title(
        batch_topics=merged_intermediate,
        global_id_to_index=global_id_to_index,
        chat_public_id=chat_public_id,
    )


def deserialize_chat_messages(chat_state: dict[str, Any]) -> list[NormalizedMessage]:
    out: list[NormalizedMessage] = []
    for row in chat_state.get("messages", []):
        if not isinstance(row, dict):
            continue

        message_id = row.get("id")
        from_name = row.get("from_name")
        date_iso = row.get("date")
        text = row.get("text")

        if not isinstance(message_id, int) or not isinstance(from_name, str) or not isinstance(date_iso, str):
            continue
        if not isinstance(text, str) or not text:
            continue

        try:
            date = datetime.fromisoformat(date_iso)
        except ValueError:
            continue

        out.append(
            NormalizedMessage(
                id=message_id,
                from_name=from_name,
                date=date,
                text=text,
            )
        )

    out.sort(key=lambda msg: msg.date)
    return out


def summarize_chat_messages(
    messages: list[NormalizedMessage],
    chat_id: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_started = time.perf_counter()
    merge_started = time.perf_counter()
    merged_messages = merge_consecutive_messages(
        messages,
        args.merge_window_seconds,
        args.max_merge_messages,
        args.max_merge_span_seconds,
    )
    merge_elapsed_ms = (time.perf_counter() - merge_started) * 1000
    if not merged_messages:
        return [], {"chunks": 0, "batches": 0, "topics_before_merge": 0, "merge_method": "none"}

    chat_public_id = to_public_chat_id(chat_id)
    global_id_to_index = {chunk.first_message_id: idx for idx, chunk in enumerate(merged_messages)}
    batches = chunk_merged_messages(merged_messages, args.batch_size)
    schema = build_schema(args.max_topics)

    all_batch_topics: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches, start=1):
        batch_started = time.perf_counter()
        allowed_ids = [chunk.first_message_id for chunk in batch]
        message_id_to_author = {chunk.first_message_id: chunk.from_name for chunk in batch}
        llm_chat_payload = build_llm_chat_payload(batch)
        prompt = build_prompt(chat_payload=llm_chat_payload, max_topics=args.max_topics)

        raw_topics_object, _ = call_ollama_generate(
            ollama_url=args.ollama_url,
            model=args.model,
            prompt=prompt,
            schema=schema,
        )
        batch_topics = validate_and_enrich_topics(
            raw_topics_object=raw_topics_object,
            allowed_ids=allowed_ids,
            chat_public_id=chat_public_id,
            message_id_to_author=message_id_to_author,
        )
        all_batch_topics.extend(batch_topics)
        LOGGER.debug(
            "summarize_batch chat_id=%s batch=%s/%s merged_msgs=%s topics=%s elapsed_ms=%.1f",
            chat_id,
            batch_index,
            len(batches),
            len(batch),
            len(batch_topics),
            (time.perf_counter() - batch_started) * 1000,
        )

    merge_method = "semantic_llm"
    try:
        semantic_started = time.perf_counter()
        topics = merge_topics_across_batches_semantic(
            batch_topics=all_batch_topics,
            global_id_to_index=global_id_to_index,
            chat_public_id=chat_public_id,
            ollama_url=args.ollama_url,
            model=args.model,
        )
        LOGGER.debug(
            "semantic_topic_merge chat_id=%s source_topics=%s result_topics=%s elapsed_ms=%.1f",
            chat_id,
            len(all_batch_topics),
            len(topics),
            (time.perf_counter() - semantic_started) * 1000,
        )
    except RuntimeError:
        merge_method = "title_fallback"
        fallback_started = time.perf_counter()
        topics = merge_topics_across_batches_by_title(
            batch_topics=all_batch_topics,
            global_id_to_index=global_id_to_index,
            chat_public_id=chat_public_id,
        )
        LOGGER.warning(
            "semantic_merge_failed_fallback_title chat_id=%s source_topics=%s result_topics=%s elapsed_ms=%.1f",
            chat_id,
            len(all_batch_topics),
            len(topics),
            (time.perf_counter() - fallback_started) * 1000,
        )

    topics = rebalance_first_message_links(
        topics=topics,
        global_id_to_index=global_id_to_index,
        chat_public_id=chat_public_id,
    )

    meta = {
        "chunks": len(merged_messages),
        "batches": len(batches),
        "topics_before_merge": len(all_batch_topics),
        "merge_method": merge_method,
        "merge_elapsed_ms": round(merge_elapsed_ms, 1),
        "total_elapsed_ms": round((time.perf_counter() - total_started) * 1000, 1),
    }
    return topics, meta


def format_topics_text(chat_title: str, topics: list[dict[str, Any]], max_topics: int) -> str:
    selected = topics[:max_topics]
    if not selected:
        return "Пока нет валидных топиков для этого чата."

    lines = [f"<b>#Дайджест сообщений в чатике!</b>", ""]
    for idx, topic in enumerate(selected, start=1):
        lines.append(f"<b>{idx}. {topic['title']}</b> (<i><a href='{topic['first_message_link']}'>линк</a></i>)")
        lines.append(topic['summary'])
        lines.append("")

    return "\n".join(lines)


def ensure_chat_state(chats: dict[str, Any], chat: dict[str, Any]) -> dict[str, Any]:
    chat_id = chat.get("id")
    if not isinstance(chat_id, int):
        raise RuntimeError("Invalid chat id in update.")

    key = str(chat_id)
    existing = chats.get(key)
    if isinstance(existing, dict):
        return existing

    title = chat.get("title")
    if not isinstance(title, str) or not title.strip():
        title = key

    state = {
        "chat_id": chat_id,
        "chat_title": title,
        "chat_type": chat.get("type", "unknown"),
        "messages": [],
        "seen_ids": [],
        "pending_count": 0,
        "last_summary_ts": 0.0,
    }
    chats[key] = state
    return state


def append_message_to_chat_state(
    chat_state: dict[str, Any],
    normalized: NormalizedMessage,
    max_buffer_messages: int,
) -> bool:
    seen_ids = chat_state.get("seen_ids")
    messages = chat_state.get("messages")
    if not isinstance(seen_ids, list):
        seen_ids = []
        chat_state["seen_ids"] = seen_ids
    if not isinstance(messages, list):
        messages = []
        chat_state["messages"] = messages

    if normalized.id in seen_ids:
        return False

    messages.append(
        {
            "id": normalized.id,
            "from_name": normalized.from_name,
            "date": normalized.date.isoformat(),
            "text": normalized.text,
        }
    )
    seen_ids.append(normalized.id)

    pending_count = chat_state.get("pending_count")
    if not isinstance(pending_count, int):
        pending_count = 0
    chat_state["pending_count"] = pending_count + 1

    overflow = len(messages) - max_buffer_messages
    if overflow > 0:
        del messages[:overflow]
        chat_state["seen_ids"] = [
            row.get("id")
            for row in messages
            if isinstance(row, dict) and isinstance(row.get("id"), int)
        ]
    elif len(seen_ids) > max_buffer_messages * 2:
        chat_state["seen_ids"] = seen_ids[-max_buffer_messages * 2 :]

    return True


def process_updates(
    state: dict[str, Any], updates: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[bool, ProcessUpdatesStats]:
    stats = ProcessUpdatesStats(total_updates=len(updates))
    changed = False
    max_update_id = state.get("offset", 0) - 1

    chats = state.get("chats")
    if not isinstance(chats, dict):
        chats = {}
        state["chats"] = chats

    for update in updates:
        update_id = update.get("update_id")
        if isinstance(update_id, int) and update_id > max_update_id:
            max_update_id = update_id

        message = update.get("message")
        if not isinstance(message, dict):
            stats.skipped_non_message += 1
            continue

        stats.message_updates += 1
        chat = message.get("chat")
        if not isinstance(chat, dict):
            stats.skipped_non_group += 1
            continue

        chat_type = chat.get("type")
        if chat_type not in {"group", "supergroup"}:
            stats.skipped_non_group += 1
            continue

        sender = message.get("from")
        if isinstance(sender, dict) and sender.get("is_bot") is True:
            stats.skipped_bot_sender += 1
            continue

        chat_state = ensure_chat_state(chats, chat)

        normalized = normalize_live_message(message, args.min_chars)
        if normalized is None:
            stats.filtered_out += 1
            continue

        seen_ids = chat_state.get("seen_ids")
        was_duplicate = isinstance(seen_ids, list) and normalized.id in seen_ids
        if append_message_to_chat_state(
            chat_state=chat_state,
            normalized=normalized,
            max_buffer_messages=args.max_buffer_messages,
        ):
            changed = True
            stats.normalized_added += 1
        elif was_duplicate:
            stats.duplicates += 1
        else:
            stats.filtered_out += 1

    if max_update_id >= state.get("offset", 0):
        state["offset"] = max_update_id + 1
        changed = True

    LOGGER.debug(
        "process_updates total=%s message_updates=%s added=%s dup=%s filtered=%s "
        "skip_non_group=%s skip_bot=%s offset=%s",
        stats.total_updates,
        stats.message_updates,
        stats.normalized_added,
        stats.duplicates,
        stats.filtered_out,
        stats.skipped_non_group,
        stats.skipped_bot_sender,
        state.get("offset"),
    )
    return changed, stats


def should_summarize(chat_state: dict[str, Any], args: argparse.Namespace, now_ts: float) -> bool:
    pending_count = chat_state.get("pending_count")
    last_summary_ts = chat_state.get("last_summary_ts")
    if not isinstance(pending_count, int):
        pending_count = 0
    if not isinstance(last_summary_ts, (int, float)):
        last_summary_ts = 0.0

    enough_messages = pending_count >= args.min_new_messages_for_summary
    cooldown_ok = (now_ts - float(last_summary_ts)) >= args.summary_cooldown_seconds
    return enough_messages and cooldown_ok


def run() -> None:
    load_dotenv(Path(".env"))
    args = parse_args()
    setup_logging(args.log_level)

    bot_token = args.bot_token or os.environ.get("BOT_TOKEN")
    if not bot_token:
        raise RuntimeError("BOT_TOKEN is required (pass --bot-token or set BOT_TOKEN in env/.env)")

    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")
    if args.max_topics < 1:
        raise RuntimeError("--max-topics must be >= 1")
    if args.max_buffer_messages < 1:
        raise RuntimeError("--max-buffer-messages must be >= 1")
    if args.max_merge_messages < 1:
        raise RuntimeError("--max-merge-messages must be >= 1")
    if args.max_merge_span_seconds < 0:
        raise RuntimeError("--max-merge-span-seconds must be >= 0")

    state_path = Path(args.state_file)
    state = load_state(state_path)

    LOGGER.info(
        "bot_loop_started poll=%.1fs batch_size=%s model=%s min_new=%s cooldown=%ss "
        "merge_window=%ss merge_max_msgs=%s merge_max_span=%ss state_file=%s",
        args.poll_interval_seconds,
        args.batch_size,
        args.model,
        args.min_new_messages_for_summary,
        args.summary_cooldown_seconds,
        args.merge_window_seconds,
        args.max_merge_messages,
        args.max_merge_span_seconds,
        state_path,
    )
    cycle_index = 0

    while True:
        cycle_index += 1
        cycle_started = time.perf_counter()
        stats = ProcessUpdatesStats()
        try:
            updates = get_updates(bot_token, state.get("offset", 0))
            changed = False
            if updates:
                changed, stats = process_updates(state, updates, args)

            if updates and changed:
                # Persist offset/buffer immediately to minimize risk of losing updates on failures later in loop.
                save_state(state_path, state)

            chats = state.get("chats")
            if not isinstance(chats, dict):
                chats = {}
                state["chats"] = chats

            now_ts = time.time()
            for chat_key, chat_state in chats.items():
                if not isinstance(chat_state, dict):
                    continue
                if not should_summarize(chat_state, args, now_ts):
                    continue

                chat_id = chat_state.get("chat_id")
                chat_title = chat_state.get("chat_title")
                if not isinstance(chat_id, int):
                    continue
                if not isinstance(chat_title, str) or not chat_title.strip():
                    chat_title = str(chat_key)

                messages = deserialize_chat_messages(chat_state)
                if not messages:
                    save_state(state_path, state)
                    continue

                topics, meta = summarize_chat_messages(messages, chat_id, args)
                text = format_topics_text(chat_title, topics, args.max_topics)

                sent_message_id = send_message(bot_token, chat_id, text, args.max_send_chars)
                if isinstance(sent_message_id, int):
                    pin_chat_message_silent(bot_token, chat_id, sent_message_id)

                # Drop successfully processed messages so they are not summarized again.
                chat_state["messages"] = []
                chat_state["seen_ids"] = []
                chat_state["pending_count"] = 0
                chat_state["last_summary_ts"] = time.time()
                save_state(state_path, state)

                LOGGER.info(
                    "summarized chat_id=%s title=%s messages=%s chunks=%s batches=%s "
                    "topics=%s merge=%s merge_elapsed_ms=%s total_elapsed_ms=%s",
                    chat_id,
                    chat_title,
                    len(messages),
                    meta["chunks"],
                    meta["batches"],
                    len(topics),
                    meta["merge_method"],
                    meta.get("merge_elapsed_ms"),
                    meta.get("total_elapsed_ms"),
                )

        except RuntimeError as exc:
            LOGGER.error("runtime_error: %s", exc)
        except Exception:
            LOGGER.exception("unexpected_error_in_cycle")

        # LOGGER.debug(
        #     "cycle_done index=%s updates=%s added=%s elapsed_ms=%.1f",
        #     cycle_index,
        #     stats.total_updates,
        #     stats.normalized_added,
        #     (time.perf_counter() - cycle_started) * 1000,
        # )

        time.sleep(args.poll_interval_seconds)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        LOGGER.info("stopped_by_user")
    except Exception as exc:  # pragma: no cover - CLI top-level failure
        LOGGER.exception("fatal_error: %s", exc)
        sys.exit(1)
