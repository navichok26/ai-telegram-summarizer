# Telegram Group Topic Summarizer (Ollama)

A Telegram bot that reads chats, summarizes **group/supergroup** discussions with Ollama, and posts topic digests back to the same chat.

## Requirements

- Docker + Docker Compose plugin
- Running Ollama instance with model available (default: `qwen3.5:9b`)

## Quick Start (Docker Compose)

1. Create config from template:

```bash
cp .env.example .env
```

2. Set `BOT_TOKEN` in `.env`.

3. Ensure Ollama is running and model is available:

```bash
ollama list
```

4. Start bot:

```bash
docker compose up -d --build
```

5. View logs:

```bash
docker compose logs -f
```

6. Stop:

```bash
docker compose down
```


## What It Does

- Polls Telegram Bot API using `getUpdates` with `timeout=0`.
- Processes only `group` and `supergroup` messages.
  - Ignores private chats and channels.
  - Ignores bot-authored messages.
- Normalizes and filters incoming messages.
  - Drops `via_bot` messages.
  - Drops media-only content without useful text.
  - Drops very short messages (`MIN_CHARS`).
- Merges nearby consecutive messages from the same author.
- Sends merged chunks to Ollama in batches (default: 100 chunks per request).
- Extracts topics with evidence message IDs.
- Semantically merges topic candidates across batches.
- Builds Telegram links to the first message of each topic.
- Posts a formatted digest back to the same chat.
- Pins the digest message **silently** (best effort).
  - If pinning fails (e.g., missing rights), the bot continues.
- Clears processed messages from state after successful post.

## Architecture

Single service, single script: `summarize.py`.

Main runtime stages:

1. `getUpdates` polling
2. Per-chat buffering in local JSON state
3. Trigger check (`min messages` / `cooldown`)
4. Message merge + batched topic extraction via Ollama
5. Cross-batch semantic topic merge
6. Digest post + optional silent pin
7. State persistence

## State and Reliability

State is stored in a JSON file (`/data/bot_state.json` in container), mounted from host (`./state`).

State includes:

- `offset` (last processed Telegram update)
- Per-chat message buffer
- `seen_ids` for deduplication
- `pending_count`
- `last_summary_ts`

Why this minimizes message loss:

- `offset` is persisted after processing update batches.
- State is written atomically (`.tmp` + rename).
- On successful digest post, processed chat messages are removed.

## Topic Link Behavior

Telegram links are built as:

`https://t.me/c/<internal_chat_id_without_-100>/<message_id>`

To reduce repeated identical links across topics, merged-chunk constraints are applied:

- `MERGE_WINDOW_SECONDS`
- `MAX_MERGE_MESSAGES`
- `MAX_MERGE_SPAN_SECONDS`

## Local Run (without Docker)

```bash
python3 summarize.py --state-file state/bot_state.json
```

You can use `.env` for configuration; `BOT_TOKEN` is required.

## Configuration

Environment variables used by Compose (with defaults):

- `BOT_TOKEN` (required)
- `LOG_LEVEL=INFO` (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `OLLAMA_URL=http://host.docker.internal:11434`
- `MODEL=qwen3.5:9b`
- `POLL_INTERVAL_SECONDS=2`
- `MIN_NEW_MESSAGES_FOR_SUMMARY=25`
- `SUMMARY_COOLDOWN_SECONDS=3600`
- `MAX_BUFFER_MESSAGES=800`
- `MIN_CHARS=12`
- `MERGE_WINDOW_SECONDS=180`
- `MAX_MERGE_MESSAGES=3`
- `MAX_MERGE_SPAN_SECONDS=90`
- `MAX_TOPICS=12`
- `BATCH_SIZE=100`
- `MAX_SEND_CHARS=3800`

## Logging and Debugging

Set `LOG_LEVEL=DEBUG` to get detailed timings and counters:

- Telegram API timings
- `getUpdates` latency and batch size
- Update processing stats (added/filtered/duplicates)
- Per-batch Ollama timings
- Merge timings
- Full cycle duration

## Pipeline Debug With Mocks

For step-by-step testing of summarization logic (filters, merges, prompts, raw LLM outputs, and final digest), run:

```bash
python3 scripts/debug_summarization.py \
  --updates-file mocks/telegram_updates.sample.json \
  --ollama-url http://localhost:11434 \
  --model qwen3.5:9b \
  --batch-size 25
```

What this script prints:

- filter trace for each update (`accepted`, `duplicate`, `via_bot`, `short_text`, etc.)
- `process_updates` stats
- normalized messages
- merged chunks
- per-batch prompts and live Ollama outputs
- semantic merge prompt and output
- final topics and formatted digest text

The script always calls live Ollama using `--ollama-url` and `--model`.

## Telegram Permissions Notes

For full message visibility in groups, bot privacy mode may need to be disabled in BotFather.

For pinning digest messages, the bot must have permission to pin messages in that chat.

## Security / Publishing Checklist

Before publishing:

- Do **not** commit `.env`.
- Do **not** commit `state/`.
- Rotate bot token if it was ever exposed.
- Keep secrets only in runtime environment.

This repository already ignores `.env` and `state/` in `.gitignore`.

## Repository Layout

- `summarize.py` — bot runtime and summarization pipeline
- `Dockerfile` — container image
- `docker-compose.yml` — service orchestration
- `.env.example` — configuration template
- `state/` — host-mounted runtime state (ignored by git)
