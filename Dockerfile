FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY summarize.py /app/summarize.py

RUN useradd -m -u 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

USER appuser

CMD ["python", "summarize.py", "--state-file", "/data/bot_state.json"]
