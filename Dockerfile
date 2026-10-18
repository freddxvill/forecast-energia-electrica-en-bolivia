# API de pronósticos (FastAPI) para Hugging Face Spaces (SDK docker, puerto 7860).
# Solo instala core + grupo `api`: sirve los artefactos precalculados de artifacts/demo/,
# no entrena ni necesita torch.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /bin/uv

# HF Spaces corre el contenedor con el usuario 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group api --no-install-project

COPY --chown=user src ./src
COPY --chown=user artifacts/demo ./artifacts/demo
RUN uv sync --locked --no-default-groups --group api

ENV FORECAST_DEMO_DIR=/home/user/app/artifacts/demo
EXPOSE 7860
CMD ["uvicorn", "forecast_energia.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
