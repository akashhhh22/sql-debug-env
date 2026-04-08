FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY models.py      .
COPY inference.py   .
COPY client.py      .
COPY openenv.yaml   .
COPY server/        server/

EXPOSE 7860

ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""
ENV PORT=7860
ENV HOST=0.0.0.0
ENV WORKERS=1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
