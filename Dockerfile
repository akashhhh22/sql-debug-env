FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy entire repo so env/, grader/, tasks/, server/ are all available
COPY . /app

RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn[standard]==0.34.2 \
    pydantic==2.11.4 \
    "openai>=2.7.2" \
    httpx==0.28.1 \
    python-dotenv \
    requests \
    "openenv-core>=0.2.0"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
