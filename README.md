---
title: Sql Debug Env
emoji: 🛠️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# SQLDebugEnv

An OpenEnv-compatible environment for debugging SQL queries against a deterministic seeded SQLite database.

## Quick Start

```bash
pip install -r requirements.txt
python demo.py                          # test all 3 tasks locally
uvicorn server.app:app --port 7860      # start server
```

## Docker

```bash
docker build -t sql-debug-env .
docker run -p 7860:7860 sql-debug-env
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /reset | Reset environment (body: `{"task_name":"easy"}`) |
| POST | /step | Apply action |
| GET | /state | Get current state |

## Tasks

| Task | Difficulty | Bug |
|------|-----------|-----|
| easy | Easy | Misspelled keyword `SELEC` → `SELECT` |
| medium | Medium | Wrong JOIN column `o.id` → `o.user_id` in aggregation |
| hard | Hard | Wrong JOIN column in COUNT + SUM report |

## Inference

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token_here
python inference.py --task easy --max-steps 10
```

## Log Format

```
[START] task=<task> env=SQLDebugEnv model=<model>
[STEP] step=1 action={...} reward=0.00 done=false error=null
[END] success=true steps=2 score=1.000 rewards=0.35,1.35
```
