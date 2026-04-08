# SQLDebugEnv 🛠️

An OpenEnv-style reinforcement-learning environment for debugging SQL queries
against a deterministic, seeded SQLite database.

## Project layout

```
sql-debug-env/
├── models.py        ← SQLDebugEnv + Action/Observation/State + DB + tasks + grader
├── inference.py     ← Baseline LLM agent
├── client.py        ← HTTP + WebSocket client helper
├── server/
│   ├── __init__.py
│   └── app.py       ← FastAPI + WebSocket server
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── openenv.yaml
└── result.json
```

## Database schema

| Table    | Columns                                          |
|----------|--------------------------------------------------|
| `users`  | id, name, email, region                          |
| `orders` | id, user_id, product, amount, status             |

## Tasks

| Task     | Bug type             |
|----------|----------------------|
| `easy`   | Syntax error         |
| `medium` | Wrong filter value   |
| `hard`   | Wrong JOIN predicate |

## Action space

```json
{"action_type": "edit_query", "new_sql": "<replacement SQL>"}
{"action_type": "run_query"}
{"action_type": "submit"}
```

## Reward design

| Event                    | Reward  |
|--------------------------|---------|
| Step penalty             | −0.05   |
| Query executes           | +0.40   |
| Syntax error fixed       | +0.30   |
| Partial match            | ×0.60 × score |
| Perfect match            | +1.00   |
| Wrong submission         | −0.50   |

## Quick start

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t sql-debug-env .
docker run --rm -p 7860:7860 \
  -e HF_TOKEN=your_token \
  sql-debug-env
```

## Inference

```bash
export HF_TOKEN=your_token
python inference.py --task easy --max-steps 10
```

Expected log format:

```
[START] task=easy env=SQLDebugEnv model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action={"action_type":"run_query"} reward=-0.05 done=false error=near "SELEC": syntax error
[STEP]  step=2 action={"action_type":"edit_query","new_sql":"SELECT id, name FROM users WHERE region = 'north' ORDER BY id;"} reward=0.65 done=false error=null
[STEP]  step=3 action={"action_type":"submit"} reward=1.35 done=true error=null
[END]   success=true steps=3 score=0.650 rewards=-0.05,0.65,1.35
```

## API

| Method | Path      | Description            |
|--------|-----------|------------------------|
| GET    | /         | Status                 |
| GET    | /health   | Health check           |
| POST   | /reset    | Reset environment      |
| POST   | /step     | Step environment       |
| GET    | /state    | Current state          |
| WS     | /ws       | WebSocket session      |
