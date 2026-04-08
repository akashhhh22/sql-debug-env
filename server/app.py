"""server/app.py – FastAPI + WebSocket server for SQLDebugEnv."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from models import Action, SQLDebugEnv

app = FastAPI(title="SQLDebugEnv OpenEnv")


class StepRequest(BaseModel):
    action_type: Optional[str] = "submit"
    new_sql:     Optional[str] = None
    task:        Optional[str] = "easy"


# ── HTTP endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "SQLDebugEnv 🛠️", "tasks": 3, "version": "1.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def http_reset(body: dict = {}):
    task_name = (body or {}).get("task_name", "easy")
    env = SQLDebugEnv(default_task=task_name)
    obs = env.reset(task_name)
    return JSONResponse(content={"observation": obs.model_dump(), "done": False, "reward": 0.0})


@app.post("/step")
async def http_step(request: StepRequest):
    task = request.task or "easy"
    env  = SQLDebugEnv(default_task=task)
    env.reset(task)
    action = Action(action_type=request.action_type, new_sql=request.new_sql)
    obs, reward, done, info = env.step(action)
    return JSONResponse(
        content={"observation": obs.model_dump(), "reward": reward, "done": done, "info": info}
    )


@app.get("/state")
async def http_state():
    env = SQLDebugEnv()
    s   = env.state()
    return JSONResponse(content=s.model_dump())


# ── WebSocket endpoint ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env = SQLDebugEnv()
    try:
        while True:
            data     = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "reset":
                task_name = data.get("task_name", "easy")
                obs = env.reset(task_name)
                await websocket.send_json(
                    {"type": "reset", "observation": obs.model_dump(), "done": False, "reward": 0.0}
                )

            elif msg_type == "step":
                action = Action(
                    action_type=data.get("action_type", "submit"),
                    new_sql=data.get("new_sql"),
                )
                obs, reward, done, info = env.step(action)
                await websocket.send_json(
                    {
                        "type":        "step",
                        "observation": obs.model_dump(),
                        "reward":      reward,
                        "done":        done,
                        "info":        info,
                    }
                )

            elif msg_type == "state":
                await websocket.send_json({"type": "state", "state": env.state().model_dump()})

            elif msg_type == "close":
                break

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        pass
    finally:
        env.close()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    uvicorn.run(
        "server.app:app",
        host=os.getenv("HOST",    "0.0.0.0"),
        port=int(os.getenv("PORT",    "7860")),
        workers=int(os.getenv("WORKERS", "1")),
    )


if __name__ == "__main__":
    main()
