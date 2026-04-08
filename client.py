"""client.py – HTTP + WebSocket client for SQLDebugEnv."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import requests
import websockets

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:7860")
WS_URL   = os.getenv(
    "OPENENV_WS_URL",
    BASE_URL.replace("http://", "ws://").replace("https://", "wss://") + "/ws",
)


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def reset(task_name: str = "easy") -> dict[str, Any]:
    r = requests.post(f"{BASE_URL}/reset", json={"task_name": task_name}, timeout=30)
    r.raise_for_status()
    return r.json()


def step(
    action_type: str = "submit",
    new_sql: str | None = None,
    task: str = "easy",
) -> dict[str, Any]:
    r = requests.post(
        f"{BASE_URL}/step",
        json={"action_type": action_type, "new_sql": new_sql, "task": task},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def state() -> dict[str, Any]:
    r = requests.get(f"{BASE_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()


# ── WebSocket demo ─────────────────────────────────────────────────────────────

async def websocket_demo(task_name: str = "easy") -> None:
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"type": "reset", "task_name": task_name}))
        print("RESET :", await ws.recv())

        await ws.send(json.dumps({"type": "step", "action_type": "run_query"}))
        print("STEP  :", await ws.recv())

        await ws.send(json.dumps({"type": "state"}))
        print("STATE :", await ws.recv())

        await ws.send(json.dumps({"type": "close"}))


if __name__ == "__main__":
    print("=== HTTP demo ===")
    try:
        print(reset())
        print(state())
    except Exception as exc:
        print(f"HTTP demo failed: {exc}")

    print("\n=== WebSocket demo ===")
    try:
        asyncio.run(websocket_demo())
    except Exception as exc:
        print(f"WebSocket demo failed: {exc}")
