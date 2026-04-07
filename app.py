"""FastAPI server entrypoint for SQLDebugEnv."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.environment import SQLDebugEnv
from env.models import Action, Observation, State
from tasks.easy   import EASY_TASK
from tasks.medium import MEDIUM_TASK
from tasks.hard   import HARD_TASK

app = FastAPI(title="SQLDebugEnv API", version="0.1.0")
environment = SQLDebugEnv()

TASK_REGISTRY = {
    "easy":   EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard":   HARD_TASK,
}

class ResetRequest(BaseModel):
    """Request body for resetting the environment."""
    task_name: str = Field(default="easy")

class StepResponse(BaseModel):
    """Typed response returned by the step endpoint."""
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]

@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Return a simple health status."""
    return {"status": "ok", "env": "SQLDebugEnv", "version": "0.1.0"}

@app.get("/tasks")
def list_tasks() -> list[dict[str, Any]]:
    """List all available tasks with metadata."""
    return [
        {
            "id":          name,
            "difficulty":  task.get("difficulty", name),
            "max_steps":   task.get("max_steps", 10),
            "description": task.get("description", ""),
        }
        for name, task in TASK_REGISTRY.items()
    ]

@app.post("/reset", response_model=Observation)
def reset_environment(request: ResetRequest) -> Observation:
    """Reset the environment to the requested task."""
    try:
        return environment.reset(request.task_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.post("/step", response_model=StepResponse)
def step_environment(action: Action) -> StepResponse:
    """Apply an action to the environment."""
    observation, reward, done, info = environment.step(action)
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )

@app.get("/state", response_model=State)
def get_state() -> State:
    """Return the current environment state."""
    return environment.state()
