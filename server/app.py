"""FastAPI server entrypoint for SQLDebugEnv."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.environment import SQLDebugEnv
from env.models import Action, Observation, State

app = FastAPI(title="SQLDebugEnv API", version="0.1.0")
environment = SQLDebugEnv()


class ResetRequest(BaseModel):
    """Request body for resetting the environment."""
    task_name: Optional[str] = Field(default="easy")


class StepResponse(BaseModel):
    """Typed response returned by the step endpoint."""
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]


@app.get("/health")
def healthcheck() -> dict[str, str]:
    """Return a simple health status."""
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    """Root endpoint."""
    return {"status": "ok", "env": "SQLDebugEnv"}


@app.post("/reset", response_model=Observation)
def reset_environment(request: Optional[ResetRequest] = None) -> Observation:
    """Reset the environment to the requested task. Body is optional."""
    try:
        task = (request.task_name if request and request.task_name else None) or "easy"
        return environment.reset(task)
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
