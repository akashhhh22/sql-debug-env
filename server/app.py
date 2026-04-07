"""FastAPI server for SQLDebugEnv."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from env.environment import SQLDebugEnv
from env.models import Action, Observation, State

app = FastAPI(title="SQLDebugEnv API", version="0.1.0")
environment = SQLDebugEnv()

class ResetRequest(BaseModel):
    task_name: str = Field(default="easy")

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]

@app.get("/health")
def healthcheck():
    return {"status": "ok", "env": "SQLDebugEnv", "version": "0.1.0"}

@app.post("/reset", response_model=Observation)
def reset_environment(request: ResetRequest):
    try:
        return environment.reset(request.task_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResponse)
def step_environment(action: Action):
    obs, reward, done, info = environment.step(action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=State)
def get_state():
    return environment.state()
