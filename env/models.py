"""Typed models for SQLDebugEnv."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

ActionType = Literal["edit_query", "run_query", "submit"]

class Action(BaseModel):
    action_type: ActionType = Field(description="The action to apply.")
    new_sql: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_payload(self):
        if self.action_type == "edit_query" and not self.new_sql:
            raise ValueError("edit_query requires a non-empty new_sql value.")
        if self.action_type != "edit_query" and self.new_sql is not None:
            raise ValueError("new_sql is only allowed for edit_query actions.")
        return self

class Observation(BaseModel):
    current_sql: str
    error_message: str | None = None
    result_preview: list[dict[str, Any]] = Field(default_factory=list)
    database_schema: str
    step_count: int = Field(ge=0)

class State(BaseModel):
    task_name: str
    current_sql: str
    error_message: str | None = None
    result_preview: list[dict[str, Any]] = Field(default_factory=list)
    database_schema: str
    step_count: int = Field(ge=0)
    done: bool = False
