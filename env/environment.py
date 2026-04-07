"""Environment core for SQLDebugEnv."""
from __future__ import annotations
from typing import Any
from grader.grader import grade_task_result
from tasks.easy import get_task as get_easy_task
from tasks.medium import get_task as get_medium_task
from tasks.hard import get_task as get_hard_task
from .db import execute_query, get_schema, initialize_database
from .models import Action, Observation, State

TASKS = {"easy": get_easy_task, "medium": get_medium_task, "hard": get_hard_task}
RESULT_PREVIEW_LIMIT = 5
STEP_PENALTY = 0.05
SYNTAX_FIX_REWARD = 0.3
QUERY_EXECUTES_REWARD = 0.4
PARTIAL_OUTPUT_REWARD = 0.6
FULL_OUTPUT_REWARD = 1.0
WRONG_SUBMISSION_PENALTY = -0.5

class SQLDebugEnv:
    """OpenEnv-style SQL debugging environment."""

    def __init__(self, default_task: str = "easy") -> None:
        self._default_task = default_task
        self._task: dict[str, Any] = {}
        self._current_sql = ""
        self._error_message: str | None = None
        self._result_preview: list[dict[str, Any]] = []
        self._step_count = 0
        self._done = False
        self._last_score = 0.0
        self.reset(default_task)

    def reset(self, task_name: str | None = None) -> Observation:
        selected_task = task_name or self._default_task
        if selected_task not in TASKS:
            raise ValueError(f"Unknown task: {selected_task!r}. Choose from: {list(TASKS)}")
        initialize_database()
        self._task = TASKS[selected_task]()
        self._current_sql = self._task["buggy_query"]
        self._error_message = None
        self._result_preview = []
        self._step_count = 0
        self._done = False
        self._last_score = 0.0
        return self._build_observation()

    def step(self, action: Action | dict[str, Any]):
        if self._done:
            return self._build_observation(), 0.0, True, {"message": "Episode already completed."}
        parsed = action if isinstance(action, Action) else Action.model_validate(action)
        prev_error = self._error_message
        self._step_count += 1
        reward = -STEP_PENALTY
        info: dict[str, Any] = {"task_name": self._task["name"], "action_type": parsed.action_type, "score": 0.0, "execution_error": None}

        if parsed.action_type == "edit_query":
            self._current_sql = parsed.new_sql or self._current_sql
            self._error_message = None
            self._result_preview = []
        else:
            result = execute_query(self._current_sql)
            if isinstance(result, str):
                self._error_message = result
                self._result_preview = []
                info["execution_error"] = result
                score = 0.0
            else:
                self._error_message = None
                self._result_preview = result[:RESULT_PREVIEW_LIMIT]
                score = grade_task_result(result, self._task)
                info["score"] = score
                reward += QUERY_EXECUTES_REWARD
                if prev_error and not self._error_message:
                    reward += SYNTAX_FIX_REWARD
                if score == 1.0:
                    reward += FULL_OUTPUT_REWARD
                elif score > 0.0:
                    reward += PARTIAL_OUTPUT_REWARD * score
                self._last_score = score

        if parsed.action_type == "submit":
            self._done = True
            if self._last_score < 1.0:
                reward += WRONG_SUBMISSION_PENALTY
            info["submitted"] = True

        info["done"] = self._done
        return self._build_observation(), round(reward, 4), self._done, info

    def state(self) -> State:
        return State(
            task_name=self._task.get("name",""),
            current_sql=self._current_sql,
            error_message=self._error_message,
            result_preview=self._result_preview,
            database_schema=get_schema(),
            step_count=self._step_count,
            done=self._done,
        )

    def _build_observation(self) -> Observation:
        return Observation(
            current_sql=self._current_sql,
            error_message=self._error_message,
            result_preview=self._result_preview,
            database_schema=get_schema(),
            step_count=self._step_count,
        )
