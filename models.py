"""models.py – SQLDebugEnv: flat single-file implementation.

Exports: Action, Observation, Reward, State, SQLDebugEnv
Also exposes helpers: initialize_database, execute_query, get_schema, get_task, grade_result
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ── Pydantic models ────────────────────────────────────────────────────────────

ActionType = Literal["edit_query", "run_query", "submit"]


class Action(BaseModel):
    """Represents a single environment action."""

    action_type: ActionType = Field(description="The action to apply.")
    new_sql: str | None = Field(
        default=None,
        description="Replacement SQL used only for edit_query actions.",
    )

    @model_validator(mode="after")
    def _validate_payload(self) -> "Action":
        if self.action_type == "edit_query" and not self.new_sql:
            raise ValueError("edit_query requires a non-empty new_sql value.")
        if self.action_type != "edit_query" and self.new_sql is not None:
            raise ValueError("new_sql is only allowed for edit_query actions.")
        return self


class Observation(BaseModel):
    """Observable state returned to the agent."""

    current_sql: str = Field(description="The current SQL query.")
    error_message: str | None = Field(
        default=None,
        description="The latest SQL execution error, if any.",
    )
    result_preview: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Preview of the first few rows from the latest query result.",
    )
    database_schema: str = Field(description="Human-readable database schema.")
    step_count: int = Field(ge=0, description="Number of steps taken so far.")


class Reward(BaseModel):
    """Typed reward wrapper."""

    value: float = Field(description="Reward assigned to the latest step.")


class State(BaseModel):
    """Serializable environment state."""

    task_name: str = Field(description="Current task identifier.")
    current_sql: str = Field(description="Current SQL query under edit.")
    error_message: str | None = Field(default=None)
    result_preview: list[dict[str, Any]] = Field(default_factory=list)
    database_schema: str = Field(description="Human-readable database schema.")
    step_count: int = Field(ge=0)
    done: bool = Field(default=False)


# ── Database layer ─────────────────────────────────────────────────────────────

_DB_PATH: Path = Path(tempfile.gettempdir()) / "sql_debug_env.sqlite3"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id      INTEGER PRIMARY KEY,
    name    TEXT    NOT NULL,
    email   TEXT    NOT NULL UNIQUE,
    region  TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    id       INTEGER PRIMARY KEY,
    user_id  INTEGER NOT NULL,
    product  TEXT    NOT NULL,
    amount   REAL    NOT NULL,
    status   TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

_USERS_SEED = [
    (1, "Alice", "alice@example.com", "north"),
    (2, "Bob",   "bob@example.com",   "south"),
    (3, "Cara",  "cara@example.com",  "west"),
    (4, "Dev",   "dev@example.com",   "north"),
]

_ORDERS_SEED = [
    (1, 1, "keyboard", 120.0, "completed"),
    (2, 1, "mouse",     40.0, "completed"),
    (3, 2, "monitor",  300.0, "pending"),
    (4, 3, "dock",     150.0, "completed"),
    (5, 4, "laptop",   900.0, "completed"),
    (6, 4, "bag",       60.0, "cancelled"),
]


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_database() -> None:
    """Create (or reset) the deterministic SQLite database."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_connection() as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.execute("DELETE FROM orders;")
        conn.execute("DELETE FROM users;")
        conn.executemany(
            "INSERT INTO users (id, name, email, region) VALUES (?, ?, ?, ?);",
            _USERS_SEED,
        )
        conn.executemany(
            "INSERT INTO orders (id, user_id, product, amount, status) VALUES (?, ?, ?, ?, ?);",
            _ORDERS_SEED,
        )
        conn.commit()


def get_schema() -> str:
    return (
        "users(id INTEGER PRIMARY KEY, name TEXT, email TEXT, region TEXT)\n"
        "orders(id INTEGER PRIMARY KEY, user_id INTEGER, product TEXT, amount REAL, status TEXT)"
    )


def execute_query(query: str) -> list[dict[str, Any]] | str:
    """Execute *query* and return rows or an error string."""
    try:
        with _get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    except Exception as exc:
        return str(exc)


# ── Task definitions ───────────────────────────────────────────────────────────

_TASKS: dict[str, dict[str, Any]] = {
    "easy": {
        "name": "easy",
        "difficulty": "easy",
        "description": "Fix a syntax error in a simple filtered SELECT query.",
        "buggy_query": "SELEC id, name FROM users WHERE region = 'north' ORDER BY id;",
        "reference_query": "SELECT id, name FROM users WHERE region = 'north' ORDER BY id;",
        "expected_output": [
            {"id": 1, "name": "Alice"},
            {"id": 4, "name": "Dev"},
        ],
    },
    "medium": {
        "name": "medium",
        "difficulty": "medium",
        "description": "Fix a logical bug in a status filter for completed orders.",
        "buggy_query": "SELECT id, product, amount FROM orders WHERE status = 'pending' ORDER BY id;",
        "reference_query": "SELECT id, product, amount FROM orders WHERE status = 'completed' ORDER BY id;",
        "expected_output": [
            {"id": 1, "product": "keyboard", "amount": 120.0},
            {"id": 2, "product": "mouse",    "amount":  40.0},
            {"id": 4, "product": "dock",     "amount": 150.0},
            {"id": 5, "product": "laptop",   "amount": 900.0},
        ],
    },
    "hard": {
        "name": "hard",
        "difficulty": "hard",
        "description": "Fix a JOIN predicate bug in an aggregated reporting query.",
        "buggy_query": (
            "SELECT u.name, COUNT(o.id) AS completed_orders, SUM(o.amount) AS total_spent "
            "FROM users u "
            "JOIN orders o ON u.id = o.id "
            "WHERE o.status = 'completed' "
            "GROUP BY u.id, u.name "
            "ORDER BY total_spent DESC, u.name ASC;"
        ),
        "reference_query": (
            "SELECT u.name, COUNT(o.id) AS completed_orders, SUM(o.amount) AS total_spent "
            "FROM users u "
            "JOIN orders o ON u.id = o.user_id "
            "WHERE o.status = 'completed' "
            "GROUP BY u.id, u.name "
            "ORDER BY total_spent DESC, u.name ASC;"
        ),
        "expected_output": [
            {"name": "Dev",   "completed_orders": 1, "total_spent": 900.0},
            {"name": "Alice", "completed_orders": 2, "total_spent": 160.0},
            {"name": "Cara",  "completed_orders": 1, "total_spent": 150.0},
        ],
    },
}


def get_task(task_name: str) -> dict[str, Any]:
    if task_name not in _TASKS:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {list(_TASKS)}")
    return _TASKS[task_name].copy()


# ── Grader ─────────────────────────────────────────────────────────────────────

def _normalize_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(row, sort_keys=True) for row in rows]


def grade_result(
    actual: list[dict[str, Any]] | str,
    expected: list[dict[str, Any]],
) -> float:
    """Return a score in [0.0, 1.0] comparing *actual* to *expected*."""
    if isinstance(actual, str) or not isinstance(actual, list):
        return 0.0
    norm_a = _normalize_rows(actual)
    norm_e = _normalize_rows(expected)
    if norm_a == norm_e:
        return 1.0
    if not norm_e:
        return 1.0 if not norm_a else 0.0
    overlap = sum((Counter(norm_a) & Counter(norm_e)).values())
    return round(overlap / len(norm_e), 4)



# Ensure all models with `Any` type hints are fully built
Action.model_rebuild()
Observation.model_rebuild()
Reward.model_rebuild()
State.model_rebuild()

# ── Environment ────────────────────────────────────────────────────────────────

_RESULT_PREVIEW_LIMIT  = 5
_STEP_PENALTY          = 0.05
_SYNTAX_FIX_REWARD     = 0.3
_QUERY_EXECUTES_REWARD = 0.4
_PARTIAL_REWARD        = 0.6
_FULL_REWARD           = 1.0
_WRONG_SUB_PENALTY     = -0.5


class SQLDebugEnv:
    """OpenEnv-style SQL debugging RL environment."""

    def __init__(self, default_task: str = "easy") -> None:
        self._default_task = default_task
        self._task: dict[str, Any] = {}
        self._current_sql = ""
        self._error_message: str | None = None
        self._result_preview: list[dict[str, Any]] = []
        self._step_count = 0
        self._done = False
        self._last_score = 0.0
        initialize_database()
        self.reset(default_task)

    # ── public API ─────────────────────────────────────────────────────────────

    def reset(self, task_name: str | None = None) -> Observation:
        """Reset the environment and return the initial observation."""
        selected = task_name or self._default_task
        initialize_database()
        self._task = get_task(selected)
        self._current_sql = self._task["buggy_query"]
        self._error_message = None
        self._result_preview = []
        self._step_count = 0
        self._done = False
        self._last_score = 0.0
        return self._build_observation()

    def step(
        self, action: "Action | dict[str, Any]"
    ) -> "tuple[Observation, float, bool, dict[str, Any]]":
        """Apply *action* and return (observation, reward, done, info)."""
        if self._done:
            return self._build_observation(), 0.0, True, {"message": "Episode already completed."}

        parsed: Action = (
            action if isinstance(action, Action) else Action.model_validate(action)
        )
        previous_error = self._error_message
        self._step_count += 1

        reward = -_STEP_PENALTY
        score = 0.0
        info: dict[str, Any] = {
            "task_name":       self._task["name"],
            "action_type":     parsed.action_type,
            "score":           0.0,
            "execution_error": None,
        }

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
            else:
                self._error_message = None
                self._result_preview = result[:_RESULT_PREVIEW_LIMIT]
                score = grade_result(result, self._task["expected_output"])
                info["score"] = score
                reward += _QUERY_EXECUTES_REWARD
                if previous_error and not self._error_message:
                    reward += _SYNTAX_FIX_REWARD
                if score == 1.0:
                    reward += _FULL_REWARD
                elif score > 0.0:
                    reward += _PARTIAL_REWARD * score
                self._last_score = score

        if parsed.action_type == "submit":
            self._done = True
            if score < 1.0:
                reward += _WRONG_SUB_PENALTY
            info["submitted"] = True

        info["done"] = self._done
        return self._build_observation(), round(reward, 4), self._done, info

    def state(self) -> State:
        """Return the full serializable environment state."""
        return State(
            task_name=self._task.get("name", ""),
            current_sql=self._current_sql,
            error_message=self._error_message,
            result_preview=self._result_preview,
            database_schema=get_schema(),
            step_count=self._step_count,
            done=self._done,
        )

    def close(self) -> None:
        pass

    # ── private ────────────────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        return Observation(
            current_sql=self._current_sql,
            error_message=self._error_message,
            result_preview=self._result_preview,
            database_schema=get_schema(),
            step_count=self._step_count,
        )
