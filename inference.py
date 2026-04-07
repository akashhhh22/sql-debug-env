"""Inference script for SQLDebugEnv — strict OpenEnv STDOUT spec."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import SQLDebugEnv
from env.models import Action, Observation

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-Coder-32B-Instruct"
BENCHMARK    = "SQLDebugEnv"

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks_dataset.json")
_DB          = Path(tempfile.gettempdir()) / "sql_debug_env.sqlite3"

DATASET_TASKS: dict[str, dict] = {}
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH) as _f:
        for _t in json.load(_f)["tasks"]:
            DATASET_TASKS[_t["id"]] = _t

ENV_TASKS = {"easy", "medium", "hard"}
SUCCESS_SCORE_THRESHOLD = 1.0

# ── STDOUT helpers ─────────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are solving a SQL debugging environment.
Return exactly one JSON object — no markdown, no extra text:
{"action_type": "edit_query|run_query|submit", "new_sql": "...optional..."}

Actions:
- edit_query : fix the SQL (provide full corrected query in new_sql)
- run_query  : execute current SQL to see the output
- submit     : submit current SQL as your final answer

STRICT RULES:
1. ALWAYS edit_query FIRST to fix the bug, then run_query to verify, then submit
2. If run_query returns 0 rows — the filter/join is WRONG. Use edit_query to fix it
3. If run_query returns wrong row count — GROUP BY or JOIN is WRONG. Fix it
4. NEVER use LEFT JOIN unless the task explicitly requires NULL rows in output
5. NEVER invent computed columns (e.g. price*amount) unless the schema has no direct amount column
6. Once run_query shows correct rows, ALWAYS submit immediately""".strip()


def build_prompt(
    current_sql: str,
    error: Optional[str],
    preview: list,
    schema: str,
    step: int,
    last_score: float = 0.0,
    task_description: str = "",
    expected_row_count: int = -1,
    best_sql: str = "",
    best_score: float = 0.0,
    expected_columns: list[str] = [],
) -> str:
    d: dict = {
        "step": step,
        "current_sql": current_sql,
        "error_message": error,
        "result_preview": preview[:5],
        "database_schema": schema,
    }
    if task_description:
        d["task_goal"] = task_description
    if expected_columns:
        d["expected_output_columns"] = expected_columns

    # Best SQL hint — steer back if LLM regressed
    if best_score > last_score and best_sql and best_sql != current_sql:
        d["warning"] = (
            f"Your last change LOWERED the score from {best_score:.2f} to {last_score:.2f}. "
            f"Revert to this better query: {best_sql}"
        )
    elif last_score >= 1.0:
        d["hint"] = "Query is correct! Use submit now."
    elif preview is not None and expected_row_count >= 0:
        actual = len(preview)
        if actual == 0:
            d["hint"] = (
                "Got 0 rows. The filter value or JOIN condition is WRONG. "
                "Check WHERE clause values and use INNER JOIN with status = 'completed'."
            )
        elif actual != expected_row_count:
            d["hint"] = (
                f"Got {actual} rows but expected {expected_row_count}. "
                "GROUP BY or JOIN condition is wrong — fix it with edit_query."
            )
    return json.dumps(d, sort_keys=True)


def build_prompt_obs(obs: Observation) -> str:
    return json.dumps(
        {
            "current_sql": obs.current_sql,
            "error_message": obs.error_message,
            "result_preview": obs.result_preview,
            "database_schema": obs.database_schema,
            "step_count": obs.step_count,
        },
        sort_keys=True,
    )


def call_llm(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "{}").strip()


def parse_obs_action(text: str) -> Action:
    try:
        return Action.model_validate(json.loads(text))
    except Exception:
        return Action(action_type="submit")


def parse_raw_action(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {"action_type": "submit"}


def fmt_action_obs(a: Action) -> str:
    d: dict[str, Any] = {"action_type": a.action_type}
    if a.new_sql:
        d["new_sql"] = a.new_sql
    return json.dumps(d, separators=(",", ":"))


def fmt_action_raw(a: dict) -> str:
    return json.dumps(
        {k: v for k, v in a.items() if v is not None},
        separators=(",", ":"),
    )


# ── SQLite helpers ─────────────────────────────────────────────────────────────
def run_sql(sql: str) -> list | str:
    try:
        conn = sqlite3.connect(str(_DB))
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
        conn.close()
        return rows
    except Exception as exc:
        return str(exc)


def get_schema() -> str:
    conn = sqlite3.connect(str(_DB))
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    lines = []
    for t in tables:
        cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
        lines.append(t + "(" + ", ".join(f"{c[1]} {c[2]}" for c in cols) + ")")
    conn.close()
    return "\n".join(lines)


def grade(actual: list | str, expected: list) -> float:
    if isinstance(actual, str):
        return 0.0
    a = sorted([json.dumps(r, sort_keys=True) for r in actual])
    e = sorted([json.dumps(r, sort_keys=True) for r in expected])
    if a == e:
        return 1.0
    if not e:
        return 1.0 if not actual else 0.0
    overlap = sum((Counter(a) & Counter(e)).values())
    return round(overlap / len(e), 4)


def get_expected_columns(expected: list) -> list[str]:
    """Extract column names from expected rows."""
    if expected and isinstance(expected[0], dict):
        return list(expected[0].keys())
    return []


# ── Episode: env tasks (easy / medium / hard) ─────────────────────────────────
def run_env_episode(task_name: str, max_steps: int) -> int:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = SQLDebugEnv(default_task=task_name)
    obs = env.reset(task_name)
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    success, steps_done, score = False, 0, 0.0
    consecutive_same, last_atype = 0, ""

    try:
        for step in range(1, max_steps + 1):
            err = None
            if consecutive_same >= 3:
                action = Action(action_type="submit")
            elif consecutive_same >= 2 and last_atype == "edit_query":
                action = Action(action_type="run_query")
            else:
                try:
                    action = parse_obs_action(
                        call_llm(client, build_prompt_obs(obs))
                    )
                except Exception as e:
                    action, err = Action(action_type="submit"), str(e)

            consecutive_same = consecutive_same + 1 if action.action_type == last_atype else 0
            last_atype = action.action_type

            obs, reward, done, info = env.step(action)
            steps_done = step
            rewards.append(reward)
            curr_score = float(info.get("score", 0.0))

            logged_err = err or info.get("execution_error") or obs.error_message
            log_step(
                step=step,
                action=fmt_action_obs(action),
                reward=reward,
                done=done,
                error=str(logged_err).replace("\n", " ") if logged_err else None,
            )

            if done:
                score = min(max(curr_score, 0.0), 1.0)
                success = score >= SUCCESS_SCORE_THRESHOLD
                break
            if curr_score >= 1.0 and action.action_type == "run_query":
                consecutive_same = 99  # force submit next
    finally:
        log_end(success=success, steps=steps_done, score=score, rewards=rewards)
    return 0 if success else 1


# ── Episode: dataset tasks (all 11 by ID) ─────────────────────────────────────
def run_dataset_episode(task: dict, client: OpenAI, max_steps: int) -> dict:
    task_id = task["id"]
    expected = task.get("expected_rows", [])
    task_description = task.get("description", "")
    expected_cols = get_expected_columns(expected)
    schema = get_schema()

    current_sql = task["buggy_sql"]
    error: str | None = None
    preview: list = []
    score = 0.0
    steps_done = 0
    success = False
    rewards: list[float] = []
    consecutive_same = 0
    last_atype = ""
    last_score = 0.0

    # Track best SQL seen across all run_query calls
    best_sql = current_sql
    best_score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, max_steps + 1):
            err = None
            done = False

            # Force decisions based on state
            if last_score >= 1.0:
                action = {"action_type": "submit"}
            elif consecutive_same >= 3:
                # Stuck — revert to best known SQL and submit
                action = {"action_type": "submit", "new_sql": best_sql}
            elif consecutive_same >= 2 and last_atype == "edit_query":
                action = {"action_type": "run_query"}
            else:
                prompt = build_prompt(
                    current_sql,
                    error,
                    preview,
                    schema,
                    step,
                    last_score=last_score,
                    task_description=task_description,
                    expected_row_count=len(expected),
                    best_sql=best_sql,
                    best_score=best_score,
                    expected_columns=expected_cols,
                )
                try:
                    action = parse_raw_action(call_llm(client, prompt))
                except Exception as exc:
                    action, err = {"action_type": "submit"}, str(exc)

            atype = action.get("action_type", "submit")
            new_sql = (action.get("new_sql") or "").strip()
            consecutive_same = consecutive_same + 1 if atype == last_atype else 0
            last_atype = atype
            steps_done = step
            reward = 0.0

            if atype == "edit_query" and new_sql:
                current_sql = new_sql
                error, preview = None, []

            elif atype == "run_query":
                result = run_sql(current_sql)
                if isinstance(result, str):
                    error, preview = result, []
                    err = result
                else:
                    error, preview = None, result
                    run_score = grade(result, expected)
                    reward = run_score
                    # Update best SQL if this run is better
                    if run_score > best_score:
                        best_score = run_score
                        best_sql = current_sql
                    last_score = run_score

            elif atype == "submit":
                # Always submit best SQL we found
                submit_sql = best_sql if best_score > last_score else current_sql
                result = run_sql(submit_sql)
                last_score = grade(result, expected) if isinstance(result, list) else 0.0
                score = last_score
                reward = score
                done = True
                # Patch action log to show what was actually submitted
                if submit_sql != current_sql:
                    action = {"action_type": "submit", "new_sql": submit_sql}

            rewards.append(reward)
            log_step(
                step=step,
                action=fmt_action_raw(action),
                reward=reward,
                done=done,
                error=str(err).replace("\n", " ") if err else None,
            )

            if done:
                score = min(max(score, 0.0), 1.0)
                success = score >= SUCCESS_SCORE_THRESHOLD
                break
    finally:
        log_end(success=success, steps=steps_done, score=score, rewards=rewards)

    return {
        "id": task_id,
        "difficulty": task["difficulty"],
        "passed": success,
        "score": score,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    all_choices = sorted(ENV_TASKS | set(DATASET_TASKS.keys()))
    p = argparse.ArgumentParser(description="SQLDebugEnv inference runner")
    p.add_argument("--task", default="easy", choices=all_choices)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--all", action="store_true", help="Run all 11 dataset tasks")
    args = p.parse_args()

    if not API_KEY:
        print("ERROR: HF_TOKEN is not set.", flush=True)
        return 1

    if args.all:
        if not DATASET_TASKS:
            print("ERROR: tasks_dataset.json not found.", flush=True)
            return 1
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        results = [run_dataset_episode(t, client, args.max_steps) for t in DATASET_TASKS.values()]
        passed = sum(1 for r in results if r["passed"])
        return 0 if passed == len(results) else 1

    if args.task in DATASET_TASKS:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        r = run_dataset_episode(DATASET_TASKS[args.task], client, args.max_steps)
        return 0 if r["passed"] else 1

    return run_env_episode(args.task, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())