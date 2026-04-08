"""
inference.py – HTTP-only baseline agent for SQLDebugEnv.

Communicates with the running Docker server via REST.
No local imports from models.py — only requests + openai.

Usage:
    python inference.py                  # runs easy → medium → hard
    python inference.py --task medium    # runs one task
    python inference.py --max-steps 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any


# ================= CONFIG =================
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")
OPENENV_URL: str = os.environ.get("OPENENV_URL", "http://localhost:7860")

# ================= SYSTEM PROMPT =================
# Designed to produce the 4-step pattern: run_query → edit_query → run_query → submit
# The last_reward field in the observation drives the decision.
SYSTEM_PROMPT = """
You are an expert SQL debugger. Your goal is to fix a buggy SQL query.

Each step you receive a JSON observation:
- current_sql      : the SQL query being debugged
- error_message    : null if no error, otherwise the SQL error string
- result_preview   : rows returned by the last run_query (empty until you run)
- database_schema  : table and column definitions
- step_count       : steps taken so far
- last_reward      : reward from your previous action (0.0 on first step)

FOLLOW THIS EXACT WORKFLOW:

STEP 1 — Always start with run_query to see the current error or results.
STEP 2 — Based on what you see:
          • If error_message is not null → syntax error. Use edit_query with the fully corrected SQL.
          • If no error but result_preview looks wrong/empty → logical bug. Use edit_query with the fix.
            Common logical bugs: wrong WHERE value (e.g. 'pending' → 'completed'),
            wrong JOIN column (e.g. o.id → o.user_id), wrong ORDER BY.
STEP 3 — After editing, always run_query once to verify the fix worked.
STEP 4 — If last_reward > 1.0 and no error_message → the query is correct. Call submit NOW.

HARD RULES:
- If last_reward > 1.0 and error_message is null: you MUST call submit. Do not run again.
- Never call submit if last_reward < 1.0.
- Never call run_query more than 2 times in a row — edit between runs.
- Never submit without having called edit_query at least once.

Return ONLY a single raw JSON object. No markdown. No explanation.
Schema: {"action_type":"edit_query|run_query|submit","new_sql":"full SQL, required only for edit_query"}
""".strip()


# ================= HELPERS =================
def build_prompt(obs: dict[str, Any], last_reward: float = 0.0) -> str:
    return json.dumps(
        {
            "current_sql": obs.get("current_sql", ""),
            "error_message": obs.get("error_message"),
            "result_preview": obs.get("result_preview", []),
            "database_schema": obs.get("database_schema", ""),
            "step_count": obs.get("step_count", 0),
            "last_reward": round(last_reward, 4),
        },
        indent=2,
    )


def clean_json_response(text: str) -> str:
    """Strip markdown code fences if model wraps its response."""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_action(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(clean_json_response(text))
        action_type = parsed.get("action_type", "run_query")
        if action_type not in ("edit_query", "run_query", "submit"):
            action_type = "run_query"
        result: dict[str, Any] = {"action_type": action_type}
        if action_type == "edit_query":
            new_sql = parsed.get("new_sql", "").strip()
            if new_sql:
                result["new_sql"] = new_sql
            else:
                result = {"action_type": "run_query"}  # malformed edit → safe fallback
        return result
    except Exception:
        return {"action_type": "run_query"}


def fmt_action(a: dict[str, Any]) -> str:
    return json.dumps(a, separators=(",", ":"))


def fmt_err(e: Any) -> str:
    return "null" if e is None else str(e).replace("\n", " ").strip()


# ================= HTTP HELPERS =================
def env_reset(task_name: str) -> dict[str, Any]:
    """POST /reset → returns the initial observation dict."""
    import requests

    resp = requests.post(
        f"{OPENENV_URL}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("observation", data)


def env_step(action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
    """POST /step → returns (observation, reward, done, info)."""
    import requests

    resp = requests.post(
        f"{OPENENV_URL}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return (
        data.get("observation", {}),
        float(data.get("reward", 0.0)),
        bool(data.get("done", False)),
        data.get("info", {}),
    )


# ================= RUN EPISODE =================
def run_episode(task_name: str, max_steps: int) -> int:
    from openai import OpenAI

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN environment variable is not set.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    try:
        obs = env_reset(task_name)
    except Exception as exc:
        print(f"[ERROR] Failed to reset env: {exc}", file=sys.stderr)
        return 1

    print(f"[START] task={task_name} env=SQLDebugEnv model={MODEL_NAME}")

    rewards: list[str] = []
    success = False
    steps = 0
    last_reward: float = 0.0
    edited_once = False  # track whether edit_query was called at least once

    for n in range(1, max_steps + 1):
        llm_err = None

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(obs, last_reward)},
                ],
            )
            raw_text = resp.choices[0].message.content or "{}"
            action = parse_action(raw_text)
        except Exception as exc:
            llm_err = str(exc)
            action = {"action_type": "run_query"}

        # ── Hard guards (code-level safety net) ──────────────────────────────
        # 1. Never submit on the very first step.
        if n == 1 and action["action_type"] == "submit":
            action = {"action_type": "run_query"}

        # 2. Never submit if last_reward < 1.0 (wrong answer penalty would apply).
        if action["action_type"] == "submit" and last_reward < 1.0:
            action = {"action_type": "run_query"}

        # 3. Never submit without at least one edit (catches pure run→submit loops).
        if action["action_type"] == "submit" and not edited_once:
            action = {"action_type": "run_query"}

        # Track whether an edit has been made.
        if action["action_type"] == "edit_query":
            edited_once = True

        # ── Step ─────────────────────────────────────────────────────────────
        try:
            obs, reward, done, info = env_step(action)
        except Exception as exc:
            print(f"[ERROR] Failed to step env: {exc}", file=sys.stderr)
            return 1

        last_reward = reward
        rewards.append(f"{reward:.2f}")
        steps = n

        logged_err = llm_err or info.get("execution_error") or obs.get("error_message")
        print(
            f"[STEP] step={n} "
            f"action={fmt_action(action)} "
            f"reward={reward:.2f} "
            f"done={str(done).lower()} "
            f"error={fmt_err(logged_err)}"
        )

        if done:
            success = info.get("score") == 1.0
            break

    score = round(sum(float(r) for r in rewards) / len(rewards), 3) if rewards else 0.0
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={','.join(rewards)}"
    )
    return 0 if success else 1


# ================= MAIN =================
ALL_TASKS = ["easy", "medium", "hard"]


def main() -> int:
    parser = argparse.ArgumentParser(description="SQLDebugEnv inference agent")
    parser.add_argument(
        "--task",
        default=None,
        choices=ALL_TASKS,
        help="Single task to run. Omit to run all 3 tasks sequentially.",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    tasks_to_run = [args.task] if args.task else ALL_TASKS

    results: list[dict[str, Any]] = []
    any_failure = False

    for task_name in tasks_to_run:
        code = run_episode(task_name, args.max_steps)
        ok = code == 0
        results.append({"task": task_name, "success": ok})
        if not ok:
            any_failure = True

    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())