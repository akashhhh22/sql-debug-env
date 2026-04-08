"""inference.py – Baseline agent for SQLDebugEnv.

Usage:
    python inference.py --task easy --max-steps 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import Action, Observation, SQLDebugEnv

# ── Config from env vars ───────────────────────────────────────────────────────
# API_BASE_URL and MODEL_NAME MUST have defaults per OpenEnv spec
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")

SYSTEM_PROMPT = """You are solving a SQL debugging environment.
Return exactly one JSON object with this schema:
{"action_type":"edit_query|run_query|submit","new_sql":"...optional..."}
- edit_query: provide the complete fixed SQL in new_sql
- run_query: execute the current query to see results
- submit: submit the current query as your final answer
Do not include markdown or extra text.""".strip()


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_prompt(obs: Observation) -> str:
    return json.dumps(
        {
            "current_sql":     obs.current_sql,
            "error_message":   obs.error_message,
            "result_preview":  obs.result_preview,
            "database_schema": obs.database_schema,
            "step_count":      obs.step_count,
        },
        sort_keys=True,
    )


def parse_action(text: str) -> Action:
    try:
        return Action.model_validate(json.loads(text))
    except Exception:
        return Action(action_type="submit")


def fmt_action(a: Action) -> str:
    d: dict[str, Any] = {"action_type": a.action_type}
    if a.new_sql:
        d["new_sql"] = a.new_sql
    return json.dumps(d, separators=(",", ":"))


def fmt_err(e: Any) -> str:
    return "null" if e is None else str(e).replace("\n", " ").strip()


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(task_name: str, max_steps: int) -> int:
    if not HF_TOKEN:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = SQLDebugEnv(default_task=task_name)
    obs = env.reset(task_name)

    print(f"[START] task={task_name} env=SQLDebugEnv model={MODEL_NAME}")

    rewards: list[str] = []
    success = False
    steps = 0

    for n in range(1, max_steps + 1):
        err: str | None = None
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(obs)},
                ],
            )
            action = parse_action(resp.choices[0].message.content or "{}")
        except Exception as e:
            action, err = Action(action_type="submit"), str(e)

        obs, reward, done, info = env.step(action)
        steps = n
        rewards.append(f"{reward:.2f}")

        logged_err = err or info.get("execution_error") or obs.error_message
        print(
            f"[STEP] step={n} action={fmt_action(action)} "
            f"reward={reward:.2f} done={str(done).lower()} error={fmt_err(logged_err)}"
        )

        if done:
            success = info.get("score") == 1.0
            break

    rstr  = ",".join(rewards)
    score = round(sum(float(r) for r in rewards) / len(rewards), 3) if rewards else 0.0
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rstr}")
    return 0 if success else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Run baseline inference for SQLDebugEnv.")
    p.add_argument("--task",      default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--max-steps", type=int,        default=10)
    args = p.parse_args()
    return run_episode(args.task, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
