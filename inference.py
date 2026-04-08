"""Baseline inference script for SQLDebugEnv."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional

from openai import OpenAI

# Make all local packages importable from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import SQLDebugEnv
from env.models import Action, Observation

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
BENCHMARK    = "SQLDebugEnv"
MAX_STEPS    = 10

SYSTEM_PROMPT = (
    "You are solving a SQL debugging environment.\n"
    "Return exactly one JSON object:\n"
    '{"action_type":"edit_query|run_query|submit","new_sql":"optional"}\n'
    "- edit_query: provide the complete fixed SQL in new_sql\n"
    "- run_query: execute the current query to see results\n"
    "- submit: submit the current query as your final answer\n"
    "No markdown or extra text."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
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
        data = json.loads(text)
        return Action.model_validate(data)
    except Exception:
        return Action(action_type="submit")


def fmt_action(a: Action) -> str:
    d: dict[str, Any] = {"action_type": a.action_type}
    if a.new_sql:
        d["new_sql"] = a.new_sql
    return json.dumps(d, separators=(",", ":"))


def fmt_err(e: Any) -> str:
    return "null" if not e else str(e).replace("\n", " ").strip()


def get_llm_action(
    client: OpenAI, obs: Observation
) -> tuple[Action, Optional[str]]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(obs)},
            ],
        )
        text = (resp.choices[0].message.content or "{}").strip()
        return parse_action(text), None
    except Exception as exc:
        return Action(action_type="submit"), str(exc)


# ── Episode ───────────────────────────────────────────────────────────────────
def run_episode(task_name: str, max_steps: int) -> int:
    rewards: List[float] = []
    success = False
    steps   = 0

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
        env    = SQLDebugEnv(default_task=task_name)
        obs    = env.reset(task_name)

        for n in range(1, max_steps + 1):
            action, err = get_llm_action(client, obs)

            obs, reward, done, info = env.step(action)
            steps = n
            rewards.append(reward)

            logged_err = (
                err
                or info.get("execution_error")
                or obs.error_message
                or None
            )
            print(
                f"[STEP] step={n} action={fmt_action(action)} "
                f"reward={reward:.2f} done={str(done).lower()} "
                f"error={fmt_err(logged_err)}",
                flush=True,
            )

            if done:
                success = info.get("score", 0.0) == 1.0
                break

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        rstr  = ",".join(f"{r:.2f}" for r in rewards)
        score = sum(rewards) / len(rewards) if rewards else 0.0
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.3f} rewards={rstr}",
            flush=True,
        )

    return 0 if success else 1


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="SQLDebugEnv baseline inference")
    p.add_argument("--task",      default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = p.parse_args()
    return run_episode(args.task, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
