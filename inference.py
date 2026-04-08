"""
inference.py – HTTP-only baseline agent for SQLDebugEnv.

Communicates with the running Docker server via REST.
No local imports from models.py — only requests + openai.

Usage:
    python inference.py --task easy --max-steps 10
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


SYSTEM_PROMPT = """
You are solving a SQL debugging environment.

Return ONLY raw JSON.
DO NOT use markdown.
DO NOT explain anything.

Schema:
{"action_type":"edit_query|run_query|submit","new_sql":"optional"}

Rules:
- edit_query → provide FULL corrected SQL query in new_sql
- run_query → test current SQL
- submit → submit only when confident query is correct
""".strip()


# ================= HELPERS =================
def build_prompt(obs: dict[str, Any]) -> str:
    return json.dumps(
        {
            "current_sql": obs.get("current_sql", ""),
            "error_message": obs.get("error_message"),
            "result_preview": obs.get("result_preview", []),
            "database_schema": obs.get("database_schema", ""),
            "step_count": obs.get("step_count", 0),
        },
        indent=2,
    )


def clean_json_response(text: str) -> str:
    """Remove markdown wrappers if model returns code block."""
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def parse_action(text: str) -> dict[str, Any]:
    try:
        cleaned = clean_json_response(text)
        parsed = json.loads(cleaned)
        action_type = parsed.get("action_type", "run_query")
        if action_type not in ("edit_query", "run_query", "submit"):
            action_type = "run_query"
        result: dict[str, Any] = {"action_type": action_type}
        if action_type == "edit_query":
            new_sql = parsed.get("new_sql")
            if new_sql:
                result["new_sql"] = new_sql
            else:
                # edit_query without sql → fall back to run_query
                result = {"action_type": "run_query"}
        return result
    except Exception:
        return {"action_type": "run_query"}


def fmt_action(a: dict[str, Any]) -> str:
    return json.dumps(a, separators=(",", ":"))


def fmt_err(e: Any) -> str:
    return "null" if e is None else str(e).replace("\n", " ").strip()


# ================= HTTP HELPERS =================
def env_reset(task_name: str) -> dict[str, Any]:
    """POST /reset and return the observation dict."""
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
    """POST /step and return (observation, reward, done, info)."""
    import requests

    resp = requests.post(
        f"{OPENENV_URL}/step",
        json=action,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    obs = data.get("observation", {})
    reward = float(data.get("reward", 0.0))
    done = bool(data.get("done", False))
    info = data.get("info", {})
    return obs, reward, done, info


# ================= RUN EPISODE =================
def run_episode(task_name: str, max_steps: int) -> int:
    from openai import OpenAI

    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN environment variable.")

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    # Reset environment via HTTP
    try:
        obs = env_reset(task_name)
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}", file=sys.stderr)
        return 1

    print(f"[START] task={task_name} env=SQLDebugEnv model={MODEL_NAME}")

    rewards: list[str] = []
    success = False
    steps = 0

    for n in range(1, max_steps + 1):
        err = None

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,
                max_tokens=150,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(obs)},
                ],
            )
            raw_text = resp.choices[0].message.content or "{}"
            action = parse_action(raw_text)
        except Exception as e:
            err = str(e)
            action = {"action_type": "run_query"}

        # Prevent dumb early submit
        if n == 1 and action.get("action_type") == "submit":
            action = {"action_type": "run_query"}

        # Step via HTTP
        try:
            obs, reward, done, info = env_step(action)
        except Exception as e:
            print(f"[ERROR] Failed to step environment: {e}", file=sys.stderr)
            return 1

        rewards.append(f"{reward:.2f}")
        steps = n

        logged_err = err or info.get("execution_error") or obs.get("error_message")

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
    rewards_str = ",".join(rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={rewards_str}"
    )

    return 0 if success else 1


# ================= MAIN =================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=10)

    args = parser.parse_args()

    return run_episode(args.task, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())