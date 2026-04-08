"""
Inference script for SQLDebugEnv.
Uses IMAGE_NAME to spin up the Docker container — exactly like the sample.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any, List, Optional

import requests
from openai import OpenAI

# ── Config — per spec ─────────────────────────────────────────────────────────
# Defaults set ONLY for API_BASE_URL and MODEL_NAME. HF_TOKEN has NO default.
IMAGE_NAME   = os.getenv("IMAGE_NAME")                         # Docker image to spin up
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")  # No default — must be set
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK    = "SQLDebugEnv"
MAX_STEPS    = int(os.getenv("MAX_STEPS", "10"))
SERVER_PORT  = 8000
TIMEOUT      = 30

SYSTEM_PROMPT = """You are solving a SQL debugging environment.
Return exactly one JSON object — no markdown, no extra text:
{"action_type": "edit_query|run_query|submit", "new_sql": "...optional..."}
- edit_query : fix the SQL; provide full corrected query in new_sql
- run_query  : execute current SQL to inspect the output
- submit     : submit current SQL as final answer
Strategy: edit_query first, then run_query to verify, then submit.""".strip()


# ── Docker helper — mirrors from_docker_image() in the sample ─────────────────
class DockerEnvServer:
    """Starts the env container from IMAGE_NAME and tears it down on close."""

    def __init__(self, image_name: str, port: int = SERVER_PORT) -> None:
        self.image_name   = image_name
        self.port         = port
        self.container_id = ""

    def start(self) -> str:
        """Start container, wait for /health, return base URL."""
        print(f"[DEBUG] Starting container from image: {self.image_name}", flush=True)
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p",
             f"{self.port}:{self.port}", self.image_name],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

        self.container_id = result.stdout.strip()
        base_url = f"http://localhost:{self.port}"

        for _ in range(30):
            try:
                r = requests.get(f"{base_url}/health", timeout=3)
                if r.status_code == 200:
                    print(f"[DEBUG] Container ready at {base_url}", flush=True)
                    return base_url
            except Exception:
                pass
            time.sleep(1)

        raise RuntimeError("Container did not become healthy within 30s")

    def stop(self) -> None:
        if self.container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self.container_id],
                    capture_output=True, timeout=15,
                )
                print(f"[DEBUG] Container stopped: {self.container_id[:12]}", flush=True)
            except Exception as e:
                print(f"[DEBUG] docker stop error: {e}", flush=True)


# ── HTTP client for the env server ────────────────────────────────────────────
class SQLDebugClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: str = "easy") -> dict:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action: dict) -> dict:
        resp = requests.post(
            f"{self.base_url}/step",
            json=action,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()


# ── STDOUT helpers — strictly per spec ────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    err_val  = str(error).replace("\n", " ").strip() if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call using OpenAI Client (mandatory per spec) ─────────────────────────
def call_llm(client: OpenAI, obs: dict) -> dict:
    prompt = json.dumps({
        "current_sql":     obs.get("current_sql", ""),
        "error_message":   obs.get("error_message", ""),
        "result_preview":  obs.get("result_preview", ""),
        "database_schema": obs.get("database_schema", ""),
        "step_count":      obs.get("step_count", 0),
    }, sort_keys=True)
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        text = (resp.choices[0].message.content or "{}").strip()
        return json.loads(text)
    except Exception:
        return {"action_type": "submit"}


def fmt_action(a: dict) -> str:
    d: dict[str, Any] = {"action_type": a.get("action_type", "submit")}
    if a.get("new_sql"):
        d["new_sql"] = a["new_sql"]
    return json.dumps(d, separators=(",", ":"))


# ── Episode ───────────────────────────────────────────────────────────────────
def run_episode(task_name: str, max_steps: int, base_url: str) -> int:
    rewards:   List[float] = []
    success                = False
    steps_done             = 0
    score                  = 0.0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        env        = SQLDebugClient(base_url)
        obs        = env.reset(task_name)

        for step in range(1, max_steps + 1):
            err    = None
            action = {"action_type": "submit"}

            try:
                action = call_llm(llm_client, obs)
            except Exception as e:
                err = str(e)

            try:
                result = env.step(action)
                obs    = result.get("observation", obs)
                reward = float(result.get("reward", 0.0))
                done   = bool(result.get("done", False))
                info   = result.get("info", {})
            except Exception as e:
                reward, done, info = 0.0, True, {}
                err = err or str(e)

            steps_done = step
            rewards.append(reward)

            logged_err = (
                err
                or info.get("execution_error")
                or (obs.get("error_message") if isinstance(obs, dict) else None)
            )
            log_step(
                step=step,
                action=fmt_action(action),
                reward=reward,
                done=done,
                error=logged_err if logged_err else None,
            )

            if done:
                raw_score = info.get("score", 0.0)
                score     = min(max(float(raw_score), 0.0), 1.0)
                success   = score >= 1.0
                break

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_done, score=score, rewards=rewards)

    return 0 if success else 1


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="SQLDebugEnv inference runner")
    p.add_argument("--task",      default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--all",       action="store_true",
                   help="Run easy, medium and hard sequentially")
    args = p.parse_args()

    docker_server: Optional[DockerEnvServer] = None

    try:
        # Always spin up from IMAGE_NAME — exactly like from_docker_image() in sample
        docker_server = DockerEnvServer(IMAGE_NAME, SERVER_PORT)
        base_url = docker_server.start()

        if args.all:
            results = [run_episode(t, args.max_steps, base_url)
                       for t in ["easy", "medium", "hard"]]
            return 0 if all(r == 0 for r in results) else 1

        return run_episode(args.task, args.max_steps, base_url)

    finally:
        if docker_server:
            docker_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
