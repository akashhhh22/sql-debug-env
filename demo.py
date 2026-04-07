#!/usr/bin/env python3
"""Smoke-test all 3 tasks without needing an LLM."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.environment import SQLDebugEnv
from env.models import Action

FIXES = {
    "easy": [
        Action(action_type="edit_query",
               new_sql="SELECT id, name FROM users WHERE region = 'north' ORDER BY id;"),
        Action(action_type="run_query"),
        Action(action_type="submit"),
    ],
    "medium": [
        Action(action_type="edit_query",
               new_sql=(
                   "SELECT u.name, SUM(o.amount) AS total_spent "
                   "FROM users u JOIN orders o ON u.id = o.user_id "
                   "WHERE o.status = 'completed' "
                   "GROUP BY u.id, u.name ORDER BY total_spent DESC;"
               )),
        Action(action_type="run_query"),
        Action(action_type="submit"),
    ],
    "hard": [
        Action(action_type="edit_query",
               new_sql=(
                   "SELECT u.name, p.category, SUM(o.amount) AS revenue "
                   "FROM users u "
                   "JOIN orders o ON u.id = o.user_id "
                   "JOIN products p ON o.product = p.name "
                   "WHERE o.status = 'completed' "
                   "GROUP BY u.name, p.category "
                   "ORDER BY revenue DESC;"
               )),
        Action(action_type="run_query"),
        Action(action_type="submit"),
    ],
}

def run():
    env = SQLDebugEnv()
    all_passed = True
    print()
    for task, actions in FIXES.items():
        env.reset(task)
        score = 0.0
        for act in actions:
            _, _, _, info = env.step(act)
            score = info.get("score", 0.0)
        ok = score == 1.0
        all_passed = all_passed and ok
        status = "PASS ✅" if ok else f"FAIL ❌ (score={score:.3f})"
        print(f"  {task:<10}  →  {status}")
    print()
    if all_passed:
        print("ALL TASKS PASSED ✅")
    else:
        print("SOME TASKS FAILED ❌")
    return 0 if all_passed else 1

if __name__ == "__main__":
    raise SystemExit(run())
