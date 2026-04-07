#!/usr/bin/env python3
"""Test all 11 tasks in tasks_dataset.json directly against SQLite."""
import json, sqlite3, os, sys, tempfile
from pathlib import Path

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks_dataset.json")

def get_conn():
    db = Path(tempfile.gettempdir()) / "sql_debug_env.sqlite3"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn

def run(sql):
    try:
        conn = get_conn()
        rows = [dict(r) for r in conn.execute(sql).fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return str(e)

def grade(actual, expected):
    import json as _j
    if isinstance(actual, str) or not isinstance(actual, list): return 0.0
    a = sorted([_j.dumps(r, sort_keys=True) for r in actual])
    e = sorted([_j.dumps(r, sort_keys=True) for r in expected])
    if a == e: return 1.0
    if not e: return 1.0 if not a else 0.0
    from collections import Counter
    overlap = sum((Counter(a) & Counter(e)).values())
    return round(overlap / len(e), 4)

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"❌ tasks_dataset.json not found"); return 1

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    tasks = dataset["tasks"]
    print(f"\nLoaded {len(tasks)} tasks from tasks_dataset.json")
    print("="*60)

    results = []
    for task in tasks:
        actual  = run(task["correct_sql"])
        expected = task.get("expected_rows", [])
        score   = grade(actual, expected)
        passed  = score == 1.0
        results.append({"id":task["id"],"difficulty":task["difficulty"],"passed":passed,"score":score})
        status = "PASS ✅" if passed else f"FAIL ❌ (score={score:.3f}, got {len(actual) if isinstance(actual,list) else actual!r} rows)"
        print(f"  [{task['difficulty']:6s}] {task['id']:<28} → {status}")

    print("="*60)
    by_diff = {}
    for r in results:
        d = r["difficulty"]
        by_diff.setdefault(d, {"pass":0,"total":0})
        by_diff[d]["total"] += 1
        if r["passed"]: by_diff[d]["pass"] += 1

    print("\nBy difficulty:")
    for d in ["easy","medium","hard"]:
        if d in by_diff:
            p,t = by_diff[d]["pass"], by_diff[d]["total"]
            print(f"  {d:8s}  {p}/{t}  {'✅'*p}{'❌'*(t-p)}")

    total, passed = len(results), sum(1 for r in results if r["passed"])
    print(f"\nOverall: {passed}/{total} passed")
    if passed == total: print("\n🎉 ALL DATASET TASKS PASSED!")
    else: print(f"\n⚠️  {total-passed} task(s) need attention.")
    return 0 if passed == total else 1

if __name__ == "__main__":
    raise SystemExit(main())
