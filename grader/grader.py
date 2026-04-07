"""Deterministic grading logic for SQLDebugEnv."""
from __future__ import annotations
import json
from collections import Counter
from typing import Any

def _normalize(rows):
    return [json.dumps(r, sort_keys=True) for r in rows]

def grade_result(actual, expected):
    if isinstance(actual, str): return 0.0
    if not isinstance(actual, list): return 0.0
    na, ne = _normalize(actual), _normalize(expected)
    if na == ne: return 1.0
    if not ne: return 1.0 if not na else 0.0
    overlap = sum((Counter(na) & Counter(ne)).values())
    return round(overlap / len(ne), 4)

def grade_task_result(actual, task_definition):
    return grade_result(actual, task_definition["expected_output"])
