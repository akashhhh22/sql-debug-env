"""Easy SQL debugging task — syntax typo in SELECT keyword."""

EASY_TASK = TASK = {
    "name": "easy",
    "difficulty": "easy",
    "max_steps": 5,
    "description": "Fix a syntax typo: SELEC → SELECT in a filtered query.",
    "buggy_query":
        "SELEC id, name FROM users WHERE region = 'north' ORDER BY id;",
    "reference_query":
        "SELECT id, name FROM users WHERE region = 'north' ORDER BY id;",
    "expected_output": [
        {"id": 1, "name": "Alice"},
        {"id": 4, "name": "Dev"},
    ],
}

def get_task() -> dict:
    return TASK.copy()
