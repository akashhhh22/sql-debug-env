"""Medium SQL debugging task — wrong JOIN key."""

MEDIUM_TASK = TASK = {
    "name": "medium",
    "difficulty": "medium",
    "max_steps": 8,
    "description": "Fix the JOIN condition: orders link via user_id, not id.",
    "buggy_query": (
        "SELECT u.name, SUM(o.amount) AS total_spent "
        "FROM users u JOIN orders o ON u.id = o.id "
        "WHERE o.status = 'completed' "
        "GROUP BY u.id, u.name ORDER BY total_spent DESC;"
    ),
    "reference_query": (
        "SELECT u.name, SUM(o.amount) AS total_spent "
        "FROM users u JOIN orders o ON u.id = o.user_id "
        "WHERE o.status = 'completed' "
        "GROUP BY u.id, u.name ORDER BY total_spent DESC;"
    ),
    "expected_output": [
        {"name": "Alice",   "total_spent": 1225.0},
        {"name": "Dev",     "total_spent": 1200.0},
        {"name": "Bob",     "total_spent":  300.0},
        {"name": "Charlie", "total_spent":  300.0},
        {"name": "Eva",     "total_spent":   80.0},
        {"name": "Frank",   "total_spent":   25.0},
    ],
}

def get_task() -> dict:
    return TASK.copy()
