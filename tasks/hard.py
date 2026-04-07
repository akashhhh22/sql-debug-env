"""Hard SQL debugging task — wrong JOIN key + wrong aggregate."""

HARD_TASK = TASK = {
    "name": "hard",
    "difficulty": "hard",
    "max_steps": 10,
    "description": "Fix two bugs: wrong JOIN key (o.id → o.user_id) and wrong aggregate (SUM(p.price) → SUM(o.amount)).",
    "buggy_query": (
        "SELECT u.name, p.category, SUM(p.price) AS revenue "
        "FROM users u "
        "JOIN orders o ON u.id = o.id "
        "JOIN products p ON o.product = p.name "
        "WHERE o.status = 'completed' "
        "GROUP BY u.name, p.category "
        "ORDER BY revenue DESC;"
    ),
    "reference_query": (
        "SELECT u.name, p.category, SUM(o.amount) AS revenue "
        "FROM users u "
        "JOIN orders o ON u.id = o.user_id "
        "JOIN products p ON o.product = p.name "
        "WHERE o.status = 'completed' "
        "GROUP BY u.name, p.category "
        "ORDER BY revenue DESC;"
    ),
    "expected_output": [
        {"name": "Alice",   "category": "electronics", "revenue": 1200.0},
        {"name": "Dev",     "category": "electronics", "revenue": 1200.0},
        {"name": "Bob",     "category": "electronics", "revenue":  300.0},
        {"name": "Charlie", "category": "electronics", "revenue":  300.0},
        {"name": "Eva",     "category": "electronics", "revenue":   80.0},
        {"name": "Alice",   "category": "accessories", "revenue":   25.0},
        {"name": "Frank",   "category": "accessories", "revenue":   25.0},
    ],
}

def get_task() -> dict:
    return TASK.copy()
