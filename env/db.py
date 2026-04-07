"""SQLite database utilities for SQLDebugEnv — 4-table schema."""

from __future__ import annotations
import sqlite3, tempfile
from pathlib import Path

# Cross-platform temp path — works on Windows, Linux, Mac
DB_PATH = Path(tempfile.gettempdir()) / "sql_debug_env.sqlite3"

SCHEMA_SQL = """
PRAGMA foreign_keys = OFF;
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS users;
PRAGMA foreign_keys = ON;

CREATE TABLE users (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE, region TEXT NOT NULL,
    age INTEGER, created_at TEXT
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
    product TEXT NOT NULL, amount REAL NOT NULL,
    status TEXT NOT NULL, ordered_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL,
    category TEXT NOT NULL, price REAL NOT NULL, stock INTEGER
);
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL, rating INTEGER NOT NULL,
    comment TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);
"""

USERS_SEED = [
    (1,"Alice",  "alice@example.com",  "north",28,"2023-01-15"),
    (2,"Bob",    "bob@example.com",    "south",34,"2023-02-20"),
    (3,"Charlie","charlie@example.com","east", 22,"2023-03-10"),
    (4,"Dev",    "dev@example.com",    "north",45,"2023-04-05"),
    (5,"Eva",    "eva@example.com",    "west", 31,"2023-05-18"),
    (6,"Frank",  "frank@example.com",  "south",27,"2023-06-22"),
]
ORDERS_SEED = [
    (1,1,"Laptop",  1200.0,"completed","2024-01-10"),
    (2,1,"Mouse",     25.0,"completed","2024-01-15"),
    (3,2,"Keyboard",  75.0,"pending",  "2024-02-01"),
    (4,3,"Monitor",  300.0,"completed","2024-02-10"),
    (5,4,"Laptop",  1200.0,"completed","2024-03-05"),
    (6,4,"Headset",  150.0,"cancelled","2024-03-12"),
    (7,5,"Webcam",    80.0,"completed","2024-04-01"),
    (8,6,"Mouse",     25.0,"completed","2024-04-15"),
    (9,2,"Monitor",  300.0,"completed","2024-05-01"),
]
PRODUCTS_SEED = [
    (1,"Laptop",  "electronics",1200.0,10),
    (2,"Mouse",   "accessories",  25.0,50),
    (3,"Keyboard","accessories",  75.0,30),
    (4,"Monitor", "electronics", 300.0,15),
    (5,"Headset", "accessories", 150.0,20),
    (6,"Webcam",  "electronics",  80.0,25),
]
REVIEWS_SEED = [
    (1,1,1,5,"Excellent laptop!"),
    (2,1,2,4,"Good mouse"),
    (3,3,4,3,"Average monitor"),
    (4,4,1,5,"Love this laptop"),
    (5,5,6,4,"Nice webcam"),
    (6,2,4,2,"Disappointing"),
]

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def initialize_database() -> Path:
    """Drop and recreate all tables — no file deletion needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)   # DROP + CREATE handles schema change
        conn.executemany("INSERT INTO users    VALUES (?,?,?,?,?,?);", USERS_SEED)
        conn.executemany("INSERT INTO orders   VALUES (?,?,?,?,?,?);", ORDERS_SEED)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?,?);",   PRODUCTS_SEED)
        conn.executemany("INSERT INTO reviews  VALUES (?,?,?,?,?);",   REVIEWS_SEED)
        conn.commit()
    return DB_PATH

def fetch_schema() -> str:
    return (
        "users(id INTEGER PRIMARY KEY, name TEXT, email TEXT, region TEXT, age INTEGER, created_at TEXT)\n"
        "orders(id INTEGER PRIMARY KEY, user_id INTEGER, product TEXT, amount REAL, status TEXT, ordered_at TEXT)\n"
        "products(id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER)\n"
        "reviews(id INTEGER PRIMARY KEY, user_id INTEGER, product_id INTEGER, rating INTEGER, comment TEXT)"
    )

def execute_query(query: str):
    try:
        with get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        return str(e)

def get_schema():
    return fetch_schema()

initialize_database()
