from __future__ import annotations

import random

from sqlalchemy import create_engine, text

from app.config import MARIADB_SQLALCHEMY_URI

_engine = create_engine(MARIADB_SQLALCHEMY_URI)

# Mirrors 06_Local_RAG_Agent/data_ingestion/init_db.py's mock employee generator
# exactly (same departments, name pools, and random seed) so both projects'
# mock company data agree.
_DEPARTMENT_TARGETS = [
    ("engineering", 16),
    ("product", 7),
    ("cloud", 6),
    ("security", 5),
    ("data-analytics", 5),
    ("sales", 9),
    ("marketing", 7),
    ("tech-ops", 6),
    ("technical-success", 4),
    ("people-talent", 6),
    ("finance", 4),
    ("legal", 3),
    ("cto", 1),
    ("ceo-team", 1),
]

_FIRST_NAMES = [
    "Alex", "Bao", "Mia", "Ethan", "Sophia", "Noah", "Ava", "Lucas",
    "Isabella", "Oliver", "Liam", "Emma", "James", "Charlotte", "Benjamin",
    "Amelia", "Henry", "Evelyn", "Alexander", "Harper", "Michael", "Abigail",
    "Daniel", "Ella", "Matthew", "Scarlett", "Joseph", "Victoria", "Samuel",
    "Grace", "David", "Chloe", "Andrew", "Zoey", "Nathan", "Lily", "Ryan",
    "Hannah", "Leo", "Aria", "Jack", "Nora", "Sebastian", "Penelope", "Wyatt",
    "Avery", "Julian", "Layla", "Isaac", "Aurora", "Connor", "Violet", "Mason",
    "Luna", "Owen", "Stella", "Caleb", "Ivy", "Noelle", "Eli", "Simon",
    "Mila", "Nathaniel", "Jade", "Omar", "Tara", "Jin", "Maggie", "Quinn",
    "Riley", "Sophie", "Tomas", "Victor", "Wendy", "Yara", "Zane", "Margo",
]

_LAST_NAMES = [
    "Chen", "Wang", "Li", "Zhang", "Liu", "Zhao", "Sun", "Gao", "Xu", "Zhou",
    "Hu", "Jin", "Yao", "Wu", "Qian", "Deng", "Fan", "Guo", "Ren", "He",
    "Nie", "Xie", "Tang", "Song", "Lin", "Luo", "Ma", "Shen", "Pei", "Yuan",
    "Cai", "Lai", "Ke", "Pan", "Dai", "Fu", "Kang", "Mo", "Bai", "Qi",
    "Ye", "Ge", "Dong", "Su", "Rao", "Kong", "Zeng", "Luo", "Fang", "Ng",
    "Park", "Smith", "Johnson", "Taylor", "Anderson", "Hernandez", "Lopez",
    "Clark", "Wong", "Nguyen", "Patel", "Kumar", "Ali", "Kim", "Chen",
]

_SALARY_GRADES = ["G3", "G4", "G5", "G6"]
_TOTAL_EMPLOYEES = sum(count for _dept, count in _DEPARTMENT_TARGETS)


def ensure_users_table() -> None:
    with _engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    department VARCHAR(50) NOT NULL,
                    salary_grade VARCHAR(10) NOT NULL,
                    remaining_vacation_day INT NOT NULL
                )
                """
            )
        )


def seed_test_employees(rows: list[dict]) -> int:
    """Insert arbitrary rows into users. General-purpose helper for tests/manual use."""
    if not rows:
        return 0

    ensure_users_table()
    with _engine.begin() as conn:
        for row in rows:
            conn.execute(
                text(
                    """
                    INSERT INTO users (name, department, salary_grade, remaining_vacation_day)
                    VALUES (:name, :department, :salary_grade, :remaining_vacation_day)
                    """
                ),
                row,
            )
    return len(rows)


def seed_mock_employees() -> int:
    """Seed the full mock company (80 employees across 14 departments), matching
    06_Local_RAG_Agent/data_ingestion/init_db.py's generator exactly. Idempotent:
    if the table is already fully seeded, does nothing and returns 0.
    """
    ensure_users_table()

    with _engine.begin() as conn:
        existing = conn.execute(text("SELECT name, department FROM users")).fetchall()

    existing_count = len(existing)
    existing_departments = {row.department for row in existing}
    requested_departments = {dept for dept, _count in _DEPARTMENT_TARGETS}

    if existing_count == _TOTAL_EMPLOYEES and existing_departments.issubset(requested_departments):
        return 0

    random.seed(42)
    target_employees = []
    name_index = 0

    for department, count in _DEPARTMENT_TARGETS:
        for _ in range(count):
            first = _FIRST_NAMES[name_index % len(_FIRST_NAMES)]
            last = _LAST_NAMES[(name_index // len(_FIRST_NAMES)) % len(_LAST_NAMES)]
            target_employees.append(
                {
                    "name": f"{first} {last}",
                    "department": department,
                    "salary_grade": random.choice(_SALARY_GRADES),
                    "remaining_vacation_day": random.randint(8, 28),
                }
            )
            name_index += 1

    with _engine.begin() as conn:
        conn.execute(text("DELETE FROM users"))
        for row in target_employees:
            conn.execute(
                text(
                    """
                    INSERT INTO users (name, department, salary_grade, remaining_vacation_day)
                    VALUES (:name, :department, :salary_grade, :remaining_vacation_day)
                    """
                ),
                row,
            )

    return len(target_employees)


def run_sql(sql: str) -> list[dict]:
    """Execute a given SELECT statement against MariaDB and return rows as dicts.

    No LLM call here by design: the future agent layer is responsible for
    generating the SQL; this module is a thin, LLM-free data-access tool.
    """
    with _engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result]