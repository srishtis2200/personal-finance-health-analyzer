"""
database/db_connect.py
Phase 4 — MySQL Database Layer (Updated for Streamlit Cloud)
=============================================================
Works in two environments automatically:
  - Local development  : reads from .env via python-dotenv
  - Streamlit Cloud    : reads from st.secrets (secrets.toml)

No code changes needed when switching environments.
"""

import mysql.connector
from mysql.connector import Error


def _get_credentials() -> dict:
    """
    Load DB credentials from st.secrets (Streamlit Cloud)
    or .env (local development) — automatically detected.
    """
    try:
        # ── Streamlit Cloud: reads from secrets.toml ──────────────────────────
        import streamlit as st
        return {
            "host":     st.secrets["DB_HOST"],
            "user":     st.secrets["DB_USER"],
            "password": st.secrets["DB_PASSWORD"],
            "database": st.secrets["DB_NAME"],
            "port":     int(st.secrets.get("DB_PORT", 3306)),
        }
    except Exception:
        # ── Local development: reads from .env ────────────────────────────────
        import os
        from dotenv import load_dotenv
        load_dotenv()
        return {
            "host":     os.getenv("DB_HOST", "127.0.0.1"),
            "user":     os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", ""),
            "database": os.getenv("DB_NAME", "finance_db"),
            "port":     int(os.getenv("DB_PORT", 3306)),
        }


def get_connection():
    """
    Opens and returns a MySQL connection.
    Uses TCP/IP (use_pure=True) to avoid Windows named pipe issues.
    Works for both local MySQL and Clever Cloud MySQL.
    """
    creds = _get_credentials()
    try:
        conn = mysql.connector.connect(
            host=             creds["host"],
            user=             creds["user"],
            password=         creds["password"],
            database=         creds["database"],
            port=             creds["port"],
            use_pure=         True,       # Force TCP/IP — fixes Windows named pipe bug
            connection_timeout=10,        # Fail fast instead of hanging
            ssl_disabled=     False,      # Clever Cloud requires SSL
        )
        return conn
    except Error as e:
        raise ConnectionError(f"MySQL connection failed: {e}")


def insert_record(user_name: str, month_year: str, input_data: dict, result: dict) -> bool:
    """
    Save one form submission to the database.
    If the user already submitted for this month, overwrites (UPSERT).

    Parameters
    ----------
    user_name  : str   e.g. "Rahul"
    month_year : str   e.g. "2024-12"
    input_data : dict  the 9 raw inputs from the form
    result     : dict  the full output dict from explainer.py

    Returns True on success, False on failure.
    """
    sql = """
        INSERT INTO finance_records (
            user_name, month_year,
            monthly_income, rent, food, emi, transport,
            subscriptions, savings, emergency_fund_months, dependents,
            health_score, risk_category, confidence
        ) VALUES (
            %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            monthly_income        = VALUES(monthly_income),
            rent                  = VALUES(rent),
            food                  = VALUES(food),
            emi                   = VALUES(emi),
            transport             = VALUES(transport),
            subscriptions         = VALUES(subscriptions),
            savings               = VALUES(savings),
            emergency_fund_months = VALUES(emergency_fund_months),
            dependents            = VALUES(dependents),
            health_score          = VALUES(health_score),
            risk_category         = VALUES(risk_category),
            confidence            = VALUES(confidence)
    """
    values = (
        user_name,
        month_year,
        input_data.get("monthly_income"),
        input_data.get("rent"),
        input_data.get("food"),
        input_data.get("emi"),
        input_data.get("transport"),
        input_data.get("subscriptions"),
        input_data.get("savings"),
        input_data.get("emergency_fund_months"),
        input_data.get("dependents"),
        result.get("score"),
        result.get("category"),
        round(result.get("confidence", 0), 4),
    )

    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()
        return True
    except Error as e:
        print(f"[DB] insert_record failed: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


def fetch_user_history(user_name: str) -> list:
    """
    Fetch all monthly records for a user, ordered oldest → newest.
    Used for month-over-month trend charts and Gemini monthly_summary / goal_planning.

    Returns list of dicts, e.g.:
    [
        {"month_year": "2024-10", "health_score": 38, "risk_category": "Critical", ...},
        {"month_year": "2024-11", "health_score": 42, "risk_category": "At Risk",  ...},
    ]
    """
    sql = """
        SELECT
            month_year, health_score, risk_category, confidence,
            monthly_income, rent, food, emi, transport,
            subscriptions, savings, emergency_fund_months, dependents,
            created_at
        FROM finance_records
        WHERE user_name = %s
        ORDER BY month_year ASC
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)   # Returns list of dicts
        cursor.execute(sql, (user_name,))
        return cursor.fetchall()
    except Error as e:
        print(f"[DB] fetch_user_history failed: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            conn.close()


def fetch_latest_record(user_name: str) -> dict | None:
    """
    Fetch only the most recent submission for a user.
    Used to pre-fill the form with last month's values.

    Returns a dict or None if no records exist.
    """
    sql = """
        SELECT *
        FROM finance_records
        WHERE user_name = %s
        ORDER BY month_year DESC
        LIMIT 1
    """
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, (user_name,))
        return cursor.fetchone()
    except Error as e:
        print(f"[DB] fetch_latest_record failed: {e}")
        return None
    finally:
        if conn and conn.is_connected():
            conn.close()


def fetch_all_users() -> list:
    """
    Fetch list of all unique user names.
    Used to populate the user dropdown on the History page.

    Returns list of strings, e.g. ["Rahul", "Priya", "Amit"]
    """
    sql = "SELECT DISTINCT user_name FROM finance_records ORDER BY user_name ASC"
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql)
        return [row[0] for row in cursor.fetchall()]
    except Error as e:
        print(f"[DB] fetch_all_users failed: {e}")
        return []
    finally:
        if conn and conn.is_connected():
            conn.close()


def delete_record(user_name: str, month_year: str) -> bool:
    """
    Delete a specific record. For testing/admin only.

    Returns True on success, False on failure.
    """
    sql = "DELETE FROM finance_records WHERE user_name = %s AND month_year = %s"
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(sql, (user_name, month_year))
        conn.commit()
        return cursor.rowcount > 0
    except Error as e:
        print(f"[DB] delete_record failed: {e}")
        return False
    finally:
        if conn and conn.is_connected():
            conn.close()


def test_connection() -> bool:
    """
    Quick health check — returns True if DB is reachable.
    Useful for debugging deployment issues.
    """
    try:
        conn = get_connection()
        if conn.is_connected():
            conn.close()
            return True
    except Exception as e:
        print(f"[DB] Connection test failed: {e}")
    return False