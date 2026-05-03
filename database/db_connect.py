# =============================================================
# Personal Finance Health Analyzer — Phase 4
# File: database/db_connect.py
# Handles all MySQL interactions: connect, insert, fetch
# =============================================================

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from datetime import datetime

# IMPORTANT: load_dotenv() must be called BEFORE any os.getenv()
load_dotenv()


# -------------------------------------------------------------
# 1. Connection helper
# -------------------------------------------------------------

def get_connection():
    """
    Creates and returns a MySQL connection using .env credentials.
    Always call .close() on the returned connection when done,
    or use it as a context manager.

    Returns:
        mysql.connector.connection.MySQLConnection | None
    """
    try:
        conn = mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "finance_db"),
        use_pure=True,
        connection_timeout=10,
        autocommit=False
        )
        
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"[DB ERROR] Could not connect to MySQL: {e}")
        return None


# -------------------------------------------------------------
# 2. Insert a new finance record
# -------------------------------------------------------------

def insert_record(user_name: str, month_year: str, inputs: dict, result: dict) -> bool:
    """
    Inserts one submission into finance_records.

    Args:
        user_name  : str  — the name entered in the Streamlit form
        month_year : str  — e.g. "May-2025"  (format: "Mon-YYYY")
        inputs     : dict — the 9 raw features from the form
                     Keys: monthly_income, rent, food, emi, transport,
                           subscriptions, savings, emergency_fund, dependents
        result     : dict — the full dict returned by explainer.explain()
                     Must contain: score, category, confidence, probabilities
                     probabilities keys: 'At Risk', 'Critical', 'Stable'

    Returns:
        True on success, False on failure.
    """
    conn = get_connection()
    if conn is None:
        return False

    sql = """
        INSERT INTO finance_records (
            user_name, month_year,
            monthly_income, rent, food, emi, transport,
            subscriptions, savings, emergency_fund, dependents,
            health_score, category, confidence,
            prob_stable, prob_at_risk, prob_critical
        ) VALUES (
            %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            monthly_income  = VALUES(monthly_income),
            rent            = VALUES(rent),
            food            = VALUES(food),
            emi             = VALUES(emi),
            transport       = VALUES(transport),
            subscriptions   = VALUES(subscriptions),
            savings         = VALUES(savings),
            emergency_fund  = VALUES(emergency_fund),
            dependents      = VALUES(dependents),
            health_score    = VALUES(health_score),
            category        = VALUES(category),
            confidence      = VALUES(confidence),
            prob_stable     = VALUES(prob_stable),
            prob_at_risk    = VALUES(prob_at_risk),
            prob_critical   = VALUES(prob_critical),
            created_at      = CURRENT_TIMESTAMP
    """
    # ON DUPLICATE KEY UPDATE allows re-submitting the same month
    # (overwrites previous entry for that user+month)

    probs = result.get("probabilities", {})

    values = (
        user_name.strip(),
        month_year.strip(),
        float(inputs["monthly_income"]),
        float(inputs["rent"]),
        float(inputs["food"]),
        float(inputs["emi"]),
        float(inputs["transport"]),
        float(inputs["subscriptions"]),
        float(inputs["savings"]),
        float(inputs["emergency_fund"]),
        int(inputs["dependents"]),
        int(result["score"]),
        result["category"],
        float(result["confidence"]),
        float(probs.get("Stable", 0.0)),
        float(probs.get("At Risk", 0.0)),
        float(probs.get("Critical", 0.0)),
    )

    try:
        cursor = conn.cursor()
        cursor.execute(sql, values)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[DB] Record saved — {user_name} / {month_year} / Score: {result['score']}")
        return True
    except Error as e:
        print(f"[DB ERROR] insert_record failed: {e}")
        conn.rollback()
        conn.close()
        return False


# -------------------------------------------------------------
# 3. Fetch full history for a user (for History page chart)
# -------------------------------------------------------------

def fetch_user_history(user_name: str) -> list[dict]:
    """
    Returns all records for a user, ordered by created_at ascending.
    Used by the Phase 6 History page to plot month-over-month trend.

    Args:
        user_name : str — exact match on user_name column

    Returns:
        List of dicts, each dict = one row.
        Empty list if no records found or on error.
    """
    conn = get_connection()
    if conn is None:
        return []

    sql = """
        SELECT
            id, user_name, month_year,
            monthly_income, rent, food, emi, transport,
            subscriptions, savings, emergency_fund, dependents,
            health_score, category, confidence,
            prob_stable, prob_at_risk, prob_critical,
            created_at
        FROM finance_records
        WHERE user_name = %s
        ORDER BY created_at ASC
    """

    try:
        cursor = conn.cursor(dictionary=True)   # returns rows as dicts
        cursor.execute(sql, (user_name.strip(),))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Error as e:
        print(f"[DB ERROR] fetch_user_history failed: {e}")
        conn.close()
        return []


# -------------------------------------------------------------
# 4. Fetch latest record for a user
# -------------------------------------------------------------

def fetch_latest_record(user_name: str) -> dict | None:
    """
    Returns the single most recent record for a user.
    Useful for showing "last submission" summary on the dashboard.

    Args:
        user_name : str

    Returns:
        A single dict (one row), or None if not found / error.
    """
    conn = get_connection()
    if conn is None:
        return None

    sql = """
        SELECT *
        FROM finance_records
        WHERE user_name = %s
        ORDER BY created_at DESC
        LIMIT 1
    """

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, (user_name.strip(),))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row    # None if no records exist
    except Error as e:
        print(f"[DB ERROR] fetch_latest_record failed: {e}")
        conn.close()
        return None


# -------------------------------------------------------------
# 5. Fetch all unique user names (for user selector dropdown)
# -------------------------------------------------------------

def fetch_all_users() -> list[str]:
    """
    Returns a sorted list of all distinct user_name values.
    Used to populate a user selector in Streamlit sidebar.

    Returns:
        List of strings. Empty list on error.
    """
    conn = get_connection()
    if conn is None:
        return []

    sql = "SELECT DISTINCT user_name FROM finance_records ORDER BY user_name ASC"

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in rows]
    except Error as e:
        print(f"[DB ERROR] fetch_all_users failed: {e}")
        conn.close()
        return []


# -------------------------------------------------------------
# 6. Delete a record (optional — useful for testing)
# -------------------------------------------------------------

def delete_record(user_name: str, month_year: str) -> bool:
    """
    Deletes a specific user+month record.
    Only useful during testing — not exposed in the UI.

    Returns:
        True on success, False on failure.
    """
    conn = get_connection()
    if conn is None:
        return False

    sql = "DELETE FROM finance_records WHERE user_name = %s AND month_year = %s"

    try:
        cursor = conn.cursor()
        cursor.execute(sql, (user_name.strip(), month_year.strip()))
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
        conn.close()
        print(f"[DB] Deleted {affected} record(s) for {user_name} / {month_year}")
        return affected > 0
    except Error as e:
        print(f"[DB ERROR] delete_record failed: {e}")
        conn.rollback()
        conn.close()
        return False


# -------------------------------------------------------------
# 7. Quick connection test — run this file directly to verify
# -------------------------------------------------------------

def test_connection():
    """
    Run `python database/db_connect.py` to verify MySQL is reachable.
    """
    conn = get_connection()
    if conn:
        print("[DB] ✅ Connection successful!")
        print(f"     Host     : {os.getenv('DB_HOST', 'localhost')}")
        print(f"     Database : {os.getenv('DB_NAME', 'finance_db')}")
        print(f"     User     : {os.getenv('DB_USER', 'root')}")
        conn.close()
    else:
        print("[DB] ❌ Connection FAILED. Check:")
        print("     1. MySQL service is running (Services > MySQL80)")
        print("     2. .env file exists with DB_HOST, DB_USER, DB_PASSWORD, DB_NAME")
        print("     3. load_dotenv() is being called (already done in this file)")
        print("     4. Database 'finance_db' exists (run schema.sql first)")


if __name__ == "__main__":
    test_connection()