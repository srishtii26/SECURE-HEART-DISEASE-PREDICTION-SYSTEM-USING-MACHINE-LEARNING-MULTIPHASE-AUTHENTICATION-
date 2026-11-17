import sqlite3


DATABASE = "database/users.db"


# ==========================================
# CREATE USERS TABLE
# ==========================================

def create_users_table():
    """Create the users table."""

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# ==========================================
# CREATE PREDICTION LOGS TABLE
# ==========================================

def create_prediction_logs_table():
    """Create the prediction activity log table."""

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# ==========================================
# CREATE LOGIN LOGS TABLE
# ==========================================

def create_login_logs_table():
    """Create the login activity log table."""

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


# ==========================================
# GET LOGIN LOGS
# ==========================================

def get_login_logs():
    """Retrieve login activity logs."""

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT username, status, timestamp
        FROM login_logs
        ORDER BY timestamp DESC
    """)

    logs = cursor.fetchall()

    connection.close()

    return logs


# ==========================================
# GET PREDICTION LOGS
# ==========================================

def get_prediction_logs():
    """Retrieve prediction activity logs."""

    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT username, prediction, confidence, timestamp
        FROM prediction_logs
        ORDER BY timestamp DESC
    """)

    logs = cursor.fetchall()

    connection.close()

    return logs


# ==========================================
# DATABASE SETUP
# ==========================================

if __name__ == "__main__":

    create_users_table()

    create_prediction_logs_table()

    create_login_logs_table()

    print("Database setup completed successfully!")