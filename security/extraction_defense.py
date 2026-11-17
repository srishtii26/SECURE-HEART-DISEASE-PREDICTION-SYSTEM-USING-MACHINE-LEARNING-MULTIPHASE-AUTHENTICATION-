from datetime import datetime, timedelta
import sqlite3


DATABASE = "database/users.db"


# Maximum number of predictions allowed
# within the monitoring window.
MAX_QUERIES = 20

WINDOW_MINUTES = 5


def get_recent_query_count(username):
    """
    Count prediction queries made by a user
    during the recent monitoring window.
    """

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cutoff_time = (
        datetime.now()
        - timedelta(
            minutes=WINDOW_MINUTES
        )
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM prediction_logs
        WHERE username = ?
        AND timestamp >= ?
        """,
        (
            username,
            cutoff_time
        )
    )

    count = cursor.fetchone()[0]

    connection.close()

    return count


def check_extraction_risk(username):
    """
    Detect unusually high prediction-query activity.
    """

    query_count = get_recent_query_count(
        username
    )

    if query_count >= MAX_QUERIES:

        return {
            "allowed": False,
            "suspicious": True,
            "query_count": query_count,
            "message":
                "Prediction query limit exceeded. "
                "Possible model extraction activity detected."
        }

    return {
        "allowed": True,
        "suspicious": False,
        "query_count": query_count,
        "message":
            "Prediction query activity is within "
            "the permitted limit."
    }