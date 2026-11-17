import sqlite3
import bcrypt
import os


# ============================================================
# DATABASE
# ============================================================

DATABASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "database",
    "users.db"
)


# ============================================================
# USER DETAILS
# ============================================================

users = [
    ("user1", "user1@123", "9606913791"),
    ("user2", "user2@123", "7483378529"),
    ("user3", "user3@123", "8310813185"),
    ("user4", "user4@123", "9606913791"),
    ("user5", "user5@123", "7483378529"),
    ("user6", "user6@123", "8310813185"),
    ("user7", "user7@123", "7483378529"),
]


# ============================================================
# DATABASE CONNECTION
# ============================================================

connection = sqlite3.connect(DATABASE)
cursor = connection.cursor()


# ============================================================
# CREATE / UPDATE USERS
# ============================================================

print("\n========== UPDATING USER ACCOUNTS ==========\n")

for username, password, whatsapp_number in users:

    # Check whether user exists
    cursor.execute(
        """
        SELECT username
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    existing_user = cursor.fetchone()

    # Create bcrypt password hash
    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    if existing_user:

        # Existing user:
        # Update password and WhatsApp number
        cursor.execute(
            """
            UPDATE users
            SET password = ?,
                whatsapp_number = ?,
                role = 'user'
            WHERE username = ?
            """,
            (
                hashed_password,
                whatsapp_number,
                username
            )
        )

        print(
            f"✅ {username} updated"
        )

    else:

        # New user
        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                password,
                role,
                whatsapp_number
            )
            VALUES (?, ?, 'user', ?)
            """,
            (
                username,
                hashed_password,
                whatsapp_number
            )
        )

        print(
            f"✅ {username} created"
        )


# ============================================================
# SAVE
# ============================================================

connection.commit()


# ============================================================
# DISPLAY ACCOUNTS
# ============================================================

print("\n========== ALL ACCOUNTS ==========\n")

cursor.execute(
    """
    SELECT
        id,
        username,
        role,
        whatsapp_number
    FROM users
    ORDER BY id
    """
)

for user in cursor.fetchall():
    print(user)


# ============================================================
# CLOSE
# ============================================================

connection.close()

print("\n========================================")
print("✅ USER ACCOUNTS UPDATED SUCCESSFULLY")
print("========================================")