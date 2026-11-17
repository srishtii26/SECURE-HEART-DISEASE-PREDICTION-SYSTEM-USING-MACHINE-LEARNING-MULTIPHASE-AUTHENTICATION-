import sqlite3
import bcrypt

DATABASE = "database/users.db"

username = "user1"
new_password = "user123"

connection = sqlite3.connect(DATABASE)
cursor = connection.cursor()

hashed_password = bcrypt.hashpw(
    new_password.encode("utf-8"),
    bcrypt.gensalt()
).decode("utf-8")

cursor.execute(
    """
    UPDATE users
    SET password = ?, role = 'user'
    WHERE username = ?
    """,
    (hashed_password, username)
)

connection.commit()

if cursor.rowcount == 1:
    print("User password reset successfully!")
    print("Username:", username)
    print("Password:", new_password)
    print("Role: user")
else:
    print("User account not found.")

connection.close()