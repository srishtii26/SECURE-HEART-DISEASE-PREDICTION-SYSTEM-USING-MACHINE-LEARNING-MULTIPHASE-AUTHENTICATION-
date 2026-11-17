import sqlite3

DATABASE = "database/users.db"

connection = sqlite3.connect(DATABASE)
cursor = connection.cursor()

print("\n========== USERS TABLE STRUCTURE ==========")

columns = cursor.execute(
    "PRAGMA table_info(users)"
).fetchall()

for column in columns:
    print(column)

print("\n========== EXISTING ACCOUNTS ==========")

users = cursor.execute(
    "SELECT username, role, whatsapp_number FROM users"
).fetchall()

for user in users:
    print(user)

connection.close()