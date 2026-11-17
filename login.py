import sqlite3
import bcrypt
import os


# ============================================================
# DATABASE PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATABASE = os.path.join(
    PROJECT_ROOT,
    "database",
    "users.db"
)


# ============================================================
# REGISTER USER
# ============================================================

def register_user(
    username,
    password,
    role="user",
    whatsapp_number=None
):
    """
    Register a new user with a securely hashed password.

    IMPORTANT:
    Public registration can ONLY create normal user accounts.
    Admin accounts cannot be created through registration.
    """

    username = str(username).strip()
    password = str(password)

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if not username:
        return False, "Username cannot be empty."

    if len(username) < 4:
        return False, "Username must contain at least 4 characters."

    if not password:
        return False, "Password cannot be empty."

    if len(password) < 6:
        return False, "Password must contain at least 6 characters."

    # --------------------------------------------------------
    # WHATSAPP NUMBER
    # --------------------------------------------------------

    if whatsapp_number is not None:
        whatsapp_number = str(
            whatsapp_number
        ).strip()

        if whatsapp_number == "":
            whatsapp_number = None

    # --------------------------------------------------------
    # FORCE NORMAL USER ROLE
    # --------------------------------------------------------

    # Never trust the role supplied by the registration form.
    role = "user"

    # --------------------------------------------------------
    # DATABASE CONNECTION
    # --------------------------------------------------------

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    try:

        # ----------------------------------------------------
        # CHECK EXISTING USERNAME
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT username
            FROM users
            WHERE username = ?
            """,
            (username,)
        )

        existing_user = cursor.fetchone()

        if existing_user is not None:

            return (
                False,
                "Username already exists. Please choose another."
            )

        # ----------------------------------------------------
        # HASH PASSWORD
        # ----------------------------------------------------

        password_bytes = password.encode(
            "utf-8"
        )

        hashed_password = bcrypt.hashpw(
            password_bytes,
            bcrypt.gensalt()
        )

        # ----------------------------------------------------
        # CREATE USER
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO users
            (
                username,
                password,
                role,
                whatsapp_number
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                hashed_password.decode("utf-8"),
                role,
                whatsapp_number
            )
        )

        connection.commit()

        return (
            True,
            "User registered successfully."
        )

    except sqlite3.IntegrityError:

        connection.rollback()

        return (
            False,
            "Username already exists."
        )

    except sqlite3.Error as error:

        connection.rollback()

        return (
            False,
            f"Database error: {error}"
        )

    finally:

        connection.close()


# ============================================================
# LOGIN ACTIVITY LOGGER
# ============================================================

def log_login(
    username,
    status
):
    """
    Store login activity.

    Passwords are never stored in login logs.
    """

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO login_logs
            (
                username,
                status,
                timestamp
            )
            VALUES (
                ?,
                ?,
                datetime('now', 'localtime')
            )
            """,
            (
                username,
                status
            )
        )

        connection.commit()

    except sqlite3.Error:

        connection.rollback()

    finally:

        connection.close()


# ============================================================
# LOGIN USER
# ============================================================

def login_user(
    username,
    password
):
    """
    Verify username and password.

    Returns:

        True, role, whatsapp_number

    or:

        False, None, None
    """

    username = str(
        username
    ).strip()

    password = str(
        password
    )

    # --------------------------------------------------------
    # BASIC INPUT CHECK
    # --------------------------------------------------------

    if not username or not password:

        if username:

            log_login(
                username,
                "FAILED"
            )

        return (
            False,
            None,
            None
        )

    # --------------------------------------------------------
    # CONNECT TO DATABASE
    # --------------------------------------------------------

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT
                username,
                password,
                role,
                whatsapp_number
            FROM users
            WHERE username = ?
            """,
            (username,)
        )

        user = cursor.fetchone()

    except sqlite3.Error:

        return (
            False,
            None,
            None
        )

    finally:

        connection.close()

    # --------------------------------------------------------
    # USER DOES NOT EXIST
    # --------------------------------------------------------

    if user is None:

        log_login(
            username,
            "FAILED"
        )

        return (
            False,
            None,
            None
        )

    # --------------------------------------------------------
    # GET USER DETAILS
    # --------------------------------------------------------

    stored_username = user[0]
    stored_password = user[1]
    role = user[2]
    whatsapp_number = user[3]

    # --------------------------------------------------------
    # CHECK BCRYPT PASSWORD
    # --------------------------------------------------------

    try:

        password_correct = bcrypt.checkpw(
            password.encode("utf-8"),
            stored_password.encode("utf-8")
        )

    except (
        ValueError,
        TypeError
    ):

        password_correct = False

    # --------------------------------------------------------
    # SUCCESSFUL LOGIN
    # --------------------------------------------------------

    if password_correct:

        log_login(
            stored_username,
            "SUCCESS"
        )

        return (
            True,
            role,
            whatsapp_number
        )

    # --------------------------------------------------------
    # FAILED LOGIN
    # --------------------------------------------------------

    log_login(
        stored_username,
        "FAILED"
    )

    return (
        False,
        None,
        None
    )


# ============================================================
# TEST LOGIN
# ============================================================

if __name__ == "__main__":

    print(
        "\n================================"
    )

    print(
        "       SECURE HEART LOGIN"
    )

    print(
        "================================\n"
    )

    username = input(
        "Enter username: "
    ).strip()

    password = input(
        "Enter password: "
    )

    success, role, whatsapp_number = login_user(
        username,
        password
    )

    if success:

        print(
            "\nLogin successful!"
        )

        print(
            "Username:",
            username
        )

        print(
            "Role:",
            role
        )

        print(
            "WhatsApp:",
            whatsapp_number
        )

    else:

        print(
            "\nInvalid username or password."
        )