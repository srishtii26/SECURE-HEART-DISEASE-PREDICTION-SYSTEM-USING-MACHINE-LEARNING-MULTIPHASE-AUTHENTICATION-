import secrets
import string
import json
import hashlib
import os
from pathlib import Path


# ============================================================
# MATRIX CONFIGURATION
# ============================================================

MATRIX_ROWS = 3
MATRIX_COLUMNS = 4
CHALLENGE_LENGTH = 3

MATRIX_DIRECTORY = Path("database") / "matrices"


# ============================================================
# CREATE MATRIX DIRECTORY
# ============================================================

def ensure_matrix_directory():
    MATRIX_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# GET SAFE FILE NAME
# ============================================================

def get_matrix_file(username):

    username_hash = hashlib.sha256(
        username.encode("utf-8")
    ).hexdigest()

    return MATRIX_DIRECTORY / f"{username_hash}.json"


# ============================================================
# GENERATE NEW PERMANENT MATRIX
# ============================================================

def create_matrix():

    characters = string.ascii_uppercase + string.digits

    values = list(characters)

    # Cryptographically secure randomization
    secrets.SystemRandom().shuffle(values)

    values = values[:MATRIX_ROWS * MATRIX_COLUMNS]

    matrix = []

    for row in range(MATRIX_ROWS):

        start = row * MATRIX_COLUMNS
        end = start + MATRIX_COLUMNS

        matrix.append(values[start:end])

    return matrix


# ============================================================
# LOAD MATRIX DATA
# ============================================================

def load_matrix_data(username):

    matrix_file = get_matrix_file(username)

    if not matrix_file.exists():
        return None

    try:
        with open(
            matrix_file,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        if not isinstance(data.get("matrix"), list):
            raise ValueError("Invalid matrix data.")

        return data

    except (json.JSONDecodeError, OSError, ValueError) as error:
        raise RuntimeError(
            "Existing matrix could not be loaded. "
            "Matrix regeneration has been blocked."
        ) from error


# ============================================================
# SAVE MATRIX DATA
# ============================================================

def save_matrix_data(username, data):

    ensure_matrix_directory()

    matrix_file = get_matrix_file(username)

    temporary_file = matrix_file.with_suffix(".tmp")

    with open(
        temporary_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=4)

    os.replace(temporary_file, matrix_file)


# ============================================================
# GET OR CREATE PERMANENT MATRIX
# ============================================================

def generate_matrix(username):

    ensure_matrix_directory()

    existing_data = load_matrix_data(username)

    # Return existing permanent matrix
    if existing_data is not None:
        return existing_data["matrix"]

    # Generate matrix only if none exists
    matrix = create_matrix()

    data = {
        "username": username,
        "rows": MATRIX_ROWS,
        "columns": MATRIX_COLUMNS,
        "matrix": matrix,
        "digital_matrix_sent": False,
        "physical_card_received": False
    }

    matrix_file = get_matrix_file(username)

    # Exclusive creation prevents overwriting an existing matrix
    try:
        with open(
            matrix_file,
            "x",
            encoding="utf-8"
        ) as file:
            json.dump(data, file, indent=4)

    except FileExistsError:
        # Another process created the matrix first
        existing_data = load_matrix_data(username)
        return existing_data["matrix"]

    return matrix


# ============================================================
# CHECK DIGITAL MATRIX DELIVERY
# ============================================================

def is_digital_matrix_sent(username):

    data = load_matrix_data(username)

    if data is None:
        return False

    return data.get("digital_matrix_sent", False)


# ============================================================
# MARK DIGITAL MATRIX AS SENT
# ============================================================

def mark_digital_matrix_sent(username):

    data = load_matrix_data(username)

    if data is None:
        raise ValueError("Matrix does not exist.")

    data["digital_matrix_sent"] = True

    save_matrix_data(username, data)


# ============================================================
# CHECK PHYSICAL CARD STATUS
# ============================================================

def is_physical_card_received(username):

    data = load_matrix_data(username)

    if data is None:
        return False

    return data.get("physical_card_received", False)


# ============================================================
# MARK PHYSICAL CARD AS RECEIVED
# ============================================================

def mark_physical_card_received(username):

    data = load_matrix_data(username)

    if data is None:
        raise ValueError("Matrix does not exist.")

    data["physical_card_received"] = True

    save_matrix_data(username, data)


# ============================================================
# FORMAT MATRIX FOR DIGITAL DELIVERY
# ============================================================

def format_matrix_message(username, matrix):

    message = f"""
SECURE HEART DISEASE PREDICTION SYSTEM

Hello {username},

Your permanent authentication matrix:

        1   2   3   4
    A   {matrix[0][0]}   {matrix[0][1]}   {matrix[0][2]}   {matrix[0][3]}
    B   {matrix[1][0]}   {matrix[1][1]}   {matrix[1][2]}   {matrix[1][3]}
    C   {matrix[2][0]}   {matrix[2][1]}   {matrix[2][2]}   {matrix[2][3]}

IMPORTANT:

1. This is your permanent matrix.
2. Your physical card will contain the SAME matrix.
3. Keep this matrix confidential.
4. It will not be sent automatically again.
5. Use this matrix during authentication.
"""

    return message.strip()


# ============================================================
# MATRIX CHALLENGE
# ============================================================

def generate_challenge():

    positions = [
        (row, col)
        for row in range(MATRIX_ROWS)
        for col in range(MATRIX_COLUMNS)
    ]

    return secrets.SystemRandom().sample(
        positions,
        CHALLENGE_LENGTH
    )


# ============================================================
# GET CHALLENGE ANSWER
# ============================================================

def get_challenge_answer(matrix, positions):

    return "".join(
        matrix[row][col]
        for row, col in positions
    )


# ============================================================
# VERIFY MATRIX
# ============================================================

def verify_matrix(matrix, positions, entered_answer):

    correct_answer = get_challenge_answer(
        matrix,
        positions
    )

    return secrets.compare_digest(
        entered_answer.strip().upper(),
        correct_answer.upper()
    )
    # ============================================================
# CHECK MATRIX ACKNOWLEDGEMENT
# ============================================================

def is_matrix_acknowledged(username):

    data = load_matrix_data(username)

    if data is None:
        return False

    return data.get("matrix_acknowledged", False)


# ============================================================
# MARK MATRIX AS ACKNOWLEDGED
# ============================================================

def mark_matrix_acknowledged(username):

    data = load_matrix_data(username)

    if data is None:
        raise ValueError("Matrix does not exist.")

    if data.get("matrix_acknowledged", False):
        return

    data["matrix_acknowledged"] = True

    save_matrix_data(username, data)