import random
import string
import json
import hashlib
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
    """
    Create the physical matrix storage directory
    if it does not already exist.
    """

    MATRIX_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# GET SAFE FILE NAME
# ============================================================

def get_matrix_file(username):
    """
    Create a safe and unique filename for each user.

    The actual username is not used directly as the filename.
    """

    username_hash = hashlib.sha256(
        username.encode("utf-8")
    ).hexdigest()

    return MATRIX_DIRECTORY / f"{username_hash}.json"


# ============================================================
# GENERATE NEW PERMANENT MATRIX
# ============================================================

def create_matrix():
    """
    Generate a permanent 3x4 matrix.

    Each character is unique.
    """

    characters = string.ascii_uppercase + string.digits

    values = random.sample(
        characters,
        MATRIX_ROWS * MATRIX_COLUMNS
    )

    matrix = []

    for row in range(MATRIX_ROWS):
        start = row * MATRIX_COLUMNS
        end = start + MATRIX_COLUMNS

        matrix.append(
            values[start:end]
        )

    return matrix


# ============================================================
# GET OR CREATE USER MATRIX
# ============================================================

def generate_matrix(username):
    """
    Return the permanent matrix assigned to a user.

    If the user already has a matrix, the same matrix
    is returned.

    If the user does not have one, a new matrix is
    created and permanently stored.
    """

    ensure_matrix_directory()

    matrix_file = get_matrix_file(username)

    # --------------------------------------------------------
    # Existing matrix
    # --------------------------------------------------------

    if matrix_file.exists():

        try:
            with open(
                matrix_file,
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            return data["matrix"]

        except (
            json.JSONDecodeError,
            KeyError,
            OSError
        ):
            pass

    # --------------------------------------------------------
    # Create new matrix
    # --------------------------------------------------------

    matrix = create_matrix()

    data = {
        "username": username,
        "rows": MATRIX_ROWS,
        "columns": MATRIX_COLUMNS,
        "matrix": matrix
    }

    with open(
        matrix_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    return matrix


# ============================================================
# MATRIX CHALLENGE
# ============================================================

def generate_challenge():
    """
    Generate a new random challenge.

    The matrix itself remains permanent.

    Only the requested positions change.
    """

    positions = [
        (row, col)
        for row in range(MATRIX_ROWS)
        for col in range(MATRIX_COLUMNS)
    ]

    return random.sample(
        positions,
        CHALLENGE_LENGTH
    )


# ============================================================
# GET CHALLENGE ANSWER
# ============================================================

def get_challenge_answer(matrix, positions):
    """
    Get the characters from the requested positions
    in the exact challenge order.
    """

    return "".join(
        matrix[row][col]
        for row, col in positions
    )


# ============================================================
# VERIFY MATRIX
# ============================================================

def verify_matrix(
    matrix,
    positions,
    entered_answer
):
    """
    Verify the characters entered by the user.
    """

    correct_answer = get_challenge_answer(
        matrix,
        positions
    )

    return (
        entered_answer.strip().upper()
        == correct_answer.upper()
    )