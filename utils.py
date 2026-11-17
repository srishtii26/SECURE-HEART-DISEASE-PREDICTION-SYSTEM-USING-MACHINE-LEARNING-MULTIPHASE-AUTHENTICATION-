import sqlite3
from datetime import datetime
import hashlib
import math


# ============================================================
# DATABASE
# ============================================================

DATABASE = "database/users.db"


# ============================================================
# 1. PATIENT INPUT VALIDATION
# ============================================================

def validate_patient_data(
    age,
    trestbps,
    chol,
    thalch,
    oldpeak,
    ca
):
    """
    Strict validation of patient clinical inputs.

    Validation is performed before the data is sent
    to the machine-learning model.

    Returns:
        (True, None) when the input is valid.
        (False, error_message) when the input is invalid.
    """

    # --------------------------------------------------------
    # STEP 1: Convert inputs to numbers
    # --------------------------------------------------------

    try:
        age = float(age)
        trestbps = float(trestbps)
        chol = float(chol)
        thalch = float(thalch)
        oldpeak = float(oldpeak)
        ca = float(ca)

    except (TypeError, ValueError):
        return (
            False,
            "All clinical measurements must be numeric."
        )

    # --------------------------------------------------------
    # STEP 2: Reject NaN and infinity
    # --------------------------------------------------------

    values = {
        "Age": age,
        "Resting blood pressure": trestbps,
        "Cholesterol": chol,
        "Maximum heart rate": thalch,
        "ST depression": oldpeak,
        "Major vessels": ca
    }

    for name, value in values.items():

        if not math.isfinite(value):

            return (
                False,
                f"{name} contains an invalid value."
            )

    # ========================================================
    # AGE VALIDATION
    # ========================================================

    if age < 1:

        return (
            False,
            "Invalid age: Age must be at least 1 year."
        )

    if age > 120:

        return (
            False,
            "Invalid age: Age cannot be greater than 120 years."
        )

    # Age must be a whole number
    if not age.is_integer():

        return (
            False,
            "Invalid age: Age must be a whole number."
        )

    # Optional realistic clinical restriction
    if age < 18:

        return (
            False,
            "Invalid age: Patient age must be at least 18 years."
        )

    # ========================================================
    # RESTING BLOOD PRESSURE
    # ========================================================

    if trestbps < 70:

        return (
            False,
            "Invalid resting blood pressure: "
            "value must be at least 70 mmHg."
        )

    if trestbps > 220:

        return (
            False,
            "Invalid resting blood pressure: "
            "value cannot exceed 220 mmHg."
        )

    # ========================================================
    # CHOLESTEROL
    # ========================================================

    if chol < 100:

        return (
            False,
            "Invalid cholesterol: "
            "value must be at least 100 mg/dl."
        )

    if chol > 600:

        return (
            False,
            "Invalid cholesterol: "
            "value cannot exceed 600 mg/dl."
        )

    # ========================================================
    # MAXIMUM HEART RATE
    # ========================================================

    if thalch < 60:

        return (
            False,
            "Invalid maximum heart rate: "
            "value must be at least 60 bpm."
        )

    if thalch > 220:

        return (
            False,
            "Invalid maximum heart rate: "
            "value cannot exceed 220 bpm."
        )

    # ========================================================
    # ST DEPRESSION
    # ========================================================

    if oldpeak < 0:

        return (
            False,
            "Invalid ST depression: "
            "value cannot be negative."
        )

    if oldpeak > 10:

        return (
            False,
            "Invalid ST depression: "
            "value cannot exceed 10."
        )

    # ========================================================
    # NUMBER OF MAJOR VESSELS
    # ========================================================

    if ca < 0:

        return (
            False,
            "Invalid number of major vessels: "
            "value cannot be negative."
        )

    if ca > 4:

        return (
            False,
            "Invalid number of major vessels: "
            "value cannot exceed 4."
        )

    # Major vessels must be a whole number
    if not ca.is_integer():

        return (
            False,
            "Invalid number of major vessels: "
            "value must be a whole number from 0 to 4."
        )

    # ========================================================
    # CROSS-FIELD VALIDATION
    # ========================================================

    # Extremely high maximum heart rate for very old patients
    if age >= 100 and thalch > 200:

        return (
            False,
            "The maximum heart rate appears "
            "unrealistic for the entered age."
        )

    # ========================================================
    # ALL VALID
    # ========================================================

    return True, None


# ============================================================
# 2. PREDICTION ACTIVITY LOGGING
# ============================================================

def log_prediction(
    username,
    prediction,
    confidence
):
    """
    Store prediction activity.

    Patient medical information is NOT stored.
    Only username, prediction, confidence and timestamp
    are recorded.
    """

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO prediction_logs
        (username, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
        """,
        (
            username,
            int(prediction),
            float(confidence),
            timestamp
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# 3. MODEL INTEGRITY CHECK
# ============================================================

def calculate_file_hash(file_path):
    """
    Calculate the SHA-256 hash of a file.
    """

    sha256_hash = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as file:

        for chunk in iter(
            lambda: file.read(4096),
            b""
        ):

            sha256_hash.update(
                chunk
            )

    return sha256_hash.hexdigest()


def verify_model_integrity(
    model_path,
    expected_hash
):
    """
    Verify that the machine-learning model
    has not been modified or corrupted.

    Returns:
        True  -> model hash matches
        False -> model hash does not match
                  or model cannot be accessed
    """

    try:

        current_hash = calculate_file_hash(
            model_path
        )

        return (
            current_hash == expected_hash
        )

    except (
        FileNotFoundError,
        OSError
    ):

        return False


# ============================================================
# 4. TRUSTED MODEL HASH
# ============================================================

TRUSTED_MODEL_HASH = (
    "a65c2eed0e97ea42d8fcde85a5fa87c1291310db0507f72f39a8ae08a8c56527"
)