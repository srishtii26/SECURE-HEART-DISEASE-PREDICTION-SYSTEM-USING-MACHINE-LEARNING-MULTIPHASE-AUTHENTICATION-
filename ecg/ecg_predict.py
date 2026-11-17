from pathlib import Path
import json

import numpy as np
from PIL import Image
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "ecg" / "ecg_model_v2.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "ecg" / "class_names.json"


# ============================================================
# LOAD MODEL
# ============================================================

_model = None
_class_names = None


def load_ecg_model():
    """
    Load the trained ECG image classification model.
    The model is loaded only once.
    """

    global _model

    if _model is None:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"ECG model not found: {MODEL_PATH}"
            )

        _model = tf.keras.models.load_model(MODEL_PATH)

    return _model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    """
    Load ECG class names from class_names.json.
    """

    global _class_names

    if _class_names is None:

        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(
                f"Class mapping not found: {CLASS_NAMES_PATH}"
            )

        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            _class_names = json.load(file)

    return _class_names


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_ecg_image(image):
    """
    Convert uploaded ECG image into the format
    expected by the CNN model.

    Expected model input:
        224 x 224 x 3
    """

    if image is None:
        raise ValueError("No ECG image was provided.")

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((224, 224))

    # Convert to NumPy
    image_array = np.asarray(image, dtype=np.float32)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ============================================================
# PREDICT ECG
# ============================================================

def predict_ecg(image):
    """
    Predict the ECG category.

    Returns:
        predicted_class
        class_name
        confidence
        probabilities
    """

    model = load_ecg_model()
    class_names = load_class_names()

    processed_image = preprocess_ecg_image(image)

    # Model prediction
    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    # Highest probability class
    predicted_class = int(np.argmax(probabilities))

    confidence = float(probabilities[predicted_class])

    class_name = class_names[predicted_class]

    return {
        "predicted_class": predicted_class,
        "class_name": class_name,
        "confidence": confidence,
        "probabilities": probabilities.tolist(),
    }


# ============================================================
# USER-FRIENDLY CLASS NAME
# ============================================================

def get_short_class_name(class_name):
    """
    Convert the dataset's long class names into
    clean names for the Streamlit UI.
    """

    if "Myocardial Infarction Patients" in class_name:
        return "Myocardial Infarction"

    if "History of MI" in class_name:
        return "History of MI"

    if "abnormal heartbeat" in class_name:
        return "Abnormal Heartbeat"

    if "Normal Person" in class_name:
        return "Normal ECG"

    return class_name


# ============================================================
# HEALTH INTERPRETATION
# ============================================================

def get_ecg_interpretation(class_name, confidence):
    """
    Generate a UI-friendly interpretation.

    IMPORTANT:
    This is a machine-learning classification result,
    not a medical diagnosis.
    """

    short_name = get_short_class_name(class_name)

    if short_name == "Normal ECG":

        message = (
            "The ECG image was classified as Normal ECG "
            "by the trained image classification model."
        )

        severity = "NORMAL"

    elif short_name == "Myocardial Infarction":

        message = (
            "The ECG image was classified as showing "
            "patterns associated with Myocardial Infarction."
        )

        severity = "ATTENTION"

    elif short_name == "History of MI":

        message = (
            "The ECG image was classified as showing "
            "patterns associated with a History of MI."
        )

        severity = "ATTENTION"

    elif short_name == "Abnormal Heartbeat":

        message = (
            "The ECG image was classified as showing "
            "patterns associated with an Abnormal Heartbeat."
        )

        severity = "ATTENTION"

    else:

        message = (
            "The ECG image classification result requires "
            "further review."
        )

        severity = "REVIEW"

    confidence_text = f"{confidence * 100:.1f}%"

    return {
        "label": short_name,
        "message": message,
        "severity": severity,
        "confidence_text": confidence_text,
    }