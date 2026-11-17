import os
import joblib
import pandas as pd
import numpy as np

from utils import (
    verify_model_integrity,
    TRUSTED_MODEL_HASH,
    validate_patient_data
)


# ============================================================
# MODEL PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "heart_model.pkl"
)


# ============================================================
# MODEL FEATURES
# THESE MUST MATCH train_model.py
# ============================================================

MODEL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca",

    "sex_Male",

    "cp_atypical angina",
    "cp_non-anginal",
    "cp_typical angina",

    "fbs_True",

    "restecg_normal",
    "restecg_st-t abnormality",

    "exang_True",

    "slope_flat",
    "slope_upsloping",

    "thal_normal",
    "thal_reversable defect"
]


# ============================================================
# RAW INPUT FEATURES
# ============================================================

RAW_FEATURES = [
    "age",
    "sex",
    "dataset",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "MODEL SECURITY ERROR: "
            f"Model not found at {MODEL_PATH}"
        )

    # Verify SHA-256 integrity
    is_valid = verify_model_integrity(
        MODEL_PATH,
        TRUSTED_MODEL_HASH
    )

    if not is_valid:

        raise RuntimeError(
            "MODEL SECURITY ERROR: "
            "The ML model has been modified, "
            "corrupted, or replaced."
        )

    model = joblib.load(
        MODEL_PATH
    )

    return model


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(patient_data):

    # Required raw fields
    for field in RAW_FEATURES:

        if field not in patient_data:

            raise ValueError(
                f"Missing required input: {field}"
            )

    try:

        age = float(
            patient_data["age"]
        )

        trestbps = float(
            patient_data["trestbps"]
        )

        chol = float(
            patient_data["chol"]
        )

        thalch = float(
            patient_data["thalch"]
        )

        oldpeak = float(
            patient_data["oldpeak"]
        )

        ca = float(
            patient_data["ca"]
        )

    except (TypeError, ValueError):

        raise ValueError(
            "Invalid numerical patient input."
        )

    valid, message = validate_patient_data(
        age,
        trestbps,
        chol,
        thalch,
        oldpeak,
        ca
    )

    if not valid:

        raise ValueError(
            f"INPUT SECURITY ERROR: {message}"
        )

    return True


# ============================================================
# PREPARE PATIENT FOR MODEL
# ============================================================

def prepare_patient_for_model(patient_data):

    """
    Convert the application's raw patient input
    into the exact 18 features used during training.
    """

    patient = dict(patient_data)

    # Start with numeric features
    prepared = {

        "age": float(patient["age"]),

        "trestbps": float(
            patient["trestbps"]
        ),

        "chol": float(
            patient["chol"]
        ),

        "thalch": float(
            patient["thalch"]
        ),

        "oldpeak": float(
            patient["oldpeak"]
        ),

        "ca": float(
            patient["ca"]
        )
    }


    # ========================================================
    # SEX
    # ========================================================

    sex = str(
        patient.get("sex", "")
    ).strip().lower()

    prepared["sex_Male"] = (
        sex == "male"
    )


    # ========================================================
    # CHEST PAIN
    # ========================================================

    cp = str(
        patient.get("cp", "")
    ).strip().lower()

    prepared["cp_atypical angina"] = (
        cp == "atypical angina"
    )

    prepared["cp_non-anginal"] = (
        cp == "non-anginal"
    )

    prepared["cp_typical angina"] = (
        cp == "typical angina"
    )


    # ========================================================
    # FASTING BLOOD SUGAR
    # ========================================================

    fbs = patient.get(
        "fbs",
        False
    )

    if isinstance(fbs, str):

        fbs = fbs.strip().lower() in [
            "true",
            "1",
            "yes"
        ]

    prepared["fbs_True"] = bool(fbs)


    # ========================================================
    # RESTING ECG
    # ========================================================

    restecg = str(
        patient.get("restecg", "")
    ).strip().lower()

    prepared["restecg_normal"] = (
        restecg == "normal"
    )

    prepared["restecg_st-t abnormality"] = (
        restecg == "st-t abnormality"
    )


    # ========================================================
    # EXERCISE ANGINA
    # ========================================================

    exang = patient.get(
        "exang",
        False
    )

    if isinstance(exang, str):

        exang = exang.strip().lower() in [
            "true",
            "1",
            "yes"
        ]

    prepared["exang_True"] = bool(
        exang
    )


    # ========================================================
    # SLOPE
    # ========================================================

    slope = str(
        patient.get("slope", "")
    ).strip().lower()

    prepared["slope_flat"] = (
        slope == "flat"
    )

    prepared["slope_upsloping"] = (
        slope == "upsloping"
    )


    # ========================================================
    # THAL
    # ========================================================

    thal = str(
        patient.get("thal", "")
    ).strip().lower()

    prepared["thal_normal"] = (
        thal == "normal"
    )

    prepared["thal_reversable defect"] = (
        thal == "reversable defect"
    )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        [prepared]
    )

    # IMPORTANT:
    # Force exact same order as training
    df = df.reindex(
        columns=MODEL_FEATURES,
        fill_value=False
    )

    return df


# ============================================================
# ADVERSARIAL STABILITY CHECK
# ============================================================

def check_adversarial_stability(
    model,
    patient_data
):

    try:

        from security.adversarial_defense import (
            check_prediction_stability
        )

        result = check_prediction_stability(
            model,
            patient_data
        )

        return result

    except ImportError:

        return {
            "stable": True,
            "original_prediction": None,
            "changed_predictions": []
        }


# ============================================================
# PRIVACY-PRESERVING CONFIDENCE
# ============================================================

def privatize_confidence(
    confidence,
    epsilon=5.0
):

    if epsilon <= 0:

        raise ValueError(
            "Privacy epsilon must be greater than zero."
        )

    confidence_value = (
        confidence / 100.0
    )

    scale = 1.0 / epsilon

    noise = np.random.laplace(
        loc=0.0,
        scale=scale
    )

    private_confidence = (
        confidence_value + noise
    )

    private_confidence = np.clip(
        private_confidence,
        0.0,
        1.0
    )

    return round(
        private_confidence * 100,
        2
    )


# ============================================================
# SECURE HEART DISEASE PREDICTION
# ============================================================

def predict_heart_disease(
    patient_data,
    enable_adversarial_check=True,
    enable_privacy=True,
    privacy_epsilon=5.0
):

    # ========================================================
    # STEP 1 — VALIDATE INPUT
    # ========================================================

    validate_input(
        patient_data
    )


    # ========================================================
    # STEP 2 — LOAD TRUSTED MODEL
    # ========================================================

    model = load_model()


    # ========================================================
    # STEP 3 — PREPARE INPUT
    # ========================================================

    patient_df = prepare_patient_for_model(
        patient_data
    )


    # ========================================================
    # STEP 4 — ADVERSARIAL CHECK
    # ========================================================

    if enable_adversarial_check:

        stability = (
            check_adversarial_stability(
                model,
                patient_data
            )
        )

        if not stability.get(
            "stable",
            True
        ):

            changed = stability.get(
                "changed_predictions",
                []
            )

            feature_name = "unknown"

            if changed:

                feature_name = (
                    changed[0].get(
                        "feature",
                        "unknown"
                    )
                )

            raise RuntimeError(
                "SECURITY WARNING: "
                "Prediction is unstable under "
                "small input perturbations "
                f"(feature: {feature_name}). "
                "Prediction blocked for safety."
            )


    # ========================================================
    # STEP 5 — FINAL MODEL PREDICTION
    # ========================================================

    # IMPORTANT:
    # Use prepared 18-feature dataframe here.
    prediction = model.predict(
        patient_df
    )[0]

    probabilities = model.predict_proba(
        patient_df
    )[0]

    raw_confidence = (
        max(probabilities) * 100
    )


    # ========================================================
    # STEP 6 — PRIVACY PROTECTION
    # ========================================================

    if enable_privacy:

        confidence = privatize_confidence(
            raw_confidence,
            epsilon=privacy_epsilon
        )

    else:

        confidence = round(
            raw_confidence,
            2
        )


    # ========================================================
    # RETURN
    # ========================================================

    return (
        int(prediction),
        confidence
    )