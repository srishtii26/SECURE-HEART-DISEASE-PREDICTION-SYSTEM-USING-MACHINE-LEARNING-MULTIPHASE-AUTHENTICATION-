import os
import sys
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "heart_model.pkl"
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "heart.csv"
)


FEATURES = [
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
# LOAD TRUSTED MODEL
# ============================================================

def load_trusted_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    return df


# ============================================================
# STANDARD MODEL PREDICTION
# ============================================================

def get_standard_prediction(model, patient):

    patient_df = pd.DataFrame(
        [patient]
    )

    patient_df = patient_df.reindex(
        columns=FEATURES
    )

    prediction = model.predict(
        patient_df
    )[0]

    probabilities = model.predict_proba(
        patient_df
    )[0]

    confidence = float(
        np.max(probabilities)
    )

    return int(prediction), confidence


# ============================================================
# PRIVACY-PRESERVING CONFIDENCE
# ============================================================

def privatize_confidence(
    confidence,
    epsilon=1.0,
    seed=None
):
    """
    Apply Laplace noise to the confidence value.

    NOTE:
    This is an output-privacy demonstration.
    It does NOT make the RandomForest training process
    formally differentially private.
    """

    if epsilon <= 0:
        raise ValueError(
            "Epsilon must be greater than zero."
        )

    rng = np.random.default_rng(seed)

    sensitivity = 1.0

    scale = sensitivity / epsilon

    noise = rng.laplace(
        loc=0.0,
        scale=scale
    )

    private_confidence = (
        confidence + noise
    )

    # Keep confidence inside valid range
    private_confidence = np.clip(
        private_confidence,
        0.0,
        1.0
    )

    return float(
        private_confidence
    )


# ============================================================
# CONFIDENCE ROUNDING
# ============================================================

def round_confidence(
    confidence,
    precision=1
):

    percentage = confidence * 100

    rounded = round(
        percentage,
        precision
    )

    return rounded


# ============================================================
# PRIVACY-PRESERVING PREDICTION
# ============================================================

def privacy_preserving_prediction(
    model,
    patient,
    epsilon=1.0,
    seed=None
):

    prediction, confidence = (
        get_standard_prediction(
            model,
            patient
        )
    )

    private_confidence = (
        privatize_confidence(
            confidence,
            epsilon=epsilon,
            seed=seed
        )
    )

    private_confidence = (
        round_confidence(
            private_confidence
        )
    )

    return (
        prediction,
        private_confidence
    )


# ============================================================
# DEMONSTRATION
# ============================================================

def main():

    print("=" * 60)
    print("PRIVACY-PRESERVING PREDICTION DEMONSTRATION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading trusted model...")

    model = load_trusted_model()

    print(
        "Trusted model loaded successfully."
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\nLoading dataset...")

    df = load_dataset()

    print(
        f"Dataset loaded: {len(df)} records"
    )

    # --------------------------------------------------------
    # SELECT A VALID PATIENT
    # --------------------------------------------------------

    print("\nSelecting test patient...")

    available = df.dropna(
        subset=FEATURES
    )

    if len(available) == 0:

        raise RuntimeError(
            "No complete patient record available."
        )

    patient = (
        available.iloc[0][FEATURES]
        .to_dict()
    )

    print(
        "Patient selected successfully."
    )

    # --------------------------------------------------------
    # STANDARD PREDICTION
    # --------------------------------------------------------

    print("\n==============================================")
    print("STANDARD MODEL OUTPUT")
    print("==============================================")

    prediction, confidence = (
        get_standard_prediction(
            model,
            patient
        )
    )

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Original confidence: "
        f"{confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # TEST DIFFERENT EPSILON VALUES
    # --------------------------------------------------------

    print("\n==============================================")
    print("PRIVACY PARAMETER TEST")
    print("==============================================")

    print(
        "Lower epsilon = stronger privacy"
    )

    print(
        "Higher epsilon = less privacy noise"
    )

    epsilon_values = [
        0.5,
        1.0,
        2.0,
        5.0
    ]

    for epsilon in epsilon_values:

        private_prediction, private_confidence = (
            privacy_preserving_prediction(
                model,
                patient,
                epsilon=epsilon,
                seed=42
            )
        )

        print(
            f"\nEpsilon = {epsilon}"
        )

        print(
            f"Prediction: "
            f"{private_prediction}"
        )

        print(
            f"Private confidence: "
            f"{private_confidence:.1f}%"
        )

    # --------------------------------------------------------
    # MULTIPLE QUERIES
    # --------------------------------------------------------

    print("\n==============================================")
    print("CONFIDENCE LEAKAGE COMPARISON")
    print("==============================================")

    standard_values = []

    private_values = []

    test_patients = (
        available.iloc[:20]
    )

    for index, row in test_patients.iterrows():

        patient_data = (
            row[FEATURES]
            .to_dict()
        )

        _, standard_confidence = (
            get_standard_prediction(
                model,
                patient_data
            )
        )

        _, private_confidence = (
            privacy_preserving_prediction(
                model,
                patient_data,
                epsilon=1.0,
                seed=index
            )
        )

        standard_values.append(
            standard_confidence * 100
        )

        private_values.append(
            private_confidence
        )

    standard_mean = np.mean(
        standard_values
    )

    private_mean = np.mean(
        private_values
    )

    print(
        f"Average standard confidence: "
        f"{standard_mean:.2f}%"
    )

    print(
        f"Average private confidence: "
        f"{private_mean:.2f}%"
    )

    # --------------------------------------------------------
    # SECURITY ANALYSIS
    # --------------------------------------------------------

    print("\n==============================================")
    print("SECURITY ANALYSIS")
    print("==============================================")

    print(
        "✅ Privacy-preserving output layer enabled."
    )

    print(
        "Confidence information is perturbed "
        "before being exposed to the user."
    )

    print(
        "This can reduce information leakage "
        "through prediction confidence."
    )

    # --------------------------------------------------------
    # IMPORTANT LIMITATION
    # --------------------------------------------------------

    print("\n==============================================")
    print("IMPORTANT PRIVACY LIMITATION")
    print("==============================================")

    print(
        "This demonstration applies noise to "
        "model output."
    )

    print(
        "It does NOT provide formal differential "
        "privacy guarantees for RandomForest training."
    )

    print(
        "Formal DP training would require a "
        "differentially-private training algorithm."
    )

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    print("\n==============================================")
    print("SAFETY CHECK")
    print("==============================================")

    print(
        "Trusted model was NOT modified."
    )

    print(
        "Original dataset was NOT modified."
    )

    print(
        "Privacy testing was performed using "
        "temporary in-memory data."
    )

    print(
        "\nPrivacy defense demonstration completed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    main()