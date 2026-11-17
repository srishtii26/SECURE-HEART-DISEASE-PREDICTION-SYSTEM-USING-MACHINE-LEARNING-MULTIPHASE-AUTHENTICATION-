import os
import sys
import joblib
import pandas as pd
import numpy as np

# Allow execution from the project root or security folder
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


def load_trusted_model():
    """Load the trusted production model."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trusted model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("Trusted model loaded successfully.")

    return model


def prepare_dataset():
    """Load and prepare the heart disease dataset."""

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(
        f"Dataset loaded: {len(df)} records"
    )

    # Keep only features used by the model
    X = df[FEATURES].copy()

    # Convert target to binary
    y = (df["num"] > 0).astype(int)

    return X, y


def calculate_prediction_confidence(
    model,
    X
):
    """Get model predictions and confidence."""

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    confidence = np.max(
        probabilities,
        axis=1
    )

    return predictions, confidence


def main():

    print("=" * 55)
    print("MEMBERSHIP INFERENCE ATTACK DEMONSTRATION")
    print("=" * 55)

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    print("\nLoading trusted model...")

    model = load_trusted_model()

    # --------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------

    print("\nLoading dataset...")

    X, y = prepare_dataset()

    # --------------------------------------------------
    # REMOVE COMPLETELY MISSING TARGET ROWS
    # --------------------------------------------------

    valid_rows = y.notna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    # --------------------------------------------------
    # SPLIT DATA INTO MEMBERS AND NON-MEMBERS
    # --------------------------------------------------

    # For demonstration:
    # first 100 records = members
    # next 100 records = non-members

    member_count = min(
        100,
        len(X) // 2
    )

    non_member_start = member_count
    non_member_end = min(
        non_member_start + member_count,
        len(X)
    )

    X_members = X.iloc[
        :member_count
    ].copy()

    X_non_members = X.iloc[
        non_member_start:non_member_end
    ].copy()

    print("\n==============================================")
    print("MEMBERSHIP DATA PREPARATION")
    print("==============================================")

    print(
        f"Member samples: {len(X_members)}"
    )

    print(
        f"Non-member samples: {len(X_non_members)}"
    )

    # --------------------------------------------------
    # QUERY TRUSTED MODEL
    # --------------------------------------------------

    print("\n==============================================")
    print("QUERYING TRUSTED MODEL")
    print("==============================================")

    member_predictions, member_confidence = (
        calculate_prediction_confidence(
            model,
            X_members
        )
    )

    non_member_predictions, non_member_confidence = (
        calculate_prediction_confidence(
            model,
            X_non_members
        )
    )

    print("Model queries completed.")

    # --------------------------------------------------
    # ANALYZE CONFIDENCE
    # --------------------------------------------------

    member_mean = np.mean(
        member_confidence
    )

    non_member_mean = np.mean(
        non_member_confidence
    )

    confidence_gap = (
        member_mean -
        non_member_mean
    )

    print("\n==============================================")
    print("MEMBERSHIP ANALYSIS")
    print("==============================================")

    print(
        f"Average member confidence: "
        f"{member_mean * 100:.2f}%"
    )

    print(
        f"Average non-member confidence: "
        f"{non_member_mean * 100:.2f}%"
    )

    print(
        f"Confidence gap: "
        f"{confidence_gap * 100:.2f}%"
    )

    # --------------------------------------------------
    # MEMBERSHIP INFERENCE RULE
    # --------------------------------------------------

    # Simple threshold-based inference.
    # The attacker assumes a record is a member
    # when model confidence is higher than the
    # combined average confidence.

    threshold = (
        member_mean +
        non_member_mean
    ) / 2

    member_guesses = (
        member_confidence >= threshold
    )

    non_member_guesses = (
        non_member_confidence >= threshold
    )

    correct_members = np.sum(
        member_guesses
    )

    correct_non_members = np.sum(
        ~non_member_guesses
    )

    total_correct = (
        correct_members +
        correct_non_members
    )

    total_samples = (
        len(member_confidence) +
        len(non_member_confidence)
    )

    attack_accuracy = (
        total_correct /
        total_samples
    ) * 100

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n==============================================")
    print("MEMBERSHIP INFERENCE RESULTS")
    print("==============================================")

    print(
        f"Attack samples: {total_samples}"
    )

    print(
        f"Inference threshold: "
        f"{threshold * 100:.2f}%"
    )

    print(
        f"Membership inference accuracy: "
        f"{attack_accuracy:.2f}%"
    )

    # --------------------------------------------------
    # SECURITY ANALYSIS
    # --------------------------------------------------

    print("\n==============================================")
    print("SECURITY ANALYSIS")
    print("==============================================")

    if attack_accuracy >= 70:

        print("⚠️ HIGH MEMBERSHIP INFERENCE RISK")

        print(
            "The attacker can distinguish member "
            "and non-member records with relatively "
            "high accuracy."
        )

    elif attack_accuracy >= 60:

        print("⚠️ MODERATE MEMBERSHIP INFERENCE RISK")

        print(
            "The model shows some distinguishable "
            "behavior between member and non-member "
            "records."
        )

    else:

        print("✅ LOW MEMBERSHIP INFERENCE RISK")

        print(
            "The tested confidence-based attack "
            "performed close to random guessing."
        )

    # --------------------------------------------------
    # PRIVACY INTERPRETATION
    # --------------------------------------------------

    print("\n==============================================")
    print("PRIVACY INTERPRETATION")
    print("==============================================")

    if abs(confidence_gap) >= 0.10:

        print(
            "⚠️ Noticeable confidence difference detected."
        )

        print(
            "Model confidence may reveal information "
            "about whether a record was part of the "
            "training data."
        )

    else:

        print(
            "✅ No large confidence gap detected "
            "under this test."
        )

    # --------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------

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
        "Membership inference was performed "
        "using read-only model queries."
    )

    print(
        "No patient medical information was "
        "stored by this demonstration."
    )

    print(
        "\nMembership inference demonstration completed."
    )


if __name__ == "__main__":
    main()