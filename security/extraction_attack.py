import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/heart_model.pkl"
DATASET_PATH = "dataset/heart.csv"

QUERY_LIMIT = 100


# ============================================================
# FEATURES USED BY THE MODEL
# ============================================================

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


NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca"
]


CATEGORICAL_FEATURES = [
    "sex",
    "dataset",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 55)
print("MODEL EXTRACTION ATTACK DEMONSTRATION")
print("=" * 55)


# ============================================================
# LOAD TRUSTED MODEL
# ============================================================

print("\nLoading trusted model...")

trusted_model = joblib.load(
    MODEL_PATH
)

print(
    "Trusted model loaded successfully."
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Dataset loaded: {len(df)} records"
)


# ============================================================
# SELECT MODEL FEATURES
# ============================================================

X = df[FEATURES].copy()


# ============================================================
# FIX NUMERIC COLUMNS
# ============================================================

print(
    "\nPreparing query data..."
)

for column in NUMERIC_FEATURES:

    # Convert strings to numbers
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

    # Fill missing numeric values
    X[column] = X[column].fillna(
        X[column].median()
    )


# ============================================================
# FIX CATEGORICAL COLUMNS
# ============================================================

for column in CATEGORICAL_FEATURES:

    X[column] = X[column].astype("object")

    X[column] = X[column].fillna(
        X[column].mode()[0]
    )


print(
    "Query data prepared successfully."
)


# ============================================================
# LIMIT ATTACKER QUERIES
# ============================================================

X_query = X.iloc[
    :QUERY_LIMIT
].copy()


print(
    f"\nAttacker query budget: "
    f"{len(X_query)}"
)


# ============================================================
# QUERY TRUSTED MODEL
# ============================================================

print(
    "\nQuerying trusted model..."
)

black_box_predictions = (
    trusted_model.predict(
        X_query
    )
)


print(
    "Model queries completed."
)


# ============================================================
# DISPLAY SAMPLE RESPONSES
# ============================================================

print(
    "\nSample model responses:"
)

for i in range(
    min(10, len(black_box_predictions))
):

    print(
        f"Query {i + 1}: "
        f"Prediction = "
        f"{black_box_predictions[i]}"
    )


# ============================================================
# CREATE ATTACKER DATASET
# ============================================================

print(
    "\nCreating attacker knowledge dataset..."
)

attacker_X = X_query.copy()

attacker_y = black_box_predictions


print(
    f"Attacker collected "
    f"{len(attacker_X)} labeled samples."
)


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

attacker_X_encoded = pd.get_dummies(
    attacker_X
)


print(
    f"Encoded attacker features: "
    f"{attacker_X_encoded.shape[1]}"
)


# ============================================================
# TRAIN SUBSTITUTE MODEL
# ============================================================

print(
    "\nTraining substitute model..."
)

substitute_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


substitute_model.fit(
    attacker_X_encoded,
    attacker_y
)


print(
    "Substitute model trained successfully."
)


# ============================================================
# TEST SUBSTITUTE MODEL
# ============================================================

print(
    "\nEvaluating model extraction..."
)

substitute_predictions = (
    substitute_model.predict(
        attacker_X_encoded
    )
)


agreement = accuracy_score(
    attacker_y,
    substitute_predictions
)


# ============================================================
# RESULTS
# ============================================================

print(
    "\n" + "=" * 55
)

print(
    "MODEL EXTRACTION RESULTS"
)

print(
    "=" * 55
)

print(
    f"Queries made: "
    f"{len(X_query)}"
)

print(
    f"Substitute-model agreement: "
    f"{agreement * 100:.2f}%"
)


# ============================================================
# SECURITY ANALYSIS
# ============================================================

print(
    "\n" + "=" * 55
)

print(
    "SECURITY ANALYSIS"
)

print(
    "=" * 55
)


if agreement >= 0.90:

    print(
        "⚠️ HIGH EXTRACTION RISK"
    )

    print(
        "The substitute model closely "
        "matches the trusted model."
    )

elif agreement >= 0.75:

    print(
        "⚠️ MODERATE EXTRACTION RISK"
    )

    print(
        "The substitute model shows "
        "substantial agreement."
    )

else:

    print(
        "✅ LOW EXTRACTION AGREEMENT"
    )

    print(
        "The substitute model does not "
        "closely reproduce the trusted model."
    )


# ============================================================
# SAFETY CHECK
# ============================================================

print(
    "\n" + "=" * 55
)

print(
    "SAFETY CHECK"
)

print(
    "=" * 55
)

print(
    "Trusted model was NOT modified."
)

print(
    "Original dataset was NOT modified."
)

print(
    "Substitute model exists only "
    "for this demonstration."
)

print(
    "\nModel extraction demonstration completed."
)