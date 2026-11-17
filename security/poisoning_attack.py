import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataset_security import scan_dataset
# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/heart.csv"

RANDOM_STATE = 42

POISONING_RATE = 0.05


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==============================================")
print("MODEL POISONING ATTACK DEMONSTRATION")
print("==============================================\n")

print("Loading original dataset...")

if not os.path.exists(DATASET_PATH):

    print("ERROR: Dataset not found.")
    print(DATASET_PATH)
    exit()

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

print("Number of records:", len(df))
print("Number of attributes:", len(df.columns))


# ============================================================
# DISPLAY DATASET INFORMATION
# ============================================================

print("\nDataset columns:")

for column in df.columns:

    print(" -", column)
    # ============================================================
# DATASET SECURITY SCAN
# ============================================================

print(
    "\n=============================================="
)

print(
    "DATASET SECURITY SCAN"
)

print(
    "==============================================\n"
)

security_report = scan_dataset(
    df,
    target_column="num"
)


print(
    "Missing values:",
    security_report["missing_values"]
)

print(
    "Duplicate records:",
    security_report["duplicate_records"]
)

print(
    "Binary target distribution:",
    security_report["binary_target_distribution"]
)

print(
    "Class imbalance ratio:",
    security_report["class_imbalance_ratio"]
)

print(
    "Infinite values:",
    security_report["infinite_values"]
)

print(
    "Potential anomalies:",
    security_report["outliers"]
)


if security_report["suspicious"]:

    print(
        "\n⚠️ DATASET SECURITY WARNING"
    )

    for reason in security_report["reasons"]:

        print(
            "-",
            reason
        )

else:

    print(
        "\n✅ DATASET BASELINE CHECK PASSED"
    )

    print(
        "No critical dataset-integrity issue detected."
    )


# ============================================================
# IDENTIFY TARGET COLUMN
# ============================================================

print("\nChecking target column...")

if "num" in df.columns:

    TARGET_COLUMN = "num"

elif "target" in df.columns:

    TARGET_COLUMN = "target"

elif "condition" in df.columns:

    TARGET_COLUMN = "condition"

else:

    print("\nERROR: Could not identify the target column.")

    print(
        "\nAvailable columns:"
    )

    print(
        list(df.columns)
    )

    exit()


print(
    "\nTarget column:",
    TARGET_COLUMN
)


# ============================================================
# PREPARE TARGET
# ============================================================

X = df.drop(
    columns=[TARGET_COLUMN]
)

y = df[TARGET_COLUMN]


# ============================================================
# CONVERT TARGET TO BINARY
# ============================================================

# UCI Heart Disease dataset commonly represents
# disease severity using values 0–4.
#
# We convert:
#
# 0 → No disease
# 1–4 → Disease
#
# If the target is already binary, it is kept as-is.

if y.max() > 1:

    y = (
        y > 0
    ).astype(int)

else:

    y = y.astype(int)


print(
    "\nTarget distribution:"
)

print(
    y.value_counts()
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    "\nTraining records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# CREATE PREPROCESSING PIPELINE
# ============================================================

numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


categorical_features = X_train.select_dtypes(
    exclude=["int64", "float64"]
).columns.tolist()


numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            numeric_pipeline,
            numeric_features
        ),

        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# FUNCTION TO CREATE MODEL
# ============================================================

def create_model():

    return Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                RandomForestClassifier(

                    n_estimators=300,

                    random_state=RANDOM_STATE,

                    n_jobs=-1
                )
            )
        ]
    )


# ============================================================
# TRAIN CLEAN MODEL
# ============================================================

print(
    "\n=============================================="
)

print(
    "TRAINING CLEAN MODEL"
)

print(
    "==============================================\n"
)


clean_model = create_model()

clean_model.fit(
    X_train,
    y_train
)


clean_predictions = clean_model.predict(
    X_test
)


clean_accuracy = accuracy_score(
    y_test,
    clean_predictions
)


print(
    "Clean model accuracy:",
    f"{clean_accuracy * 100:.2f}%"
)


# ============================================================
# CREATE POISONED DATASET
# ============================================================

print(
    "\n=============================================="
)

print(
    "CREATING POISONED TRAINING DATA"
)

print(
    "==============================================\n"
)


poisoned_X_train = X_train.copy()

poisoned_y_train = y_train.copy()


number_to_poison = int(
    len(poisoned_X_train)
    * POISONING_RATE
)


print(
    "Poisoning rate:",
    f"{POISONING_RATE * 100:.1f}%"
)


print(
    "Records to poison:",
    number_to_poison
)


# ============================================================
# SELECT RANDOM TRAINING RECORDS
# ============================================================

rng = np.random.RandomState(
    RANDOM_STATE
)


poison_indices = rng.choice(

    poisoned_X_train.index,

    size=number_to_poison,

    replace=False
)


# ============================================================
# FLIP LABELS
# ============================================================

print(
    "\nInjecting incorrect labels..."
)


for index in poison_indices:

    poisoned_y_train.loc[index] = (
        1 - poisoned_y_train.loc[index]
    )


print(
    "Labels modified:",
    len(poison_indices)
)


# ============================================================
# TRAIN POISONED MODEL
# ============================================================

print(
    "\n=============================================="
)

print(
    "TRAINING POISONED MODEL"
)

print(
    "==============================================\n"
)


poisoned_model = create_model()


poisoned_model.fit(

    poisoned_X_train,

    poisoned_y_train
)


poisoned_predictions = poisoned_model.predict(
    X_test
)


poisoned_accuracy = accuracy_score(

    y_test,

    poisoned_predictions
)


print(
    "Poisoned model accuracy:",
    f"{poisoned_accuracy * 100:.2f}%"
)


# ============================================================
# COMPARE MODELS
# ============================================================

accuracy_change = (
    clean_accuracy
    -
    poisoned_accuracy
)


print(
    "\n=============================================="
)

print(
    "ATTACK RESULTS"
)

print(
    "==============================================\n"
)


print(
    "Clean model accuracy:",
    f"{clean_accuracy * 100:.2f}%"
)


print(
    "Poisoned model accuracy:",
    f"{poisoned_accuracy * 100:.2f}%"
)


print(
    "Accuracy change:",
    f"{accuracy_change * 100:.2f}%"
)


# ============================================================
# BASIC POISONING DETECTION
# ============================================================

print(
    "\n=============================================="
)

print(
    "POISONING DETECTION"
)

print(
    "==============================================\n"
)


# Detect suspicious label changes by checking
# whether the number of labels changed is
# above the permitted threshold.

detected_poisoning = (
    len(poison_indices)
    >
    len(poisoned_y_train) * 0.01
)


if detected_poisoning:

    print(
        "⚠️ POISONING DETECTED"
    )

    print(
        "Suspicious label modifications "
        "exceeded the permitted threshold."
    )

else:

    print(
        "No significant poisoning detected."
    )


# ============================================================
# SECURITY DECISION
# ============================================================

print(
    "\n=============================================="
)

print(
    "SECURITY DECISION"
)

print(
    "==============================================\n"
)


if detected_poisoning:

    print(
        "❌ POISONED TRAINING DATA REJECTED"
    )

    print(
        "The trusted production model remains unchanged."
    )

else:

    print(
        "✅ TRAINING DATA ACCEPTED"
    )


# ============================================================
# IMPORTANT SAFETY MESSAGE
# ============================================================

print(
    "\n=============================================="
)

print(
    "SAFETY CHECK"
)

print(
    "==============================================\n"
)


print(
    "The original dataset was NOT modified."
)

print(
    "The trusted heart_model.pkl was NOT modified."
)

print(
    "The poisoned model exists only in memory."
)

print(
    "\nModel poisoning demonstration completed."
)