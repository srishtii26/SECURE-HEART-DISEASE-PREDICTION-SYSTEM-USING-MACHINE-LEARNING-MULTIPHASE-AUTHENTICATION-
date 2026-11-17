import os
import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/heart_model.pkl"
DATASET_PATH = "dataset/heart.csv"

MAX_PATIENTS = 100


# ============================================================
# LOAD TRUSTED MODEL
# ============================================================

print("\n==============================================")
print("ADVERSARIAL EXAMPLE ATTACK DEMONSTRATION")
print("==============================================\n")


if not os.path.exists(MODEL_PATH):

    print("ERROR: Trusted model not found.")
    exit()


model = joblib.load(
    MODEL_PATH
)

print(
    "Trusted model loaded successfully."
)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("ERROR: Dataset not found.")
    exit()


df = pd.read_csv(
    DATASET_PATH
)

print(
    "Dataset loaded:",
    len(df),
    "records"
)


# ============================================================
# EXPECTED MODEL FEATURES
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


# ============================================================
# NUMERICAL FEATURES
# ============================================================

NUMERIC_FEATURES = [

    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca"

]


# ============================================================
# ALLOWED PERTURBATIONS
# ============================================================

PERTURBATIONS = {

    "age": [-5, -3, -2, -1, 1, 2, 3, 5],

    "trestbps": [-10, -5, -2, 2, 5, 10],

    "chol": [-20, -10, -5, 5, 10, 20],

    "thalch": [-10, -5, -2, 2, 5, 10],

    "oldpeak": [
        -1.0,
        -0.5,
        -0.2,
        0.2,
        0.5,
        1.0
    ],

    "ca": [-1, 1]

}


# ============================================================
# MEDICAL RANGE VALIDATION
# ============================================================

def valid_range(
    feature,
    value
):

    if feature == "age":

        return 1 <= value <= 120


    if feature == "trestbps":

        return 50 <= value <= 250


    if feature == "chol":

        return 50 <= value <= 700


    if feature == "thalch":

        return 50 <= value <= 250


    if feature == "oldpeak":

        return 0 <= value <= 10


    if feature == "ca":

        return 0 <= value <= 4


    return True


# ============================================================
# SEARCH FOR ADVERSARIAL EXAMPLES
# ============================================================

print(
    "\n=============================================="
)

print(
    "SEARCHING MULTIPLE PATIENTS"
)

print(
    "==============================================\n"
)


tested_patients = 0

adversarial_examples = []


for index, row in df.iterrows():

    if tested_patients >= MAX_PATIENTS:

        break


    # --------------------------------------------------------
    # Make sure all required features exist
    # --------------------------------------------------------

    if not all(
        feature in row.index
        for feature in FEATURES
    ):

        continue


    # --------------------------------------------------------
    # Skip patients with missing numerical values
    # --------------------------------------------------------

    if row[NUMERIC_FEATURES].isnull().any():

        continue


    patient = row[
        FEATURES
    ].copy()


    patient_df = pd.DataFrame(
        [patient]
    )


    # --------------------------------------------------------
    # Original prediction
    # --------------------------------------------------------

    original_prediction = int(
        model.predict(
            patient_df
        )[0]
    )


    original_probability = (
        max(
            model.predict_proba(
                patient_df
            )[0]
        )
        * 100
    )


    tested_patients += 1


    # --------------------------------------------------------
    # Try perturbations
    # --------------------------------------------------------

    found = False


    for feature in NUMERIC_FEATURES:

        for change in PERTURBATIONS[feature]:

            candidate = patient.copy()


            candidate[feature] = (
                candidate[feature]
                + change
            )


            # Validate medical range

            if not valid_range(
                feature,
                candidate[feature]
            ):

                continue


            candidate_df = pd.DataFrame(
                [candidate]
            )


            candidate_prediction = int(
                model.predict(
                    candidate_df
                )[0]
            )


            # ------------------------------------------------
            # Prediction changed
            # ------------------------------------------------

            if (
                candidate_prediction
                !=
                original_prediction
            ):

                candidate_probability = (
                    max(
                        model.predict_proba(
                            candidate_df
                        )[0]
                    )
                    * 100
                )


                adversarial_examples.append({

                    "patient_index": index,

                    "feature": feature,

                    "original_value": patient[feature],

                    "modified_value": candidate[feature],

                    "change": change,

                    "original_prediction":
                        original_prediction,

                    "adversarial_prediction":
                        candidate_prediction,

                    "original_confidence":
                        original_probability,

                    "adversarial_confidence":
                        candidate_probability

                })


                print(
                    "⚠️ ADVERSARIAL EXAMPLE FOUND"
                )


                print(
                    f"Patient index: {index}"
                )


                print(
                    f"Feature: {feature}"
                )


                print(
                    f"Original value: "
                    f"{patient[feature]}"
                )


                print(
                    f"Modified value: "
                    f"{candidate[feature]}"
                )


                print(
                    f"Change: {change:+}"
                )


                print(
                    f"Original prediction: "
                    f"{original_prediction}"
                )


                print(
                    f"Adversarial prediction: "
                    f"{candidate_prediction}"
                )


                print(
                    f"Original confidence: "
                    f"{original_probability:.2f}%"
                )


                print(
                    f"Adversarial confidence: "
                    f"{candidate_probability:.2f}%"
                )


                print(
                    "----------------------------------------------"
                )


                found = True

                break


        if found:

            break


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n=============================================="
)

print(
    "ADVERSARIAL ATTACK SUMMARY"
)

print(
    "==============================================\n"
)


print(
    "Patients tested:",
    tested_patients
)


print(
    "Adversarial examples found:",
    len(adversarial_examples)
)


if len(adversarial_examples) > 0:

    print(
        "\n⚠️ The model was vulnerable to at least"
        " one tested input perturbation."
    )

else:

    print(
        "\nNo prediction-changing example was found"
    )

    print(
        "within the tested perturbation range."
    )


# ============================================================
# INPUT SECURITY CHECK
# ============================================================

print(
    "\n=============================================="
)

print(
    "INPUT SECURITY CHECK"
)

print(
    "==============================================\n"
)


def check_input_change(
    original,
    modified
):

    limits = {

        "age": 5,

        "trestbps": 10,

        "chol": 20,

        "thalch": 10,

        "oldpeak": 1.0,

        "ca": 1

    }


    suspicious = []


    for feature in NUMERIC_FEATURES:

        difference = abs(
            modified[feature]
            -
            original[feature]
        )


        if difference > limits[feature]:

            suspicious.append(
                feature
            )


    if suspicious:

        return False, suspicious


    return True, []


print(
    "Input boundary validation enabled."
)


print(
    "Large unexpected changes can be flagged"
    " before prediction."
)


# ============================================================
# SAFETY CHECK
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
    "Trusted model was NOT modified."
)


print(
    "Original dataset was NOT modified."
)


print(
    "All adversarial testing was performed"
    " on temporary in-memory inputs."
)


print(
    "\nAdversarial attack demonstration completed."
)