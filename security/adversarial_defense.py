import pandas as pd


# ============================================================
# NUMERIC FEATURES USED FOR ADVERSARIAL PERTURBATION
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
# MODEL FEATURE COLUMNS
# ============================================================

EXPECTED_FEATURES = [
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
# CONVERT PATIENT INPUT TO MODEL FEATURES
# ============================================================

def prepare_patient_for_model(patient):
    """
    Convert raw patient input into the same feature structure
    that was used while training the Random Forest model.
    """

    data = {}

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    for feature in [
        "age",
        "trestbps",
        "chol",
        "thalch",
        "oldpeak",
        "ca"
    ]:
        data[feature] = patient.get(feature, 0)


    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------

    sex = str(patient.get("sex", "")).strip().lower()

    data["sex_Male"] = (
        sex == "male"
    )


    # --------------------------------------------------------
    # Chest pain type
    # --------------------------------------------------------

    cp = str(
        patient.get("cp", "")
    ).strip().lower()

    data["cp_atypical angina"] = (
        cp == "atypical angina"
    )

    data["cp_non-anginal"] = (
        cp == "non-anginal"
    )

    data["cp_typical angina"] = (
        cp == "typical angina"
    )


    # --------------------------------------------------------
    # Fasting blood sugar
    # --------------------------------------------------------

    fbs = patient.get("fbs", False)

    if isinstance(fbs, str):
        fbs = fbs.strip().lower() in [
            "true",
            "yes",
            "1",
            "high"
        ]

    data["fbs_True"] = bool(fbs)


    # --------------------------------------------------------
    # Resting ECG
    # --------------------------------------------------------

    restecg = str(
        patient.get("restecg", "")
    ).strip().lower()

    data["restecg_normal"] = (
        restecg == "normal"
    )

    data["restecg_st-t abnormality"] = (
        restecg in [
            "st-t abnormality",
            "st-t abnormality"
        ]
    )


    # --------------------------------------------------------
    # Exercise induced angina
    # --------------------------------------------------------

    exang = patient.get("exang", False)

    if isinstance(exang, str):
        exang = exang.strip().lower() in [
            "true",
            "yes",
            "1"
        ]

    data["exang_True"] = bool(exang)


    # --------------------------------------------------------
    # Slope
    # --------------------------------------------------------

    slope = str(
        patient.get("slope", "")
    ).strip().lower()

    data["slope_flat"] = (
        slope == "flat"
    )

    data["slope_upsloping"] = (
        slope == "upsloping"
    )


    # --------------------------------------------------------
    # Thalassemia
    # --------------------------------------------------------

    thal = str(
        patient.get("thal", "")
    ).strip().lower()

    data["thal_normal"] = (
        thal == "normal"
    )

    data["thal_reversable defect"] = (
        thal in [
            "reversable defect",
            "reversible defect"
        ]
    )


    # --------------------------------------------------------
    # Create DataFrame with EXACT training columns
    # --------------------------------------------------------

    df = pd.DataFrame(
        [data]
    )

    df = df.reindex(
        columns=EXPECTED_FEATURES,
        fill_value=False
    )

    return df


# ============================================================
# TRAINING DATA STATISTICS
# ============================================================

def calculate_feature_statistics(df):

    statistics = {}

    for feature in NUMERIC_FEATURES:

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna()

        statistics[feature] = {
            "mean": float(
                values.mean()
            ),

            "std": float(
                values.std()
            ),

            "min": float(
                values.min()
            ),

            "max": float(
                values.max()
            )
        }

    return statistics


# ============================================================
# INPUT ANOMALY DETECTION
# ============================================================

def detect_input_anomaly(
    patient,
    statistics,
    z_threshold=3.0
):

    suspicious_features = []
    reasons = []

    for feature in NUMERIC_FEATURES:

        value = patient.get(feature)

        if value is None:
            continue

        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        mean = statistics[feature]["mean"]
        std = statistics[feature]["std"]

        minimum = statistics[feature]["min"]
        maximum = statistics[feature]["max"]


        # ----------------------------------------------------
        # Range validation
        # ----------------------------------------------------

        if value < minimum or value > maximum:

            suspicious_features.append(
                feature
            )

            reasons.append(
                f"{feature}: outside training-data range"
            )

            continue


        # ----------------------------------------------------
        # Statistical anomaly detection
        # ----------------------------------------------------

        if std > 0:

            z_score = abs(
                (value - mean) / std
            )

            if z_score > z_threshold:

                suspicious_features.append(
                    feature
                )

                reasons.append(
                    f"{feature}: abnormal statistical deviation"
                )


    return {
        "suspicious":
            len(suspicious_features) > 0,

        "features":
            suspicious_features,

        "reasons":
            reasons
    }


# ============================================================
# PREDICTION STABILITY CHECK
# ============================================================

def check_prediction_stability(
    model,
    patient,
    perturbation=0.02
):

    # --------------------------------------------------------
    # Prepare ORIGINAL input
    # --------------------------------------------------------

    original_df = prepare_patient_for_model(
        patient
    )


    # --------------------------------------------------------
    # Original prediction
    # --------------------------------------------------------

    original_prediction = int(
        model.predict(
            original_df
        )[0]
    )


    changed_predictions = []


    # --------------------------------------------------------
    # Test small perturbations
    # --------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        value = patient.get(
            feature
        )

        if value is None:
            continue

        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        if pd.isna(value):
            continue


        # ----------------------------------------------------
        # Calculate perturbation
        # ----------------------------------------------------

        delta = max(
            abs(value) * perturbation,
            0.1
        )


        # ----------------------------------------------------
        # Test both directions
        # ----------------------------------------------------

        for direction in [-1, 1]:

            candidate = patient.copy()

            candidate[feature] = (
                value
                + direction * delta
            )


            # ------------------------------------------------
            # IMPORTANT:
            # Convert raw patient data into encoded features
            # before sending it to the model.
            # ------------------------------------------------

            candidate_df = (
                prepare_patient_for_model(
                    candidate
                )
            )


            prediction = int(
                model.predict(
                    candidate_df
                )[0]
            )


            # ------------------------------------------------
            # Record prediction changes
            # ------------------------------------------------

            if prediction != original_prediction:

                changed_predictions.append({

                    "feature":
                        feature,

                    "original_value":
                        value,

                    "modified_value":
                        candidate[feature],

                    "original_prediction":
                        original_prediction,

                    "modified_prediction":
                        prediction
                })


    # --------------------------------------------------------
    # Return security result
    # --------------------------------------------------------

    return {

        "stable":
            len(changed_predictions) == 0,

        "original_prediction":
            original_prediction,

        "changed_predictions":
            changed_predictions
    }