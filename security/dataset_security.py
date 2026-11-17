import pandas as pd
import numpy as np


# ============================================================
# DATASET SECURITY SCANNER
# ============================================================

def scan_dataset(df, target_column="num"):

    report = {}

    # --------------------------------------------------------
    # 1. Missing values
    # --------------------------------------------------------

    missing_values = df.isnull().sum().sum()

    report["missing_values"] = int(
        missing_values
    )

    # Missing values are reported but NOT automatically
    # classified as poisoning because the ML pipeline
    # already handles missing values using imputation.


    # --------------------------------------------------------
    # 2. Duplicate records
    # --------------------------------------------------------

    duplicates = df.duplicated().sum()

    report["duplicate_records"] = int(
        duplicates
    )


    # --------------------------------------------------------
    # 3. Convert UCI target to binary
    # --------------------------------------------------------

    if target_column in df.columns:

        raw_target = df[target_column]

        binary_target = (
            raw_target > 0
        ).astype(int)

        target_distribution = (
            binary_target
            .value_counts()
            .sort_index()
            .to_dict()
        )

    else:

        target_distribution = {}


    report["binary_target_distribution"] = (
        target_distribution
    )


    # --------------------------------------------------------
    # 4. Binary class balance
    # --------------------------------------------------------

    if len(target_distribution) >= 2:

        largest_class = max(
            target_distribution.values()
        )

        smallest_class = min(
            target_distribution.values()
        )

        imbalance_ratio = (
            largest_class /
            smallest_class
        )

    else:

        imbalance_ratio = 1.0


    report["class_imbalance_ratio"] = round(
        float(imbalance_ratio),
        2
    )


    # --------------------------------------------------------
    # 5. Numeric outlier analysis
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    outlier_counts = {}


    for column in numeric_columns:

        # Target column is not an input feature
        if column == target_column:
            continue

        # ID is an identifier, not a medical measurement
        if column == "id":
            continue

        series = df[column].dropna()

        if len(series) == 0:
            continue

        Q1 = series.quantile(0.25)

        Q3 = series.quantile(0.75)

        IQR = Q3 - Q1

        if IQR == 0:

            outlier_counts[column] = 0

            continue

        lower_bound = (
            Q1 - 1.5 * IQR
        )

        upper_bound = (
            Q3 + 1.5 * IQR
        )

        outliers = (
            (df[column] < lower_bound)
            |
            (df[column] > upper_bound)
        ).sum()

        outlier_counts[column] = int(
            outliers
        )


    report["outliers"] = (
        outlier_counts
    )


    # --------------------------------------------------------
    # 6. Infinite values
    # --------------------------------------------------------

    numeric_data = df.select_dtypes(
        include=np.number
    )

    infinite_values = np.isinf(
        numeric_data
    ).sum().sum()

    report["infinite_values"] = int(
        infinite_values
    )


    # --------------------------------------------------------
    # 7. Security decision
    # --------------------------------------------------------

    suspicious = False

    reasons = []


    # Infinite values are a strong data-integrity
    # warning.

    if infinite_values > 0:

        suspicious = True

        reasons.append(
            "Infinite numeric values detected"
        )


    # Duplicate records are reported for review,
    # but are not automatically considered poisoning.

    if duplicates > 0:

        reasons.append(
            "Duplicate records detected for review"
        )


    # Outliers are reported as anomalies,
    # not automatically classified as attacks.

    total_outliers = sum(
        outlier_counts.values()
    )


    if total_outliers > 0:

        reasons.append(
            "Potential numeric anomalies detected for review"
        )


    # Only flag severe class imbalance if it exists
    # AFTER binary conversion.

    if imbalance_ratio > 10:

        suspicious = True

        reasons.append(
            "Severe binary class imbalance detected"
        )


    report["suspicious"] = (
        suspicious
    )

    report["reasons"] = reasons


    return report