import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("dataset/heart.csv")

print("Original dataset shape:", data.shape)


# ==========================================
# 2. REMOVE UNNECESSARY COLUMNS
# ==========================================

data = data.drop(columns=["id", "dataset"])


# ==========================================
# 3. CONVERT TARGET TO BINARY
# ==========================================

# 0 = No Heart Disease
# 1,2,3,4 = Heart Disease

data["num"] = data["num"].apply(
    lambda x: 0 if x == 0 else 1
)


# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================

numerical_columns = [
    "age",
    "trestbps",
    "chol",
    "thalch",
    "oldpeak",
    "ca"
]

categorical_columns = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "thal"
]

# Numerical → median
for column in numerical_columns:
    data[column] = data[column].fillna(
        data[column].median()
    )

# Categorical → most frequent value
for column in categorical_columns:
    data[column] = data[column].fillna(
        data[column].mode()[0]
    )


# ==========================================
# 5. ENCODE CATEGORICAL FEATURES
# ==========================================

data = pd.get_dummies(
    data,
    columns=categorical_columns,
    drop_first=True
)


# ==========================================
# 6. DISPLAY RESULT
# ==========================================

print("\nDataset after encoding:")

print(data.head())

print("\nNew dataset shape:")
print(data.shape)

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())
# ==========================================
# 7. SEPARATE FEATURES AND TARGET
# ==========================================

# X = input features
# y = target

X = data.drop(columns=["num"])
y = data["num"]

print("\nFeatures (X) shape:")
print(X.shape)

print("\nTarget (y) shape:")
print(y.shape)

print("\nFeature columns:")
print(X.columns.tolist())
# ==========================================
# 8. TRAIN-TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())
# ==========================================
# 9. TRAIN RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")
# ==========================================
# 10. MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# ==========================================
# 11. SAVE TRAINED MODEL
# ==========================================

import joblib

# Save the trained Random Forest model
joblib.dump(model, "models/heart_model.pkl")

print("\nModel saved successfully!")
print("Location: models/heart_model.pkl")
# ==========================================
# TEST MODEL WITH ONE ACTUAL PATIENT
# ==========================================

import joblib

# Load the saved model
saved_model = joblib.load("models/heart_model.pkl")

# Take one actual patient from the dataset
test_patient = X_test.iloc[0:1]

actual_result = y_test.iloc[0]

predicted_result = saved_model.predict(test_patient)[0]

probabilities = saved_model.predict_proba(test_patient)[0]

print("\n========== SINGLE PATIENT TEST ==========")

print("Actual result:", actual_result)
print("Predicted result:", predicted_result)

print(
    "Probability of No Disease:",
    round(probabilities[0] * 100, 2),
    "%"
)

print(
    "Probability of Disease:",
    round(probabilities[1] * 100, 2),
    "%"
)
# ==========================================
# TEST SAVED MODEL
# ==========================================

import joblib

saved_model = joblib.load("models/heart_model.pkl")

# Test several actual patients from the test dataset
sample_patients = X_test.iloc[:10]

predictions = saved_model.predict(sample_patients)

print("\n========== MODEL PREDICTION TEST ==========")
print("Predictions for first 10 test patients:")
print(predictions)

print("\nActual values:")
print(y_test.iloc[:10].values)

print("\nNumber of positive predictions:")
print(sum(predictions == 1))

print("Number of negative predictions:")
print(sum(predictions == 0))