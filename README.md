# SECURE-HEART-DISEASE-PREDICTION-SYSTEM-USING-MACHINE-LEARNING-MULTIPHASE-AUTHENTICATION FOR ENHANCED CYBERSECURITY
## 1. ABSTRACT

The Secure Heart Disease Prediction System is a machine-learning-based healthcare application designed to predict the likelihood of heart disease while incorporating multiple security mechanisms to protect the application, its users, and the underlying machine-learning model.

Traditional machine-learning prediction systems primarily concentrate on prediction accuracy. However, healthcare applications handle sensitive patient information and are exposed to several security threats, including unauthorized access, model tampering, data manipulation, adversarial attacks, model extraction, and privacy violations.

This project addresses these concerns by combining a machine-learning prediction pipeline with a multi-layer security architecture.

The system uses a Random Forest classification algorithm trained on the UCI Heart Disease dataset. It provides a Streamlit-based interface through which authenticated users can enter patient information and obtain a heart disease prediction.

The security architecture incorporates password hashing using bcrypt, role-based access control, WhatsApp OTP verification, a permanent security matrix, administrative face verification, SHA-256 model integrity verification, input validation, activity logging, and mechanisms intended to examine and mitigate machine-learning-specific attacks.

The project therefore focuses on two major objectives:

1. Developing a functional machine-learning system for heart disease prediction.
2. Protecting the prediction system against unauthorized access, manipulation, and misuse through layered security mechanisms.

The main objective is to demonstrate that machine-learning applications in healthcare should consider both predictive performance and information security.

---

# 2. INTRODUCTION

## 2.1 Background

Heart disease is one of the major health conditions requiring early identification and medical attention.

Machine learning can assist in identifying patterns in patient information and estimating the likelihood of heart disease.

A machine-learning model can analyse patient attributes such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting electrocardiographic results
- Maximum heart rate
- Exercise-induced angina
- ST depression
- ST slope
- Number of major vessels
- Thalassemia-related information

However, healthcare prediction applications introduce additional security requirements.

For example:

- An unauthorized person could access patient prediction records.
- A malicious user could modify the trained model.
- Incorrectly manipulated input data could affect predictions.
- An attacker could attempt to extract information about the trained model.
- Sensitive user information could be exposed through insecure storage.
- An attacker could attempt to impersonate an administrator.

Therefore, a secure heart disease prediction system must protect the entire application rather than only its login page.

## 2.2 Problem Statement

Many basic machine-learning healthcare applications provide predictions without implementing sufficient security mechanisms.

Such systems may be vulnerable to:

1. Unauthorized user access.
2. Password theft and credential compromise.
3. Model file modification.
4. Malicious input manipulation.
5. Model poisoning.
6. Adversarial examples.
7. Model extraction attempts.
8. Exposure of sensitive prediction information.
9. Unauthorized administrative access.

The problem addressed by this project is the development of a heart disease prediction application that combines machine-learning functionality with multiple security controls.

## 2.3 Objectives

The major objectives are:

1. Develop a machine-learning model capable of predicting heart disease.
2. Use appropriate preprocessing techniques to prepare the dataset.
3. Implement a user-friendly Streamlit application.
4. Implement secure user authentication.
5. Protect passwords using bcrypt hashing.
6. Introduce WhatsApp OTP verification.
7. Implement a permanent 3 × 4 security matrix.
8. Add an additional face-verification layer for administrators.
9. Protect the trained model using SHA-256 integrity verification.
10. Validate patient inputs before prediction.
11. Maintain activity and prediction logs.
12. Examine machine-learning security threats.
13. Implement mechanisms to improve the security of model predictions.
14. Provide separate administrative and regular-user access.

## 2.4 Scope

The project focuses on:

- Machine-learning-based heart disease prediction.
- Secure authentication.
- Role-based access.
- Model integrity.
- Input validation.
- Machine-learning security testing.
- Prediction activity monitoring.
- Administrative security.

The system is intended as an academic prototype and is not a replacement for professional medical diagnosis.

---

# 3. TECHNOLOGIES AND TOOLS USED

| Technology | Purpose |
|---|---|
| Python 3.11.9 | Main programming language |
| Streamlit | Web application interface |
| Scikit-learn | Machine-learning model and preprocessing |
| Random Forest | Heart disease classification |
| Pandas | Dataset processing |
| NumPy | Numerical operations |
| SQLite | User and application data storage |
| bcrypt | Secure password hashing |
| OpenCV 4.10.0 | Face detection and recognition |
| OpenCV Contrib | LBPH face-recognition functionality |
| Haar Cascade | Face detection |
| SHA-256 | Model integrity verification |
| Joblib/Pickle | Model serialization |
| WhatsApp OTP integration | Additional authentication |
| Matplotlib/other supported libraries | Analysis and visualization, where used |

### Development environment

- Operating System: Windows
- IDE: Visual Studio Code
- Python version: 3.11.9
- Virtual environment: Python venv
- Application framework: Streamlit

OpenCV was verified in the virtual environment, including availability of the `cv2.face` module.

---

# 4. DATASET DESCRIPTION

## 4.1 Dataset Source

The project uses the UCI Heart Disease dataset.

The dataset contains:

- 920 records.
- 16 columns.
- Patient-related medical attributes.
- A target variable indicating the presence or absence of heart disease.

The target was converted into a binary classification problem.

The two target classes are:

| Class | Number of records |
|---|---:|
| Heart disease absent | 411 |
| Heart disease present | 509 |
| Total | 920 |

The dataset has a relatively small class imbalance, with a class ratio of approximately 1.24.

## 4.2 Data Preprocessing

The raw dataset cannot always be directly supplied to a machine-learning algorithm.

Therefore, preprocessing is performed before classification.

The preprocessing pipeline includes:

### A. Handling missing values

The dataset contains missing entries.

Missing values are handled through imputation within the machine-learning pipeline.

Imputation replaces missing values with appropriate calculated values rather than simply discarding every incomplete record.

### B. Categorical encoding

Some attributes contain categorical values.

Machine-learning algorithms require numerical representations.

One-hot encoding is used to transform categorical features into numerical columns.

### C. Numerical scaling

Numerical features are passed through a scaling stage in the preprocessing pipeline.

This provides a consistent preprocessing structure.

### D. Target conversion

The original target information is converted into two classes:

- 0 – Heart disease absent.
- 1 – Heart disease present.

The resulting problem is binary classification.

## 4.3 Dataset Quality Analysis

The dataset was also examined for potential quality issues.

The recorded scan reported:

| Dataset property | Result |
|---|---:|
| Total records | 920 |
| Total columns | 16 |
| Missing-value count | 1,759 |
| Duplicate records | 0 |
| Class imbalance ratio | Approximately 1.24 |
| Suspicious dataset flag | False |

The missing-value count refers to the missing entries across the dataset, not the number of incomplete patient records.

---

# 5. MACHINE LEARNING MODEL

## 5.1 Algorithm Used: Random Forest Classifier

The system uses a Random Forest classification model.

Random Forest is an ensemble-learning algorithm that combines predictions from multiple decision trees.

Instead of relying on one decision tree, it creates several trees and combines their results.

For classification, the final prediction is generally determined by majority voting.

### Working principle

1. The training dataset is supplied to the algorithm.
2. Multiple decision trees are constructed.
3. Each tree learns patterns from the training data.
4. Each tree produces a classification.
5. The individual classifications are combined.
6. The final class is selected.

### Why Random Forest?

- Handles different types of features.
- Can model nonlinear relationships.
- Combines multiple decision trees.
- Generally provides good classification performance.
- Reduces dependence on a single decision tree.

## 5.2 Model Pipeline

The project uses a Scikit-learn Pipeline.

The pipeline combines preprocessing and classification.

The overall structure is:

```text
Patient Dataset
       |
       v
Missing Value Imputation
       |
       v
Categorical Feature Encoding
       |
       v
Numerical Feature Scaling
       |
       v
Random Forest Classifier
       |
       v
Predicted Class
```

Using a pipeline helps ensure that the same preprocessing operations are applied consistently.

It also reduces the risk of accidentally applying different transformations during training and prediction.

## 5.3 Model Performance

The recorded final project metrics are:

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 84.78% |
| Precision | 85.58% |
| Recall | 87.25% |
| F1-score | 86.41% |

### Accuracy

Accuracy measures the proportion of correctly classified records.

Accuracy = Correct Predictions / Total Predictions

The model achieved an accuracy of 84.78%.

### Precision

Precision measures how many positive predictions were actually positive.

Precision = TP / (TP + FP)

The precision is 85.58%.

### Recall

Recall measures how many actual positive cases were correctly identified.

Recall = TP / (TP + FN)

The recall is 87.25%.

### F1-score

The F1-score is the harmonic mean of precision and recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

The model achieved an F1-score of 86.41%.

Recall is particularly important in medical prediction because missing a positive case can be significant.

However, these results are dataset-based model evaluation results and should not be interpreted as proof of clinical effectiveness.

---

# 6. SYSTEM ARCHITECTURE

The system consists of four major layers.

## 6.1 Presentation Layer

This layer provides the interface through Streamlit.

It includes:

- Login interface.
- Registration interface.
- OTP verification.
- Matrix verification.
- Face verification for administrators.
- Patient input form.
- Prediction results.
- Administrative dashboard.

## 6.2 Authentication and Authorization Layer

This layer verifies the identity and permissions of users.

It includes:

- Username and password authentication.
- bcrypt password verification.
- WhatsApp OTP verification.
- Security matrix verification.
- Administrative face verification.
- Role-based access control.

## 6.3 Machine Learning Layer

This layer performs the prediction.

It includes:

- Input preprocessing.
- Model loading.
- Model integrity verification.
- Random Forest classification.
- Prediction generation.

## 6.4 Security and Storage Layer

This layer protects the application and stores relevant records.

It includes:

- SQLite database.
- Password hashes.
- SHA-256 model hash.
- Login activity records.
- Prediction activity records.
- Administrative activity logs.
- Input validation.
- Security testing mechanisms.

## 6.5 Overall Architecture Diagram

```text
                 USER / ADMIN
                       |
                       v
              STREAMLIT INTERFACE
                       |
                       v
               LOGIN AUTHENTICATION
                       |
                       v
              PASSWORD VERIFICATION
                       |
                       v
                 WHATSAPP OTP
                       |
                       v
                MATRIX VERIFICATION
                       |
             +---------+----------+
             |                    |
             v                    v
        REGULAR USER             ADMIN
             |                    |
             |                    v
             |              FACE VERIFICATION
             |                    |
             +---------+----------+
                       |
                       v
               AUTHENTICATED SESSION
                       |
                       v
             ROLE-BASED ACCESS CONTROL
                       |
              +--------+--------+
              |                 |
              v                 v
         USER PREDICTION    ADMIN DASHBOARD
              |                 |
              v                 v
       PATIENT INPUT       ADMIN OPERATIONS
              |
              v
        INPUT VALIDATION
              |
              v
       MODEL INTEGRITY CHECK
              |
              v
       PREPROCESSING PIPELINE
              |
              v
       RANDOM FOREST MODEL
              |
              v
        PREDICTION RESULT
              |
              v
         ACTIVITY LOGGING
```

---

# 7. APPLICATION WORKING PROCEDURE

## 7.1 User Registration

New users can register through the application.

The registration process collects the required account information.

The password is not intended to be stored as plain text.

Instead, bcrypt is used to generate a password hash.

The user information is stored in the SQLite database.

The account is assigned the regular-user role.

### Registration flow

```text
       NEW USER
           |
           v
    REGISTRATION FORM
           |
           v
    INPUT VALIDATION
           |
           v
    PASSWORD RECEIVED
           |
           v
      BCRYPT HASHING
           |
           v
     SQLITE DATABASE
           |
           v
     USER ACCOUNT CREATED
```

## 7.2 Login

The user enters their registered username and password.

The application checks the credentials against the stored account information.

The submitted password is verified against the bcrypt hash.

If authentication succeeds, the application proceeds to OTP verification.

If authentication fails, access is denied.

## 7.3 OTP Verification

The next authentication layer is WhatsApp OTP verification.

A one-time password is sent through the configured WhatsApp OTP integration.

The user must enter the correct OTP.

Only after successful verification can the user proceed to the matrix-verification stage.

OTP verification adds an additional authentication factor beyond the password.

## 7.4 Security Matrix Verification

The system uses a permanent 3 × 4 security matrix.

The matrix contains twelve character positions.

For every login attempt, the application generates a challenge containing three requested positions.

For example:

```text
       SECURITY MATRIX

        C1    C2    C3    C4

R1      A     K     P     M

R2      T     B     X     D

R3      H     Q     N     Z
```

Example challenge:

```text
R1C2 → R2C4 → R3C1
```

The user must enter the characters from those positions in the same order.

In the example above, the response would be:

```text
KDH
```

The application compares the entered response with the expected characters.

Three incorrect attempts result in the matrix-verification session being reset and the user being required to log in again.

The permanent matrix is not displayed on the verification screen. The user must use their assigned matrix or card.

## 7.5 Administrative Face Verification

An additional face-verification layer is being integrated specifically for administrators.

Regular users do not undergo face verification.

The administrator must complete:

1. Password authentication.
2. WhatsApp OTP verification.
3. Security matrix verification.
4. Face verification.

The face-recognition component uses OpenCV and the Local Binary Patterns Histograms (LBPH) algorithm.

### Registered face profiles

The authorized administrator profiles include:

- `admin_srishti`
- `admin_vishrutha`

Both profiles are intended to authorize the same Admin account.

The recognition system loads the images from the administrator profile folders and trains an LBPH recognizer.

### Face recognition working

1. The camera captures an image.
2. OpenCV decodes the image.
3. The image is converted into grayscale.
4. Haar Cascade detects the face.
5. The largest detected face is selected.
6. The face is resized to 200 × 200 pixels.
7. Histogram equalization is applied.
8. The LBPH recognizer compares the captured face with the trained profiles.
9. The recognition distance is compared with the configured threshold.
10. If the result satisfies the threshold, face verification succeeds.

The configured threshold is 75.

A lower LBPH distance represents a closer match.

### Face-recognition flowchart

```text
          ADMIN PASSES MATRIX
                   |
                   v
          FACE VERIFICATION PAGE
                   |
                   v
             CAMERA CAPTURE
                   |
                   v
          IMAGE CONVERSION
                   |
                   v
          GRAYSCALE IMAGE
                   |
                   v
          HAAR FACE DETECTION
                   |
                   v
          FACE EXTRACTION
                   |
                   v
          RESIZE TO 200 × 200
                   |
                   v
          HISTOGRAM EQUALIZATION
                   |
                   v
            LBPH RECOGNITION
                   |
                   v
          COMPARE DISTANCE
                   |
             +-----+-----+
             |           |
             v           v
         ACCEPTED      REJECTED
             |           |
             v           v
        ADMIN LOGIN   RETRY / REJECT
```

### Important limitation

The current LBPH implementation does not provide reliable liveness detection.

A photograph or replayed video may potentially bypass a basic camera-based face-recognition mechanism.

Therefore, this feature is an additional authentication layer for the academic prototype, not a guarantee against sophisticated biometric spoofing.

The integration has passed the Python syntax check, but its complete runtime operation still requires testing.

---

# 8. SECURITY ARCHITECTURE – MAIN PROJECT FOCUS

Security is the most important additional component of this project.

The system does not rely on a single security mechanism.

Instead, it uses multiple security layers to protect different parts of the application.

These layers address:

1. Authentication security.
2. Authorization security.
3. Password security.
4. Session security.
5. Model integrity.
6. Input security.
7. Machine-learning attack resistance.
8. Data privacy.
9. Activity monitoring.
10. Administrative security.

## 8.1 Multi-Factor Authentication

Multi-factor authentication means verifying a user's identity through multiple independent authentication mechanisms.

The system combines several verification stages.

### Regular-user authentication

```text
Username and Password
         |
         v
    WhatsApp OTP
         |
         v
  Security Matrix
         |
         v
   User Dashboard
```

### Administrator authentication

```text
Username and Password
         |
         v
    WhatsApp OTP
         |
         v
  Security Matrix
         |
         v
   Face Recognition
         |
         v
  Admin Dashboard
```

This provides different authentication requirements according to the account role.

The administrative account has an additional biometric verification stage.

## 8.2 Password Security Using bcrypt

Passwords are sensitive authentication information.

Storing passwords directly in a database creates a serious security risk.

If the database is exposed, plain-text passwords can be immediately read.

The project uses bcrypt password hashing.

### How bcrypt works

1. The user creates a password.
2. bcrypt processes the password.
3. A salt is incorporated into the hashing process.
4. A password hash is generated.
5. The hash is stored in SQLite.
6. During login, the entered password is checked against the stored hash.

### Password security flow

```text
          USER PASSWORD
                 |
                 v
            BCRYPT
                 |
                 v
          PASSWORD HASH
                 |
                 v
           SQLITE STORAGE
                 |
                 |
         USER LOGS IN AGAIN
                 |
                 v
        ENTERED PASSWORD
                 |
                 v
         BCRYPT VERIFICATION
                 |
           +-----+-----+
           |           |
           v           v
        MATCH         NO MATCH
           |           |
           v           v
        CONTINUE     DENY ACCESS
```

### Security benefit

Even if an attacker obtains the password database, the stored hashes do not directly reveal the original passwords.

bcrypt is deliberately computationally expensive compared with ordinary fast hashing functions.

This makes large-scale password guessing more difficult.

Importantly, bcrypt is a password-hashing algorithm, not an encryption algorithm.

## 8.3 Role-Based Access Control

The system distinguishes between regular users and administrators.

Each account is assigned a role.

### Regular user permissions

- Access the prediction application.
- Enter patient information.
- Obtain prediction results.
- Access permitted user functionality.

### Administrator permissions

- Access administrative functionality.
- View administrative information.
- Access authorized management features.
- Access the administrative dashboard.

The application checks the authenticated role before displaying administrative functionality.

### Role-based access flow

```text
       AUTHENTICATED USER
               |
               v
         CHECK USER ROLE
               |
          +----+----+
          |         |
          v         v
        USER      ADMIN
          |         |
          v         v
      USER PAGE  ADMIN PAGE
```

This reduces the risk of ordinary users accessing administrative functionality.

The role should also be checked when performing protected operations, not merely when hiding interface elements.

## 8.4 WhatsApp OTP Security

The OTP mechanism provides an additional verification stage.

A password alone may be compromised through phishing, password reuse, or credential theft.

OTP verification introduces a second possession-based check.

The process is:

1. The user enters their credentials.
2. The application validates the credentials.
3. An OTP is generated through the configured integration.
4. The OTP is delivered through WhatsApp.
5. The user enters the received code.
6. The application verifies the code.
7. Successful verification allows the next authentication stage.

The OTP mechanism should use expiration, attempt limits, and secure secret handling.

These controls should be verified in the final implementation before claiming complete protection against OTP abuse.

## 8.5 Permanent Security Matrix

The matrix provides an additional challenge-response mechanism.

It uses a 3 × 4 matrix containing twelve character positions.

A random set of three positions is requested during login.

The user must supply the characters corresponding to the requested positions.

### Security benefits

- Adds another verification stage.
- Uses a challenge that can vary between login attempts.
- Does not display the permanent matrix on the verification screen.
- Limits incorrect attempts to three.

The mechanism is intended to make simple credential-only access more difficult.

However, the matrix itself must be protected from theft, copying, and disclosure.

## 8.6 SHA-256 Model Integrity Verification

This is one of the most important security mechanisms in the project.

A machine-learning model is stored as a serialized file.

An attacker who can modify the model file may be able to alter the application's predictions without changing the visible application interface.

For example, an attacker could replace the original model with a modified model that produces misleading results.

To detect this, the project uses SHA-256 hashing.

### What is SHA-256?

SHA-256 is a cryptographic hash function.

It accepts input data of any supported size and produces a fixed-length hash of 256 bits.

The resulting hexadecimal representation contains 64 characters.

A small change in the input file should produce a substantially different hash.

### Working procedure

During the trusted setup stage:

1. The original model file is selected.
2. SHA-256 is calculated.
3. The trusted hash is recorded.

During application execution:

1. The model file is accessed.
2. SHA-256 is calculated again.
3. The new hash is compared with the trusted hash.
4. If the values match, the integrity check passes.
5. If the values differ, the application detects a possible modification and blocks the prediction workflow.

### Model integrity flowchart

```text
          MODEL FILE
               |
               v
       CALCULATE SHA-256
               |
               v
       CURRENT FILE HASH
               |
               v
       COMPARE WITH TRUSTED HASH
               |
          +----+----+
          |         |
          v         v
        MATCH     MISMATCH
          |         |
          v         v
      INTEGRITY   POSSIBLE
       PASSED     TAMPERING
          |         |
          v         v
      CONTINUE    BLOCK ACCESS
      PREDICTION  / ALERT
```

### Project model hash

The recorded SHA-256 value for the current model file is:

`b1ee71af273771274d9803587655418bbc414d2e008c5c55c7720733f45eaf67`

This value is associated with the recorded model version.

If the model is legitimately retrained or replaced, the trusted hash must be updated through a controlled process.

### Security importance

This mechanism helps detect:

- Unauthorized model replacement.
- Accidental model modification.
- File corruption.
- Unexpected changes to the serialized model.

### Important limitation

A hash stored in a file that an attacker can modify along with the model is not sufficient protection against a privileged attacker.

For stronger protection, the trusted hash should be stored separately or protected through a digital signature or trusted deployment mechanism.

SHA-256 verifies file integrity; it does not establish that the original model was trained on trustworthy data.

## 8.7 Input Validation

Input validation protects the prediction system from malformed or unexpected patient information.

The application receives patient information through a form.

Without validation, invalid inputs may cause:

- Runtime errors.
- Incorrect preprocessing.
- Unexpected model behaviour.
- Manipulation of prediction results.

The project includes schema-based input validation.

The validation process checks whether the submitted information is consistent with the expected input structure.

### Input-validation flow

```text
       PATIENT INPUT
             |
             v
      INPUT RECEIVED
             |
             v
      SCHEMA VALIDATION
             |
       +-----+-----+
       |           |
       v           v
     VALID       INVALID
       |           |
       v           v
   CONTINUE     REJECT INPUT
       |
       v
   PREPROCESSING
       |
       v
   MODEL PREDICTION
```

Validation should include:

- Required fields.
- Appropriate data types.
- Permitted categorical values.
- Valid numerical ranges.
- Missing or malformed values.

## 8.8 Activity Logging

The application records security-related activities.

Logging is important because authentication and prediction operations should be traceable.

The recorded activity categories include:

- Login activities.
- Prediction activities.
- Administrative actions.
- Other relevant security events.

### Why logging is important

1. Helps identify repeated failed logins.
2. Supports monitoring of user activity.
3. Helps administrators investigate suspicious operations.
4. Provides a record of prediction activity.
5. Supports auditing.

The project uses activity logs as part of its security monitoring architecture.

Logs themselves must be protected against unauthorized modification and should avoid unnecessarily storing sensitive patient information.

---

# 9. MACHINE-LEARNING SECURITY

Traditional application security protects accounts, files, and interfaces.

However, machine-learning systems have additional threats because attackers may target the training data, model behaviour, or prediction interface.

This project specifically examines machine-learning-related security threats.

## 9.1 Model Poisoning Attack

### Definition

A model poisoning attack occurs when an attacker manipulates training data or the training process to influence the model's learned behaviour.

An attacker may introduce incorrect records, modify labels, or manipulate selected features.

### Example

Suppose a heart disease dataset contains patient records and corresponding labels.

An attacker modifies selected records so that the model learns incorrect relationships between patient attributes and heart disease.

The resulting model may produce unreliable predictions.

### Project testing

A poisoning-related baseline test was run.

The recorded result indicated that the baseline poisoning check passed.

This should be described as a successful baseline check, not proof that the model is immune to every poisoning attack.

### Protection approach

- Examine dataset quality.
- Check missing values.
- Check duplicate records.
- Examine class distribution.
- Keep training data controlled.
- Protect the trained model.
- Re-evaluate the model after retraining.

## 9.2 Adversarial Attack

### Definition

An adversarial attack attempts to manipulate model predictions by making small changes to the input data.

The modified input may appear similar to the original input but may cause the model to produce a different result.

### Example

A patient record may initially contain:

```text
oldpeak = 1.0
thalch = 144
```

A modified record may contain:

```text
oldpeak = 0.0
thalch = 154
```

The model's prediction can be examined to determine whether such changes produce unexpected behaviour.

### Project testing

The recorded adversarial testing identified examples involving:

- `oldpeak`: 1.0 → 0.0
- `thalch`: 144 → 154

These examples were identified during adversarial testing.

The project also examined prediction stability.

One recorded instability involved:

`thalch`: 150 → 153

This indicates that small input changes can affect prediction behaviour and deserve further investigation.

### Protection approach

- Validate input ranges.
- Examine prediction sensitivity.
- Compare predictions before and after controlled input changes.
- Identify unstable features.
- Investigate unusually sensitive model behaviour.

A stability test can help identify suspicious sensitivity, but it does not guarantee robustness against all adversarial examples.

## 9.3 Model Extraction Attack

### Definition

A model extraction attack attempts to reproduce or approximate a machine-learning model by repeatedly querying its prediction interface.

An attacker may submit multiple inputs and collect the corresponding outputs.

The collected input-output pairs can then be used to train a substitute model.

### Example

An attacker repeatedly submits different patient records.

The application returns predictions.

The attacker collects enough responses to approximate the original model's decision behaviour.

### Project testing status

The extraction-attack testing module was started.

However, the recorded execution encountered a pandas median-related error.

Therefore, the extraction test should currently be classified as incomplete.

It would be inaccurate to claim that the system has successfully defeated model extraction.

### Possible protection mechanisms

- Restrict prediction access.
- Monitor repeated prediction requests.
- Apply appropriate rate limits.
- Log unusual query patterns.
- Avoid exposing unnecessary model internals.
- Monitor repeated queries with small feature variations.

## 9.4 Membership Inference Attack

### Definition

A membership inference attack attempts to determine whether a particular record was included in the model's training dataset.

For example, an attacker may try to discover whether a particular patient's record was used during model training.

This is a privacy concern because training membership can itself reveal sensitive information.

### Relevance to the project

The project considers membership inference as part of its machine-learning security objectives.

A complete membership-inference evaluation requires a defined attack implementation, a suitable evaluation dataset, and measured attack performance.

The available recorded results do not establish a completed membership-inference experiment.

It should therefore be listed as an evaluated security concern or pending test unless the final code and results demonstrate otherwise.

## 9.5 Differential-Privacy-Inspired Noise

### Definition

Differential privacy is a mathematical framework for limiting how much information about an individual record can be revealed through an analysis.

It introduces carefully calibrated randomness into a computation or output.

The purpose is to reduce the ability to infer information about a specific individual.

### Project approach

The project includes a differential-privacy-inspired noise mechanism in its prediction security design.

The purpose is to explore how introducing controlled noise may reduce information leakage.

### Important distinction

Adding random noise alone does not establish formal differential privacy.

A formal differential privacy guarantee requires:

- A defined privacy mechanism.
- A defined sensitivity calculation.
- A calibrated noise distribution.
- A privacy parameter such as epsilon.
- Appropriate accounting for repeated queries.

Unless these are implemented and documented, the correct description is:

**Differential-privacy-inspired noise mechanism, not a formally verified differential privacy guarantee.**

---

# 10. SECURITY TESTING AND OBSERVED RESULTS

Security testing is necessary to determine whether the implemented mechanisms behave as expected.

The recorded observations are:

| Security test | Recorded observation | Status |
|---|---|---|
| Dataset quality scan | 920 records, zero duplicates, suspicious flag false | Completed |
| Password protection | bcrypt-based password storage implemented | Implemented |
| Role-based access | Admin and user roles implemented | Implemented |
| Model integrity | SHA-256 integrity verification implemented | Implemented |
| Matrix verification | Three-position challenge and three-attempt limit | Implemented |
| Poisoning baseline | Baseline check passed | Tested |
| Adversarial testing | Feature changes identified | Tested |
| Prediction stability | Instability found for a small `thalch` change | Tested |
| Model extraction | Pandas median error occurred | Incomplete |
| Membership inference | Security concern identified; completion not established | Pending verification |
| Face recognition | LBPH integration added and syntax-checked | Runtime testing pending |

### Interpretation

The results demonstrate that the project goes beyond basic machine-learning prediction.

It includes application-level security controls and initial machine-learning security experiments.

However, identifying a vulnerability or passing a baseline test is not equivalent to demonstrating complete attack resistance.

Further testing is required to establish the effectiveness of each defence.

---

# 11. SOFTWARE MODULE STRUCTURE

The project is organized into separate modules so that different application responsibilities can be maintained independently.

The main known components include:

| File / Module | Responsibility |
|---|---|
| `app.py` | Main Streamlit application and navigation |
| `utils.py` | Shared application utility functions |
| `predict.py` | Prediction-related functionality |
| `login.py` | Login-related functionality |
| `database.py` | SQLite database operations |
| `matrix_security.py` | Permanent matrix and challenge verification |
| `security.py` | Security-related operations, where included |
| `face_auth_streamlit.py` | Streamlit-compatible face recognition |
| `admin_verification/` | Face profiles and Haar Cascade resources |
| `heart_model.pkl` | Serialized machine-learning pipeline |
| `venv/` | Python virtual environment |

The exact module contents should be matched against the final project folder before submission.

## 11.1 Main Application – app.py

The main application controls the Streamlit interface.

Its responsibilities include:

- Session initialization.
- Login-page routing.
- OTP verification.
- Matrix verification.
- Face-verification routing.
- User and administrator session handling.
- Prediction-page access.
- Administrative dashboard access.

The application maintains session information to determine which authentication stage is currently active.

## 11.2 Database Module

SQLite is used to store account-related information and application records.

The database supports:

- User accounts.
- Password hashes.
- User roles.
- Relevant activity records.

Password hashes are stored instead of plain-text passwords.

## 11.3 Matrix Security Module

This module is responsible for:

- Generating or retrieving the permanent matrix.
- Generating challenge positions.
- Verifying submitted matrix responses.
- Supporting the failed-attempt limit.

## 11.4 Prediction Module

The prediction component receives patient information and uses the trained machine-learning pipeline to generate the output.

It should only be accessed after the required authentication and security checks.

## 11.5 Face Authentication Module

`face_auth_streamlit.py` adapts the face-recognition functionality for Streamlit.

It includes:

- Face detector initialization.
- Face extraction.
- Image normalization.
- Image augmentation for training.
- LBPH recognizer training.
- Camera-image verification.

The face profiles are stored under:

```text
admin_verification/
    admins/
        admin_srishti/
        admin_vishrutha/
    cascades/
        haarcascade_frontalface_default.xml
```

Both registered profiles are intended to authorize the same administrator account.

---

# 12. COMPLETE APPLICATION FLOWCHART

The following diagram represents the intended complete working flow.

```text
                  START
                    |
                    v
             OPEN APPLICATION
                    |
                    v
               LOGIN PAGE
                    |
                    v
           ENTER USER CREDENTIALS
                    |
                    v
           VERIFY PASSWORD HASH
                    |
              +-----+-----+
              |           |
              v           v
           INVALID       VALID
              |           |
              v           v
          DENY ACCESS   SEND OTP
                          |
                          v
                    VERIFY OTP
                          |
                    +-----+-----+
                    |           |
                    v           v
                 INVALID       VALID
                    |           |
                    v           v
                RETRY /       MATRIX
                 REJECT     VERIFICATION
                                |
                          +-----+-----+
                          |           |
                          v           v
                       INVALID       VALID
                          |           |
                          v           v
                       RETRY       CHECK ROLE
                                      |
                              +-------+-------+
                              |               |
                              v               v
                            USER            ADMIN
                              |               |
                              |               v
                              |        FACE VERIFICATION
                              |               |
                              |         +-----+-----+
                              |         |           |
                              |         v           v
                              |      REJECTED    ACCEPTED
                              |         |           |
                              |         v           |
                              |       RETRY         |
                              |         |           |
                              +---------+-----------+
                                        |
                                        v
                               AUTHENTICATED SESSION
                                        |
                                        v
                               ACCESS CONTROL CHECK
                                        |
                                        v
                                 PREDICTION PAGE
                                        |
                                        v
                                 PATIENT INPUT
                                        |
                                        v
                                 INPUT VALIDATION
                                        |
                                        v
                               SHA-256 MODEL CHECK
                                        |
                                  +-----+-----+
                                  |           |
                                  v           v
                                FAIL        PASS
                                  |           |
                                  v           v
                              BLOCK       PREPROCESS
                              ACCESS          |
                                              v
                                      RANDOM FOREST
                                              |
                                              v
                                       PREDICTION
                                              |
                                              v
                                        LOG EVENT
                                              |
                                              v
                                            END
```

---

# 13. SECURITY LAYER SUMMARY

The security architecture can be summarized as follows:

| Security Layer | Threat Addressed | Mechanism |
|---|---|---|
| Password security | Password disclosure | bcrypt hashing |
| OTP verification | Credential-only impersonation | WhatsApp OTP |
| Matrix verification | Unauthorized access | Random position challenge |
| Face verification | Unauthorized administrator access | OpenCV LBPH |
| Role-based access | Privilege misuse | Admin/user role checks |
| Model integrity | Model file tampering | SHA-256 |
| Input validation | Malformed or manipulated inputs | Schema validation |
| Activity logging | Untraceable activity | Login and action logs |
| Poisoning analysis | Training data manipulation | Baseline testing and data inspection |
| Adversarial analysis | Prediction manipulation | Feature perturbation and stability checks |
| Extraction analysis | Model replication | Query monitoring and attack testing |
| Privacy analysis | Training-record inference | Membership-inference considerations |
| Privacy mechanism | Information leakage | Differential-privacy-inspired noise |

The security mechanisms operate at different stages.

For example:

- bcrypt protects account credentials.
- OTP and matrix verification protect account access.
- Face recognition adds another administrative verification layer.
- SHA-256 protects the model file against undetected modification.
- Input validation protects the prediction interface.
- Activity logs support security monitoring.
- Machine-learning security tests examine threats to model behaviour and privacy.

No individual mechanism protects the entire system. Their combined use creates a defence-in-depth architecture.

---

# 14. ADVANTAGES OF THE PROJECT

1. Combines machine learning and cybersecurity.
2. Uses a Random Forest classification model.
3. Includes a complete preprocessing pipeline.
4. Uses bcrypt instead of plain-text password storage.
5. Provides multi-stage authentication.
6. Uses a permanent security matrix.
7. Provides additional administrator face verification.
8. Includes role-based access control.
9. Detects unexpected model-file modifications using SHA-256.
10. Validates patient inputs.
11. Maintains activity logs.
12. Examines machine-learning-specific security threats.
13. Uses a modular application structure.
14. Provides an accessible Streamlit interface.

---

# 15. LIMITATIONS

Although the project contains several security features, certain limitations remain.

## 15.1 Model Limitations

The Random Forest model is trained on a limited dataset.

Its predictions depend on the quality and representativeness of the training data.

The reported metrics do not establish clinical reliability.

## 15.2 Dataset Limitations

The dataset contains missing values and may not represent every population or clinical setting.

Additional external validation would be required before any real-world medical use.

## 15.3 Face Recognition Limitations

The LBPH algorithm is relatively simple.

Its accuracy may be affected by:

- Lighting.
- Camera quality.
- Face angle.
- Image quality.
- Changes in appearance.

The current implementation does not provide reliable liveness detection.

## 15.4 Model Integrity Limitations

SHA-256 detects file changes when compared against a trusted reference hash.

It does not protect against an attacker who can modify both the model and the trusted hash.

## 15.5 Machine-Learning Security Limitations

The recorded extraction test encountered an error.

Membership-inference evaluation is not confirmed complete.

The differential-privacy-inspired mechanism should not be described as formally guaranteed differential privacy without mathematical validation.

## 15.6 Deployment Limitations

The application is currently an academic prototype.

A production deployment would require additional security controls, infrastructure hardening, secure secret management, and privacy review.

---

# 16. FUTURE ENHANCEMENTS

Possible future improvements include:

1. Implement reliable liveness detection for face verification.
2. Introduce signed model artifacts.
3. Store trusted integrity references outside the application directory.
4. Implement comprehensive model extraction testing.
5. Complete membership-inference evaluation.
6. Improve adversarial robustness.
7. Introduce formally calibrated differential privacy if appropriate.
8. Add rate limiting for prediction requests.
9. Improve audit-log protection.
10. Conduct systematic security testing.
11. Perform external model validation.
12. Improve the user interface and reporting.
13. Introduce secure backup and recovery mechanisms.
14. Add controlled model versioning.

---

# 17. CONCLUSION

The Secure Heart Disease Prediction System Using Machine Learning demonstrates the integration of predictive analytics and cybersecurity within a healthcare-oriented application.

The system uses a Random Forest classifier trained on the UCI Heart Disease dataset and provides a Streamlit interface for authenticated users to submit patient information and receive predictions.

The major contribution of the project is its multi-layer security architecture.

At the application level, bcrypt password hashing, WhatsApp OTP verification, a permanent security matrix, role-based access control, and additional administrator face verification provide multiple layers of authentication and authorization.

At the model level, SHA-256 integrity verification helps detect unauthorized modification of the serialized machine-learning model.

Input validation and activity logging provide additional safeguards and support monitoring.

The project also examines machine-learning-specific threats, including poisoning, adversarial manipulation, model extraction, and membership inference.

The recorded model achieved 84.78% accuracy, 85.58% precision, 87.25% recall, and an F1-score of 86.41%.

The security experiments identified adversarial sensitivity and highlighted areas requiring further investigation.

Some security evaluations, particularly model extraction and membership inference, require completion and verification. The administrative face-recognition integration also requires full runtime testing.

Overall, the project demonstrates that a healthcare machine-learning application should be designed with security as an essential component rather than treating security as an optional feature.

The system provides an academic foundation for exploring secure machine learning, healthcare prediction, authentication, model integrity, and protection against machine-learning-related threats.

Contact

Developer: Srishti S Rao

Degree: B.Tech – Electronics and Computer  Engineering (AI/ML Specialization)

Institution: Christ University (Deemed to be University)

Location: Bengaluru, India

Email: srishtisrao26@gmail.com

LinkedIn: https://www.linkedin.com/in/srishti-s-rao-b09189316

GitHub: https://github.com/srishtii26

⭐ If you find this project interesting, consider giving the repository a star!
