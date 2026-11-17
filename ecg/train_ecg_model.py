import os
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SECURE HEART - IMPROVED ECG MODEL TRAINING
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "ecg",
    "cardiac_images",
    "ECG_DATA"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "ecg"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ecg_model_v2.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names_v2.json"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

NUM_CLASSES = 4


# ============================================================
# START
# ============================================================

print("=" * 65)
print("SECURE HEART - IMPROVED ECG MODEL TRAINING")
print("=" * 65)

print("\nDataset:")
print(DATASET_DIR)

print("\nTraining:")
print(TRAIN_DIR)

print("\nTesting:")
print(TEST_DIR)


# ============================================================
# VERIFY DIRECTORIES
# ============================================================

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(
        f"Testing directory not found:\n{TEST_DIR}"
    )

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TRAIN DATA
# ============================================================

print("\nLoading training images...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading testing images...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\nDetected classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nClass mapping saved:")
print(CLASS_NAMES_PATH)


# ============================================================
# DATA PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    AUTOTUNE
)


# ============================================================
# CALCULATE CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

train_labels = []

for _, labels in train_dataset.unbatch():
    train_labels.append(int(labels.numpy()))

train_labels = np.array(train_labels)

class_indices = np.arange(
    len(class_names)
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_indices,
    y=train_labels
)

class_weights = {
    int(i): float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:")

for i, weight in class_weights.items():
    print(
        f"{i} ({class_names[i]}): "
        f"{weight:.4f}"
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(
            factor=0.02
        ),

        layers.RandomZoom(
            height_factor=0.05,
            width_factor=0.05
        ),

        layers.RandomTranslation(
            height_factor=0.02,
            width_factor=0.02
        ),

        layers.RandomContrast(
            factor=0.10
        )
    ],
    name="ecg_augmentation"
)


# ============================================================
# EFFICIENTNET BACKBONE
# ============================================================

print("\nLoading EfficientNetB0...")

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)

base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="ecg_image"
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dropout(
    0.40
)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="prediction"
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="SecureECGEfficientNet"
)


# ============================================================
# COMPILE - INITIAL TRAINING
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel created successfully.")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)


# ============================================================
# PHASE 1 - TRAIN CLASSIFIER
# ============================================================

print("\n")
print("=" * 65)
print("PHASE 1 - TRAINING CLASSIFICATION HEAD")
print("=" * 65)

history_1 = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)


# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n")
print("=" * 65)
print("PHASE 2 - FINE TUNING")
print("=" * 65)


# Unfreeze the backbone
base_model.trainable = True


#