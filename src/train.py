import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


# ==========================================
# PATHS
# ==========================================

DATA_PATH = "data/processed/features.csv"
MODEL_PATH = "models/attention_model.pkl"


# ==========================================
# CHECK DATASET
# ==========================================

if not os.path.exists(DATA_PATH):

    print("ERROR: Dataset not found!")
    print(DATA_PATH)

    exit()


# ==========================================
# LOAD DATASET
# ==========================================

print()
print("=" * 60)
print(" STUDENT ATTENTION ML TRAINING")
print("=" * 60)

print()
print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print(
    f"Dataset shape: {data.shape}"
)


# ==========================================
# FEATURE COLUMNS
# ==========================================

FEATURES = [

    "left_ear",
    "right_ear",
    "average_ear",

    "head_turn",
    "head_vertical",
    "head_tilt",

    "nose_x_normalized",
    "nose_y_normalized",
    "nose_z_normalized",

    "mouth_opening",

    "eye_center_distance",

    "face_width",
    "face_height",

    "face_detected"

]


TARGET = "label"


# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

missing_columns = [

    column

    for column in FEATURES + [TARGET]

    if column not in data.columns

]


if missing_columns:

    print()
    print("ERROR: Missing columns:")

    for column in missing_columns:
        print("-", column)

    exit()


# ==========================================
# REMOVE INVALID ROWS
# ==========================================

data = data.dropna(
    subset=FEATURES + [TARGET]
)


print(
    f"Valid samples: {len(data)}"
)


# ==========================================
# INPUT AND OUTPUT
# ==========================================

X = data[FEATURES]

y = data[TARGET]


print()
print("Classes:")

print(
    y.value_counts()
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


print()
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

print()
print("Creating Random Forest model...")


model = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=1,

    max_features="sqrt",

    class_weight="balanced",

    random_state=42,

    n_jobs=-1

)


# ==========================================
# TRAIN
# ==========================================

print()
print("Training model...")

model.fit(
    X_train,
    y_train
)


print("Training completed!")


# ==========================================
# PREDICTION
# ==========================================

print()
print("Evaluating model...")


predictions = model.predict(
    X_test
)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)


print()
print("=" * 60)
print(" MODEL PERFORMANCE")
print("=" * 60)

print()

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print()
print("Classification Report")
print("-" * 60)

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print()
print("Confusion Matrix")
print("-" * 60)

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print()
print("Feature Importance")
print("-" * 60)


importance = pd.DataFrame({

    "Feature": FEATURES,

    "Importance":
        model.feature_importances_

})


importance = importance.sort_values(

    by="Importance",

    ascending=False

)


print(
    importance.to_string(
        index=False
    )
)


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


# ==========================================
# FINAL MESSAGE
# ==========================================

print()
print("=" * 60)

print(
    "MODEL SAVED SUCCESSFULLY!"
)

print()

print(
    f"Location: {MODEL_PATH}"
)

print("=" * 60)