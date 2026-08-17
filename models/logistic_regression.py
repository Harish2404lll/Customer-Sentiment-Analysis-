import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATA
# ==========================================

train_df = pd.read_csv("data/train_clean.csv")
val_df = pd.read_csv("data/validation_clean.csv")
test_df = pd.read_csv("data/test_clean.csv")


print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)


# ==========================================
# 2. SEPARATE TEXT AND LABEL
# ==========================================

X_train_text = train_df["clean_text"]
y_train = train_df["label"]

X_val_text = val_df["clean_text"]
y_val = val_df["label"]

X_test_text = test_df["clean_text"]
y_test = test_df["label"]


# ==========================================
# 3. TF-IDF
# ==========================================

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


X_train = tfidf.fit_transform(X_train_text)

X_val = tfidf.transform(X_val_text)

X_test = tfidf.transform(X_test_text)


print("\nTF-IDF:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==========================================
# 4. LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(
    max_iter=1000
)


# ==========================================
# 5. TRAIN
# ==========================================

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 6. VALIDATION
# ==========================================

y_val_pred = model.predict(X_val)

val_accuracy = accuracy_score(
    y_val,
    y_val_pred
)

val_f1 = f1_score(
    y_val,
    y_val_pred,
    average="macro"
)


print("\nValidation Results")
print("------------------")

print("Accuracy:", val_accuracy)
print("Macro F1:", val_f1)

print("\nValidation Classification Report:")

print(
    classification_report(
        y_val,
        y_val_pred
    )
)


# ==========================================
# 7. TEST
# ==========================================

y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)


print("\nTest Results")
print("------------")

print("Accuracy:", test_accuracy)
print("Macro F1:", test_f1)

print("\nTest Classification Report:")

print(
    classification_report(
        y_test,
        y_test_pred
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=[
        "negative",
        "neutral",
        "positive"
    ]
)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "negative",
        "neutral",
        "positive"
    ]
)

disp.plot()

plt.title(
    "Logistic Regression - Test Confusion Matrix"
)

plt.tight_layout()

plt.show()


# ==========================================
# 9. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

joblib.dump(
    tfidf,
    "models/tfidf_logistic.pkl"
)


print("\n========================================")
print("Models saved successfully!")
print("========================================")

print(
    "models/logistic_regression.pkl"
)

print(
    "models/tfidf_logistic.pkl"
)