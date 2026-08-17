import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

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


# ==========================================
# 2. TEXT AND LABEL
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


# ==========================================
# 4. LINEAR SVM
# ==========================================

model = LinearSVC(
    C=1.0,
    class_weight="balanced"
)


# ==========================================
# 5. TRAIN
# ==========================================

print("Training Linear SVM...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 6. VALIDATION
# ==========================================

y_val_pred = model.predict(X_val)

print("\nValidation Results")
print("------------------")

print(
    "Accuracy:",
    accuracy_score(y_val, y_val_pred)
)

print(
    "Macro F1:",
    f1_score(
        y_val,
        y_val_pred,
        average="macro"
    )
)

print("\nClassification Report:")

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

test_macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro"
)

print("\nTest Results")
print("------------")

print("Accuracy:", test_accuracy)

print("Macro F1:", test_macro_f1)

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
    labels=["negative", "neutral", "positive"]
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

plt.title("Linear SVM - Test Confusion Matrix")

plt.tight_layout()

plt.show()

joblib.dump(model, "models/svm.pkl")
joblib.dump(tfidf, "models/tfidf_svm.pkl")

print("SVM model saved.")