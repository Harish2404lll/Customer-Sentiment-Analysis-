import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

train_df = pd.read_csv("data/train_clean.csv")
val_df = pd.read_csv("data/validation_clean.csv")
test_df = pd.read_csv("data/test_clean.csv")


print("Training data:", train_df.shape)
print("Validation data:", val_df.shape)
print("Testing data:", test_df.shape)


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
# 3. CREATE TF-IDF VECTORIZER
# ==========================================

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


# ==========================================
# 4. FIT ONLY ON TRAINING DATA
# ==========================================

X_train = tfidf.fit_transform(X_train_text)


# ==========================================
# 5. TRANSFORM VALIDATION AND TEST DATA
# ==========================================

X_val = tfidf.transform(X_val_text)

X_test = tfidf.transform(X_test_text)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\nTF-IDF completed successfully!")

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

print("\nNumber of TF-IDF features:")
print(len(tfidf.get_feature_names_out()))

print("\nFirst 20 features:")
print(tfidf.get_feature_names_out()[:20])