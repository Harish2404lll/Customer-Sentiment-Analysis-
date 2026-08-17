import streamlit as st
import joblib
import re


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():

    logistic_model = joblib.load(
        "models/logistic_regression.pkl"
    )

    logistic_tfidf = joblib.load(
        "models/tfidf_logistic.pkl"
    )

    svm_model = joblib.load(
        "models/svm.pkl"
    )

    svm_tfidf = joblib.load(
        "models/tfidf_svm.pkl"
    )

    return (
        logistic_model,
        logistic_tfidf,
        svm_model,
        svm_tfidf
    )


# Load models
try:

    (
        logistic_model,
        logistic_tfidf,
        svm_model,
        svm_tfidf
    ) = load_models()

except Exception as e:

    st.error("Unable to load the trained models.")

    st.write(
        "Make sure the following files exist:"
    )

    st.code(
        """
models/logistic_regression.pkl
models/tfidf_logistic.pkl
models/svm.pkl
models/tfidf_svm.pkl
        """
    )

    st.stop()


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove unwanted characters
    text = re.sub(
        r"[^a-zA-Z0-9\s']",
        "",
        text
    )

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# =========================================================
# HEADER
# =========================================================

st.title("💬 Customer Sentiment Analysis")

st.markdown(
    """
Analyze customer reviews using two machine-learning models:

- **Logistic Regression**
- **Linear SVM**

The system classifies the review as:

**Negative · Neutral · Positive**
"""
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Model Information")

st.sidebar.info(
    """
Dataset:
jbeno/sentiment_merged

Features:
TF-IDF

Models:
Logistic Regression
Linear SVM

Classes:
Negative
Neutral
Positive
"""
)


# =========================================================
# INPUT
# =========================================================

st.subheader("Enter Customer Review")

review = st.text_area(
    "Customer Review",
    placeholder="Example: The product quality is excellent and I am very happy with my purchase.",
    height=150
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button(
    "🔍 Analyze Sentiment",
    type="primary"
):

    if not review.strip():

        st.warning(
            "Please enter a customer review."
        )

    else:

        # ---------------------------------------------
        # CLEAN TEXT
        # ---------------------------------------------

        cleaned_review = clean_text(review)


        # ---------------------------------------------
        # LOGISTIC REGRESSION
        # ---------------------------------------------

        logistic_input = logistic_tfidf.transform(
            [cleaned_review]
        )

        logistic_prediction = logistic_model.predict(
            logistic_input
        )[0]


        # ---------------------------------------------
        # LOGISTIC PROBABILITY
        # ---------------------------------------------

        logistic_probability = logistic_model.predict_proba(
            logistic_input
        )[0]

        logistic_classes = logistic_model.classes_


        # ---------------------------------------------
        # SVM
        # ---------------------------------------------

        svm_input = svm_tfidf.transform(
            [cleaned_review]
        )

        svm_prediction = svm_model.predict(
            svm_input
        )[0]


        # ---------------------------------------------
        # DISPLAY INPUT
        # ---------------------------------------------

        st.subheader("Review")

        st.write(review)


        # ---------------------------------------------
        # RESULTS
        # ---------------------------------------------

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)


        # =============================================
        # LOGISTIC REGRESSION
        # =============================================

        with col1:

            st.markdown(
                "### Logistic Regression"
            )

            if logistic_prediction == "positive":

                st.success(
                    f"😊 {logistic_prediction.upper()}"
                )

            elif logistic_prediction == "negative":

                st.error(
                    f"😞 {logistic_prediction.upper()}"
                )

            else:

                st.warning(
                    f"😐 {logistic_prediction.upper()}"
                )


            # Probability
            st.markdown(
                "**Prediction probabilities:**"
            )

            for sentiment, probability in zip(
                logistic_classes,
                logistic_probability
            ):

                st.write(
                    f"{sentiment.capitalize()}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


        # =============================================
        # SVM
        # =============================================

        with col2:

            st.markdown(
                "### Linear SVM"
            )

            if svm_prediction == "positive":

                st.success(
                    f"😊 {svm_prediction.upper()}"
                )

            elif svm_prediction == "negative":

                st.error(
                    f"😞 {svm_prediction.upper()}"
                )

            else:

                st.warning(
                    f"😐 {svm_prediction.upper()}"
                )


        # =============================================
        # MODEL AGREEMENT
        # =============================================

        st.subheader("Model Comparison")

        if logistic_prediction == svm_prediction:

            st.success(
                f"Both models agree: "
                f"**{logistic_prediction.upper()}**"
            )

        else:

            st.warning(
                "The models disagree on the sentiment."
            )

            st.write(
                f"Logistic Regression: "
                f"**{logistic_prediction}**"
            )

            st.write(
                f"Linear SVM: "
                f"**{svm_prediction}**"
            )


        # =============================================
        # CLEANED TEXT
        # =============================================

        with st.expander(
            "View Preprocessed Text"
        ):

            st.write(cleaned_review)