## Project Overview:

This project is a Customer Sentiment Analysis application developed using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It classifies customer reviews into three sentiment categories: **Positive, Negative, and Neutral**.

The project uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert textual customer reviews into numerical feature vectors. Two machine learning models, **Logistic Regression** and **Linear Support Vector Machine (SVM)**, are trained and evaluated for sentiment classification.

The workflow consists of:

- Loading a sentiment-labelled customer review dataset from the Hugging Face Datasets Hub.
- Performing data cleaning and exploratory data analysis.
- Removing duplicate sentences and checking for missing values.
- Preprocessing the customer review text.
- Converting text into numerical features using TF-IDF.
- Training Logistic Regression and Linear SVM classification models.
- Evaluating the models using Accuracy, Precision, Recall, F1-score, Macro F1-score, Classification Report, and Confusion Matrix.
- Saving the trained models and TF-IDF vectorizer for inference.
- Creating an interactive web application using Streamlit.
- Providing real-time sentiment predictions for new customer reviews.

The application allows users to enter a customer review and obtain sentiment predictions from both trained machine learning models.

---

## Project Structure:

- `app.py` – Streamlit application for real-time customer sentiment prediction.
- `data/ax.py` – Dataset loading, data analysis, preprocessing, and EDA.
- `models/logistic_regression.py` – Training and evaluation of the Logistic Regression model.
- `models/svm.py` – Training and evaluation of the Linear SVM model.
- `model_files/` – Contains saved machine learning models and the TF-IDF vectorizer.
- `requirements.txt` – Python dependencies required to run the project.
- `README.md` – Project documentation.
- `.gitignore` – Files and folders excluded from GitHub.

---

## Dataset:

- **Name:** `sentiment_merged`
- **Source:** Hugging Face Datasets Hub
- **Dataset Identifier:** `jbeno/sentiment_merged`
- **Dataset Link:** https://huggingface.co/datasets/jbeno/sentiment_merged
- **Task:** Multi-class sentiment classification
- **Classes:** Positive, Negative, Neutral

The dataset contains the following main fields:

| Column | Description |
|---|---|
| `sentence` | Customer review/text |
| `label` | Sentiment label |
| `source` | Original source of the text |
| `split` | Dataset split |

### Dataset Size:

- **Training:** 102,097 samples
- **Validation:** 5,421 samples
- **Test:** 6,530 samples

### Training Sentiment Distribution:

- **Neutral:** 49,148
- **Positive:** 31,039
- **Negative:** 21,910

---

## Methodology:

The project follows the following machine learning pipeline:

<img width="1122" height="1402" alt="ChatGPT Image Aug 16, 2026, 06_54_08 PM" src="https://github.com/user-attachments/assets/491c3b81-92c7-4ef0-9991-652c0e1e72ae" />

## Output: 
<img width="1816" height="960" alt="image" src="https://github.com/user-attachments/assets/bc1614d2-9f06-418e-9239-16cfa0c778a6" />
<img width="1822" height="962" alt="image" src="https://github.com/user-attachments/assets/a3fbd862-a869-4f8a-a032-64f0774012d4" />

## 🚀 Future Development

The project can be further enhanced from a basic sentiment classifier into an advanced customer-feedback analysis system.

### 📅 Short-Term (1–2 Years)

- Improve TF-IDF and model hyperparameters.
- Perform detailed error analysis.
- Experiment with **BERT, DistilBERT, and RoBERTa**.
- Improve the Streamlit UI with better visualizations.
- Add model versioning and automated testing.

### 📅 Medium-Term (3–5 Years)

- Implement **Aspect-Based Sentiment Analysis** for aspects such as price, quality, and delivery.
- Add multilingual sentiment analysis.
- Develop REST API and database integration.
- Implement model monitoring and automated retraining.
- Build an interactive customer-feedback dashboard.

### 📅 Long-Term (5+ Years)

- Integrate Generative AI for automatic review summarization.
- Detect customer complaints, emotions, and emerging issues.
- Integrate with CRM and customer-support systems.
- Develop a scalable customer-intelligence platform.

### 🎯 Key Success Factors

- High-quality and diverse training data.
- Reliable sentiment labels.
- Continuous model evaluation.
- Adequate computing resources.
- Data privacy and security.
- Regular monitoring for model drift.

### ⚠️ Potential Challenges

| Challenge | Mitigation |
|---|---|
| Class imbalance | Macro F1 and class weighting |
| Sarcasm and context | Transformer models |
| Model drift | Continuous monitoring |
| Poor-quality data | Data validation |
| High computation cost | Lightweight models |

### 📊 Progress Indicators

Future improvements will be measured using:

- Accuracy
- Macro F1-score
- Precision and Recall
- Confusion Matrix
- Inference Time
- Model Reliability

The current **Logistic Regression baseline (≈65% accuracy, 0.6502 Macro F1)** will be used as the benchmark for evaluating future models.

### 🛣️ Development Roadmap

<img width="1024" height="1536" alt="ChatGPT Image Aug 17, 2026, 08_50_13 AM" src="https://github.com/user-attachments/assets/2caf2ab9-6cd6-485b-9143-fa0deedc2c9d" />


