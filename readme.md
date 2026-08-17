# Customer Sentiment Analysis

## 📌 Project Overview
This project provides a complete machine learning pipeline for analyzing customer reviews and classifying their sentiment[cite: 7, 8]. It leverages Natural Language Processing (NLP) techniques to categorize text into three distinct classes: **Negative, Neutral, and Positive**[cite: 3, 7]. The project includes scripts for data exploration, text preprocessing, model training, and provides both a Command-Line Interface (CLI) and an interactive Streamlit web application for real-time inference[cite: 6, 7, 8].

## ✨ Features
*   **Comprehensive Data Preprocessing:** Cleans raw review data by converting it to lowercase, stripping URLs, removing special characters (keeping only alphanumerics and apostrophes), and eliminating extra and leading/trailing spaces[cite: 6].
*   **Exploratory Data Analysis (EDA):** Analyzes the `jbeno/sentiment_merged` dataset to check for missing values, identify duplicates, and visualize sentiment distributions and review length frequencies[cite: 1, 2].
*   **Advanced Text Vectorization:** Utilizes `TfidfVectorizer` with bigrams (`ngram_range=(1, 2)`), sublinear TF scaling, and limits the vocabulary to the top 20,000 features[cite: 3, 4, 5].
*   **Dual-Model Architecture:** Trains and evaluates both a Logistic Regression model (`max_iter=1000`) and a Linear SVM model (`class_weight="balanced"`)[cite: 3, 4].
*   **Interactive Web App:** Includes a Streamlit dashboard that allows users to input reviews, view prediction probabilities, and compare results between the Logistic Regression and SVM models[cite: 7].
*   **Command-Line Inference:** Offers a simple CLI tool for quickly testing reviews against both trained models directly from the terminal[cite: 8].

## 📊 Dataset
The project utilizes the `jbeno/sentiment_merged` dataset loaded via the Hugging Face `datasets` library[cite: 1, 2, 6]. 
*   The raw data is split into training, validation, and testing sets[cite: 1].
*   Duplicate sentences are identified and removed across all splits to ensure data integrity[cite: 2, 6].
*   The target variable is the `label` column, which contains the sentiment categories[cite: 1].

## 🚀 Pipeline Execution

### 1. Data Cleaning and Preparation
The text cleaning script prepares the data for modeling[cite: 6]. 
*   It removes empty text entries and duplicate sentences[cite: 6].
*   The cleaned splits are saved locally as `train_clean.csv`, `validation_clean.csv`, and `test_clean.csv` inside a `data/` directory[cite: 6].

### 2. Exploratory Data Analysis
The EDA scripts provide insights into the dataset's characteristics[cite: 2].
*   They generate bar charts for sentiment distribution and histograms for character length distribution using Matplotlib[cite: 2].
*   They calculate the top 20 most frequent words overall, as well as the top 15 most frequent words specific to each sentiment class[cite: 2].

### 3. Model Training
Two separate scripts handle the training and evaluation of the models[cite: 3, 4].
*   **Logistic Regression:** Trained on the TF-IDF features and evaluated using Accuracy, Macro F1-score, and a Confusion Matrix[cite: 3].
*   **Linear SVM:** Configured with a balanced class weight to handle potential imbalances, evaluated using the same metrics, and visualized with a Confusion Matrix[cite: 4].
*   **Artifact Generation:** The trained models and their respective TF-IDF vectorizers are exported using `joblib` into a `models/` directory (e.g., `logistic_regression.pkl`, `svm.pkl`, `tfidf_logistic.pkl`, `tfidf_svm.pkl`)[cite: 3, 4].

## 💻 Usage & Interfaces

### Streamlit Web Application
To run the interactive UI, ensure the models are generated and execute the Streamlit script[cite: 7]. 
*   The app features a sidebar with model and dataset information[cite: 7].
*   Users can type a custom review into a text area to analyze the sentiment[cite: 7].
*   The app displays the cleaned text, the prediction from both models, and a detailed probability breakdown for the Logistic Regression model[cite: 7].
*   It alerts the user if the two models disagree on the sentiment[cite: 7].

### Command-Line Interface (CLI)
For quick testing without a GUI, the CLI script can be executed[cite: 8].
*   The user is prompted with `Enter customer review: `[cite: 8].
*   The script cleans the text in the background and prints the resulting predictions from both the Logistic Regression and Linear SVM models directly to the console[cite: 8].

## 🛠 Dependencies
To run this project, the following primary Python libraries are required based on the source code:
*   `pandas`[cite: 1, 2, 3]
*   `datasets` (Hugging Face)[cite: 1, 2, 6]
*   `scikit-learn`[cite: 3, 4, 5]
*   `matplotlib`[cite: 2, 3, 4]
*   `joblib`[cite: 3, 4, 7]
*   `streamlit`[cite: 7]
*   `re` (Standard Python Library)[cite: 2, 6, 7]