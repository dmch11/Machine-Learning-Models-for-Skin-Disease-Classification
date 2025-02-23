# Machine-Learning-Models-for-Skin-Disease-Classification
-
# Skin Disease Text Classification with Naive Bayes, XGBoost, and BERT

## **Overview**
This project focuses on text classification for skin disease descriptions using machine learning and deep learning models. The dataset consists of textual descriptions of skin conditions, which are preprocessed and vectorized before being classified using three different models:
- **Naive Bayes**
- **XGBoost**
- **BERT (Bidirectional Encoder Representations from Transformers)**

## **Dataset**
- The dataset is named **"Skin_text_classifier.csv"**.
- It contains textual descriptions of skin diseases, which are labeled for classification.
- Data preprocessing includes text cleaning, tokenization, stopword removal, and feature extraction using **TF-IDF**.

## **Methodology**
### **1. Data Preprocessing**
- Tokenization and stopword removal using `NLTK`.
- Text vectorization using **TF-IDF**.
- Handling class imbalance using **SMOTE** and **Random Oversampling**.

### **2. Models Implemented**
#### **Naive Bayes**
- Baseline model using **Multinomial Naive Bayes**.
- Hyperparameter tuning on TF-IDF vectorizer.

#### **XGBoost**
- **Randomized Search CV** for hyperparameter tuning.
- Applied **SMOTE** to handle imbalanced data.
- Evaluated using **Macro Average Comparison**.

#### **BERT**
- Implemented using the **Hugging Face Transformers library**.
- Fine-tuned on labeled text data using **PyTorch**.
- Used `Trainer` API for model training.

## **Dependencies**
To run the notebook, install the required libraries:
```bash
pip install numpy pandas seaborn scikit-learn xgboost imbalanced-learn transformers torch
