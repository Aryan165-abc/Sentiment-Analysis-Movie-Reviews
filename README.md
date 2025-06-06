# 🎬 Sentiment Analysis on IMDB Movie Reviews

## 🧠 Project Overview
This project focuses on analyzing sentiment in movie reviews using **Natural Language Processing (NLP)** techniques. The goal is twofold:
- Apply **K-Means Clustering** to group reviews based on similarities.
- Build a **Sentiment Classification Model** to predict whether a review is positive or negative.

---

## 🎯 Objectives
- Preprocess raw text data using NLP.
- Extract meaningful features using **TF-IDF vectorization**.
- Perform **unsupervised learning** using K-Means Clustering.
- Build a **Logistic Regression classifier** for sentiment analysis.
- Visualize clustering results and evaluate model performance.

---

## 📁 Dataset
- **Source**: IMDB Dataset (50,000 reviews)
- **Columns**: `review`, `sentiment`
- Reviews are labeled as either **positive** or **negative**.

---

## 🛠️ Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🧹 Key Steps

### 1. Text Preprocessing
- Lowercasing
- Removing punctuation, numbers, HTML tags
- Removing stopwords (using NLTK)

### 2. Feature Engineering
- TF-IDF Vectorizer (Top 1000 features)

### 3. K-Means Clustering
- Unsupervised clustering of review vectors
- PCA used for 2D visualization

### 4. Sentiment Classification
- Logistic Regression model trained on labeled data
- Accuracy and classification report generated

---

## 📊 Results

- **Clustering**:
  - Reviews were grouped into 2 clusters based on content similarity.
- **Classification Accuracy**:
  - Achieved high accuracy in distinguishing positive vs negative reviews.
  - Model evaluated using accuracy, precision, recall, and F1 score.

---

## 📎 Files in the Repository
- `sentiment_analysis.ipynb`: Jupyter notebook with full code and output
- `IMDB Dataset.csv`: Dataset used for training and clustering
- `README.md`: Project documentation


> 📁 **Note**: Due to file size limitations, the dataset is not uploaded here.  
> You can download the dataset directly from [Kaggle - IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).



