# Hate Speech Detection on Twitter

This project implements a **Hate Speech Detection model** for Twitter posts using **Natural Language Processing (NLP)** techniques and **Machine Learning**. It classifies tweets into three categories:  

1. **Hate Speech**  
2. **Offensive Language**  
3. **No Hate and Offensive Language**

---

## Dataset

- The dataset is a Twitter dataset (`twitter.csv`) containing the following columns:
  - `tweet`: The text of the tweet
  - `class`: Numerical label for classification
    - `0` → Hate Speech  
    - `1` → Offensive Language  
    - `2` → No Hate and Offensive  
- Total samples:
  - Hate Speech: 1,430  
  - Offensive Language: 19,190  
  - No Hate and Offensive: 4,163  

> Only the `tweet` and `class` columns are used for training.

---

## Preprocessing

- Convert all text to lowercase.
- Remove:
  - Retweet tags (`rt`)  
  - Mentions (`@username`)  
  - URLs (`http://...`, `https://...`)  
  - Punctuation and digits  
- Tokenization, stopwords removal, and lemmatization using **NLTK**.
- Example of cleaned text:


---

## Model

- **Vectorization:** TF-IDF (`TfidfVectorizer`) with max features 20,000 and n-grams (1,2)  
- **Classifier:** Logistic Regression (`class_weight='balanced'`, solver=`saga`)  
- **Pipeline:** Combined preprocessing + TF-IDF + classifier using `sklearn.Pipeline`.

---

## Training

- Train-test split: 67% train, 33% test (stratified)  
- Label encoding applied to convert text labels to numeric format  
- Training size: 16,604 samples  
- Test size: 8,179 samples  

**Performance:**

| Class                  | Precision | Recall | F1-score |
|------------------------|-----------|--------|----------|
| Hate Speech            | 0.25      | 0.65   | 0.36     |
| No Hate and Offensive  | 0.74      | 0.94   | 0.83     |
| Offensive Language     | 0.98      | 0.80   | 0.88     |
| **Accuracy**           |           |        | 0.82     |

> Confusion matrix indicates some confusion between Hate Speech and Offensive Language due to class imbalance.

---

## Saving the Model

- The trained pipeline and label encoder are saved using **joblib**:

```python
joblib.dump(pipeline, "hate_detector_pipeline.joblib")
joblib.dump(le, "label_encoder.joblib")
