## 🌐 News Classifier AI

An end-to-end **Natural Language Processing (NLP)** system that classifies news articles into
**World, Sports, Business, and Sci/Tech** categories using both traditional machine learning
and deep learning models, wrapped in a production-ready inference layer and interactive UI.

Soon, we will extend it further with more categories and advanced AI models.

---

##  Project Overview

This project demonstrates a complete machine learning lifecycle for text classification, covering:

- Exploratory data analysis and text preprocessing
- Feature engineering using TF-IDF
- Training and evaluation of classical ML models
- Deep learning with neural networks
- Model comparison and selection
- Production-grade inference logic
- Interactive Streamlit web application

The focus of this project is not just model accuracy, but **clean system design, reproducibility,
and deployability**, similar to real-world ML engineering workflows.

---

##  System Architecture

Raw News Data
↓
Text Cleaning & Preprocessing
↓
Feature Extraction
├── TF-IDF → ML Models
└── Tokenization → Neural Network
↓
Trained Models (Saved Artifacts)
↓
Inference Layer
↓
Streamlit Web Application

---

## 🧠 Models Used & Design Decisions

### Traditional Machine Learning Models
- **TF-IDF + Logistic Regression**
- **TF-IDF + Support Vector Machine (SVM)**

These models serve as strong baselines for text classification tasks.  
They are fast, interpretable, and perform exceptionally well on structured news data.

### Deep Learning Model
- **Neural Network with word embeddings**

The neural network was trained to capture non-linear patterns and contextual information
beyond bag-of-words representations.
Deep Learning Model

Soon, we will improve it further and extend it with more advanced architectures.

### Model Selection Rationale
Although ensemble approaches were explored, a single SVM-based model was preferred for
deployment due to its strong performance, lower inference latency, and reduced system
complexity. This aligns with real-world engineering trade-offs where simplicity and
maintainability are critical.

---

## 📊 Model Evaluation

Models were evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-score

| Model | Accuracy |
|------|----------|
| Logistic Regression | 90.39% |
| SVM | 90.71% |
| Neural Network | 91.01% |

> Detailed experiments and evaluations can be found in the Jupyter notebooks.

---

##  Application Preview

### Overview
<img width="1366" height="725" alt="overview" src="https://github.com/user-attachments/assets/df3ecc17-0a4a-49cf-ab7e-1464590c5ed5" />

### Prediction Output 
#### Business Atrticle
<img width="1366" height="701" alt="business news" src="https://github.com/user-attachments/assets/5d3afdab-e68e-4b0d-8f86-245289e04601" />

#### Sports Article
<img width="1366" height="701" alt="sports news" src="https://github.com/user-attachments/assets/7e88a2a2-8d50-4813-824c-cf1b2c1b80b0" />

#### World Article
<img width="1366" height="701" alt="World news" src="https://github.com/user-attachments/assets/792f4c9e-0f82-49cb-8823-62cee7051f1e" />

#### Sci/Tech Article
<img width="1366" height="701" alt="tech news" src="https://github.com/user-attachments/assets/98cccce1-d3ba-46a7-8bf4-6eed555c1ae3" />


## ⚙️ Run Locally

Clone the repository and run the application locally:

```bash
git clone https://github.com/AzharMehmood4/news-classifier
cd news-classifier

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run src/app.py
````

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning:** Scikit-learn
* **Deep Learning:** TensorFlow / Keras
* **NLP:** NLTK
* **Web App:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 👤 Authors

- **Azhar Mehmood** (8397)  
- **Alyan Asghar** (8651)  
- **Muhammad Moez Khan** (8774)
- **Salik Bashir** (5620)

---
