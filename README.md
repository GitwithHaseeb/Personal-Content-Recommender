# Personal Content Recommender

A smart hybrid recommendation system for **Movies, Series, Books, Videos & Songs**.

It learns from your personal ratings (`my_ratings.csv`) and recommends content using three powerful approaches:

- **Content-Based Filtering** (TF-IDF + Cosine Similarity)
- **Collaborative Filtering** (Biased Matrix Factorization)
- **Semantic Embeddings** (Sentence-Transformers)

Built with **Python**, **Streamlit**, **scikit-learn**, and **Sentence-Transformers**.

---

## Features

- Personalized recommendations based on your taste
- Beautiful Streamlit web interface
- Offline evaluation metrics (Precision@5, Recall@5)
- Jupyter Notebook for Google Colab
- Easy to customize with your own data

---

## Tech Stack

- Python 3.10+
- Streamlit
- scikit-learn
- Sentence-Transformers
- Pandas & NumPy

---

## Quick Start

```bash
git clone https://github.com/GitwithHaseeb/Personal-Content-Recommender.git
cd Personal-Content-Recommender
pip install -r requirements.txt
streamlit run app.py
