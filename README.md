# 🎬 Movie Recommender System

A movie recommendation system built on the **MovieLens** dataset, implemented in two stages: a classic **item-based collaborative filtering** baseline and a **deep learning** model (matrix factorization with embeddings, in PyTorch). The trained model powers an interactive **Streamlit web app** that recommends movies to new users based on titles they already like.

---

## ✨ Overview

The project predicts how a user would rate a movie they haven't seen, by learning the relationships between users and movies from historical ratings. It evolves from a hand-crafted similarity approach into a learned neural model, and compares the two.

The final web app solves the **cold-start problem**: a brand-new visitor simply picks a few movies they enjoyed, and the system recommends similar ones using the learned movie embeddings — no prior user history required.

## 🖥️ Demo

The Streamlit app lets users select movies they liked and get personalized recommendations:

> _Add a screenshot of the app here, e.g._ `![app screenshot](docs/app.png)`

Run it locally with:

```bash
streamlit run app.py
```

## 📊 Dataset

[MovieLens Latest Small](https://grouplens.org/datasets/movielens/) — recommended for education and development:

- ~100,000 ratings
- 610 users
- 9,724 movies

Ratings are filtered to keep movies with ≥10 ratings and users with ≥5 ratings, reducing sparsity.

## 🧠 Approach

### 1. Classic baseline — item-based collaborative filtering
For each target movie, the system finds the most **similar movies** the user has already rated and predicts the rating as a weighted average of those ratings. Two similarity measures are compared:

- **Cosine similarity**
- **Pearson correlation**

### 2. Deep learning — matrix factorization (PyTorch)
Each user and movie is represented by a learned **embedding vector**. A rating is predicted as the dot product of the user and movie embeddings, plus **bias terms** (per-user, per-movie, and a global bias):

```
prediction = (user_vec · movie_vec) + user_bias + movie_bias + global_bias
```

The model is trained with **MSE loss** and the **Adam** optimizer. **Weight decay** (L2 regularization) is used to control overfitting, and hyperparameters (`embedding_dim`, `weight_decay`, `epochs`) were tuned experimentally.

## 📈 Results

Evaluation on a held-out test set (80/20 split):

| Method | Precision | Recall | MAE |
|--------|-----------|--------|-----|
| Collaborative filtering (Cosine) | 0.6643 | 0.7304 | 0.6506 |
| Collaborative filtering (Pearson) | 0.6907 | 0.6620 | 0.6481 |
| Matrix Factorization (Deep Learning) | — | — | **0.81** |

Best deep learning configuration: `embedding_dim=16`, `weight_decay=1e-3`, `400 epochs`.

## 🔑 Key findings

- On this **small dataset**, the classic collaborative filtering baseline (MAE ≈ 0.65) **outperformed** the deep learning model (MAE ≈ 0.81). This is expected: classic collaborative filtering is very strong on small, dense datasets, while neural models tend to shine on much larger ones where they can learn richer patterns.
- Tuning `weight_decay` produced the classic **regularization U-curve**: too little → overfitting, too much → underfitting, with a sweet spot at `1e-3`.
- Adding **bias terms** to the matrix factorization model was the single most impactful improvement, as it lets the embeddings focus on genuine preferences instead of modeling per-user/per-movie rating tendencies.

## 🗂️ Project structure

```
movie-recommender/
├── data/
│   ├── ratings.csv
│   └── movies.csv
├── src/
│   ├── data.py            # load, filter, train/test split
│   ├── classic_model.py   # cosine / pearson similarity + prediction
│   ├── evaluate.py        # precision, recall, F1, MAE
│   ├── neural_data.py     # ID mapping + tensor conversion
│   └── neural_model.py    # matrix factorization model (embeddings + bias)
├── main.py                # runs the classic baseline
├── train.py               # trains the neural model, saves movie_model.pth
├── recommend.py           # recommendation logic (similarity-based)
├── app.py                 # Streamlit web app
├── requirements.txt
└── README.md
```

## ▶️ How to run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the classic baseline**
```bash
python main.py
```

**3. Train the deep learning model** (saves `movie_model.pth`)
```bash
python train.py
```

**4. Launch the web app**
```bash
streamlit run app.py
```

## ⚠️ Limitations & future work

- **Collaborative filtering, not content-based:** the model learns from *rating patterns*, so "similar" means "rated similarly by similar people" — not thematically similar. Recommendations can therefore look surprising.
- **Popularity bias:** in a small dataset, very popular movies tend to be recommended regardless of input.
- **Future improvements:**
  - Add a **content-based / hybrid** component using movie `genres` for more thematically relevant recommendations.
  - Scale to a **larger MovieLens dataset** (e.g. ml-1m / ml-25m) with mini-batching and GPU training.
  - Deploy the app publicly (e.g. Streamlit Community Cloud) for a live demo link.

## 🛠️ Tech stack

Python · PyTorch · pandas · NumPy · scikit-learn · Streamlit