# -*- coding: utf-8 -*-
"""
classic_model.py — Κλασικό item-based collaborative filtering.

Ιδέα: όμοιες ταινίες παίρνουν όμοιες βαθμολογίες από τον ίδιο χρήστη.
Πρόβλεψη = σταθμισμένος μέσος όρος των βαθμολογιών του χρήστη στις N πιο όμοιες ταινίες.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(train_matrix, method='cosine'):
    """Υπολογίζει ομοιότητα ταινία-προς-ταινία (item-item).

    Προσοχή: χρησιμοποιούμε τον ανάστροφο (.T) ώστε η ομοιότητα να είναι μεταξύ ταινιών, όχι χρηστών.
    """
    if method == 'cosine':
        sim = cosine_similarity(train_matrix.T)
        return pd.DataFrame(sim, index=train_matrix.columns, columns=train_matrix.columns)
    elif method == 'pearson':
        # οι στήλες του train_matrix είναι οι ταινίες -> corr() δίνει ομοιότητα ταινία-ταινία
        return train_matrix.corr(method='pearson')
    else:
        raise ValueError("method πρέπει να είναι 'cosine' ή 'pearson'")


def predict_rating(user_id, movie_id, similarity_matrix, train_matrix, N=20):
    """Προβλέπει τη βαθμολογία ενός χρήστη για μια ταινία.

    Βήματα: πάρε τις ομοιότητες της ταινίας -> κράτα μόνο ταινίες που έχει βαθμολογήσει ο χρήστης
    -> διάλεξε τις N πιο όμοιες -> σταθμισμένος μέσος όρος (βάρος = ομοιότητα).
    """
    # Αν η ταινία ή ο χρήστης δεν υπάρχουν στο training, δεν μπορούμε να προβλέψουμε
    if movie_id not in similarity_matrix.columns or user_id not in train_matrix.index:
        return 0.0

    sim_scores = similarity_matrix[movie_id]
    user_ratings = train_matrix.loc[user_id]

    # Ταινίες που ο χρήστης έχει βαθμολογήσει
    rated_movies = user_ratings[user_ratings > 0].index
    sim_scores = sim_scores[sim_scores.index.intersection(rated_movies)]

    if sim_scores.empty:
        return 0.0

    # N πιο όμοιες
    top = sim_scores.sort_values(ascending=False).iloc[:N]
    top_ratings = user_ratings[top.index]

    denominator = top.sum()
    if denominator == 0:
        return 0.0
    return float(np.dot(top.values, top_ratings.values) / denominator)


def predict_all(test_matrix, similarity_matrix, train_matrix, N=20):
    """Προβλέπει βαθμολογίες ΜΟΝΟ για τα ζεύγη (χρήστης, ταινία) που υπάρχουν στο test set.

    (Πιο γρήγορο από το να προβλέπουμε όλο τον πίνακα — χρειαζόμαστε μόνο ό,τι αξιολογείται.)
    """
    predictions = pd.DataFrame(0.0, index=test_matrix.index, columns=test_matrix.columns)

    stacked = test_matrix.stack()
    to_predict = stacked[stacked > 0].index  # ζεύγη (userId, movieId) που ο χρήστης όντως βαθμολόγησε

    for user_id, movie_id in to_predict:
        predictions.loc[user_id, movie_id] = predict_rating(
            user_id, movie_id, similarity_matrix, train_matrix, N
        )
    return predictions
