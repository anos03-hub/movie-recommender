# -*- coding: utf-8 -*-
"""
data.py — Φόρτωση, φιλτράρισμα και προετοιμασία των δεδομένων.

Ροή: CSV -> φιλτράρισμα σπάνιων ταινιών/χρηστών -> train/test split -> user-movie matrices.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_filter(path, min_movie_ratings=10, min_user_ratings=5):
    """Φορτώνει το ratings.csv και κρατά μόνο ταινίες/χρήστες με αρκετές βαθμολογίες.

    Args:
        path: διαδρομή προς το ratings.csv
        min_movie_ratings: ελάχιστες βαθμολογίες ανά ταινία (M)
        min_user_ratings: ελάχιστες βαθμολογίες ανά χρήστη (M')
    Returns:
        DataFrame με στήλες userId, movieId, rating (φιλτραρισμένο)
    """
    ratings = pd.read_csv(path)

    # Κράτα ταινίες με τουλάχιστον M βαθμολογίες
    movie_counts = ratings['movieId'].value_counts()
    keep_movies = movie_counts[movie_counts >= min_movie_ratings].index
    ratings = ratings[ratings['movieId'].isin(keep_movies)]

    # Κράτα χρήστες με τουλάχιστον M' βαθμολογίες
    user_counts = ratings['userId'].value_counts()
    keep_users = user_counts[user_counts >= min_user_ratings].index
    ratings = ratings[ratings['userId'].isin(keep_users)]

    return ratings


def split_and_pivot(ratings, test_size=0.2, random_state=42):
    """Χωρίζει σε train/test και φτιάχνει τους πίνακες χρήστη-ταινίας.

    Returns:
        (train_matrix, test_matrix) — γραμμές=userId, στήλες=movieId, τιμές=rating (0 = χωρίς βαθμολογία)
    """
    train_data, test_data = train_test_split(
        ratings, test_size=test_size, random_state=random_state
    )
    train_matrix = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    test_matrix = test_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    return train_matrix, test_matrix
