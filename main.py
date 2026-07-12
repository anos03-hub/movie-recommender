# -*- coding: utf-8 -*-
"""
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from data import load_and_filter, split_and_pivot          # noqa: E402
from classic_model import compute_similarity, predict_all   # noqa: E402
from evaluate import evaluate                                # noqa: E402

# Απόλυτη διαδρομή στο dataset (δουλεύει ανεξάρτητα από το πού τρέχεις)
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ratings.csv')

N = 20  # αριθμός γειτόνων


def main():
    print("Φόρτωση & φιλτράρισμα δεδομένων...")
    ratings = load_and_filter(DATA_PATH, min_movie_ratings=10, min_user_ratings=5)
    print(f"  Βαθμολογίες μετά το φιλτράρισμα: {len(ratings)}")

    train_matrix, test_matrix = split_and_pivot(ratings)
    print(f"  Train: {train_matrix.shape} | Test: {test_matrix.shape}\n")

    for method in ['cosine', 'pearson']:
        print(f"=== {method.upper()} ===")
        similarity = compute_similarity(train_matrix, method=method)
        predictions = predict_all(test_matrix, similarity, train_matrix, N=N)
        results = evaluate(test_matrix, predictions, train_matrix)
        for k, v in results.items():
            print(f"  {k:10s}: {v:.4f}")
        print()


if __name__ == '__main__':
    main()
