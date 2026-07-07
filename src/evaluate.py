# -*- coding: utf-8 -*-
"""
evaluate.py — Αξιολόγηση των προβλέψεων.

Δύο ειδών μετρικές:
  - MAE: πόσο μακριά είναι η προβλεπόμενη βαθμολογία από την πραγματική.
  - precision / recall / F1: με βάση το αν μια ταινία είναι "σχετική" (βαθμολογία >= μέσος όρος του χρήστη).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error


def evaluate(test_matrix, predicted_matrix, train_matrix):
    """Επιστρέφει λεξικό με precision, recall, f1, accuracy, mae."""
    # Κράτα μόνο τα ζεύγη που ο χρήστης όντως βαθμολόγησε στο test
    actual = test_matrix.stack()
    actual = actual[actual > 0]

    # Ευθυγράμμισε τις προβλέψεις με τα ίδια ζεύγη
    predicted = predicted_matrix.stack().reindex(actual.index)
    mask = predicted.notna()
    actual, predicted = actual[mask], predicted[mask]

    # --- MAE ---
    mae = mean_absolute_error(actual.values, predicted.values)

    # --- Binary relevance: "σχετική" αν βαθμολογία >= μέσος όρος του χρήστη (από το train) ---
    user_means = train_matrix[train_matrix > 0].mean(axis=1)
    means = pd.Series(actual.index.map(lambda idx: user_means.get(idx[0], np.nan)),
                      index=actual.index)
    valid = means.notna()

    actual_bin = (actual[valid] >= means[valid]).astype(int)
    predicted_bin = (predicted[valid] >= means[valid]).astype(int)

    # labels=[0,1] ώστε ο πίνακας να είναι πάντα 2x2 (ακόμα κι αν λείπει μια κλάση)
    tn, fp, fn, tp = confusion_matrix(actual_bin, predicted_bin, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy, 'mae': mae}
