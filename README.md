# 🎬 Movie Recommender System

Σύστημα συστάσεων ταινιών σε δεδομένα MovieLens, με **item-based collaborative filtering**.
Πρόβλεψη της βαθμολογίας που θα έδινε ένας χρήστης σε μια ταινία που δεν έχει δει, βρίσκοντας
όμοιες ταινίες και παίρνοντας σταθμισμένο μέσο όρο.

> 🚧 Σε εξέλιξη: αναβάθμιση από κλασικό collaborative filtering σε deep learning (matrix factorization → NCF)
> και deployment ως Streamlit app.

## 📊 Δεδομένα
- **MovieLens** — ~100.000 βαθμολογίες, ~600 χρήστες, ~9.000 ταινίες.
- Φιλτράρισμα: ταινίες με ≥10 βαθμολογίες, χρήστες με ≥5 βαθμολογίες.

## 🧠 Προσέγγιση
1. **Δεδομένα** (`src/data.py`): φόρτωση, φιλτράρισμα, train/test split (80/20), user-movie matrix.
2. **Μοντέλο** (`src/classic_model.py`): ομοιότητα ταινιών με **cosine** & **Pearson**·
   πρόβλεψη = σταθμισμένος μέσος όρος των N πιο όμοιων ταινιών που έχει βαθμολογήσει ο χρήστης.
3. **Αξιολόγηση** (`src/evaluate.py`): binary relevance (βαθμολογία ≥ μέσος όρος χρήστη) →
   confusion matrix → **precision, recall, F1**, καθώς και **MAE** στην πρόβλεψη βαθμολογίας.

## 📈 Αποτελέσματα (baseline)

| Μέθοδος | Precision | Recall | F1 | MAE |
|---------|-----------|--------|----|-----|
| Cosine  | 0.6643   |0.7304|0.6958|0.6506|
| Pearson | 0.6907   |0.6620|0.6760|0.6481|

Παρατηρήσεις:

-Φαίνεται ότι το Cosine έχει καλύτερο recall, δηλαδή πιάνει περισσότερες από τις καλές ταίνιες.
-Το Pearson έχει καλύτερο precision (όταν προτείνει κάτι, είναι πιο συχνά σωστό).
-Το F1 (που τα ισορροπεί) είναι σχεδόν ίδιο, με ελαφρύ προβάδισμα του cosine.

## ▶️ Πώς τρέχει
```bash
pip install -r requirements.txt
python src/main.py        # ή όποιο είναι το entry point σου
```

## 🗺️ Roadmap
- [x] Κλασικό collaborative filtering (cosine / Pearson)
- [ ] Matrix factorization με embeddings (PyTorch)
- [ ] Neural Collaborative Filtering (NCF)
- [ ] Σύγκριση classic vs deep learning
- [ ] Streamlit app + deployment

## 🛠️ Τεχνολογίες
Python · pandas · NumPy · scikit-learn *(σύντομα: PyTorch, Streamlit)*
