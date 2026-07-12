import torch
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from neural_data import prepare_ids
from neural_model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ratings.csv')
MOVIES_PATH = os.path.join(BASE_DIR, 'data', 'movies.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'movie_model.pth')


ratings = pd.read_csv(DATA_PATH)
ratings, num_users, num_movies = prepare_ids(ratings)

# τίτλοι ταινιών
movies_titles = pd.read_csv(MOVIES_PATH)

# μοντέλο: άδειο καλούπι + γέμισμα με weights
model = Model(num_users, num_movies, embedding_dim=16)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

print("Model loaded!")
# movieId → movie_idx (και το αντίστροφο)
movieid_to_idx = dict(zip(ratings['movieId'], ratings['movie_idx']))
idx_to_movieid = dict(zip(ratings['movie_idx'], ratings['movieId']))

print(list(movieid_to_idx.items())[:5])   # δες μερικά ζευγάρια    

def recommend(selected_titles, top_n =5):
    selected_idx = []
    for title in selected_titles:
         movie_id = movies_titles[movies_titles['title'] == title]['movieId'].values[0]
         idx = movieid_to_idx[movie_id]
         selected_idx.append(idx)
    with torch.no_grad():
        idx_tensors = torch.tensor(selected_idx)
        vectors = model.em_movies(idx_tensors)
        profile = vectors.mean(dim=0)
        all_movies = model.em_movies.weight
        similarities = torch.cosine_similarity(profile.unsqueeze(0), all_movies)
        top_indices = torch.argsort(similarities, descending=True)
    
    recommendations = []
    for idx in top_indices:
        idx = idx.item()
        if idx not in selected_idx:
            movie_id = idx_to_movieid[idx]
            titles = movies_titles[movies_titles['movieId'] == movie_id]['title'].values[0]
            recommendations.append(titles)
        if len(recommendations) == top_n:
            break
    return recommendations

print(recommend(["Toy Story (1995)", "Jumanji (1995)"]))


    
        


print(recommend(["Toy Story (1995)", "Jumanji (1995)"]))