import torch
import torch.nn as nn
import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from neural_data import prepare_ids, to_tensors
from neural_model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ratings.csv')

ratings = pd.read_csv(DATA_PATH)
ratings, num_users, num_movies = prepare_ids(ratings)

users, movies, ratings_t = to_tensors(ratings)
model = Model(num_users, num_movies, embedding_dim= 32)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

print(num_users, num_movies)
print(users.dtype, ratings_t.dtype)

#ορίζουμε πόσα είναι τα epochs
epochs = 400

#Αρχή loop
for epoch in range(epochs):
    #μηδενίζουμε τα gradients
    optimizer.zero_grad()

    #forward:
    predictions = model(users,movies)

    #loss function
    loss = loss_fn(predictions, ratings_t)

    #backward
    loss.backward()

    #step
    optimizer.step()

    #print every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}")
print(f"Τελικό loss: {loss.item():.4f}")