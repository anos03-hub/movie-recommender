import torch
import torch.nn as nn
import pandas as pd
import sys, os
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from neural_data import prepare_ids, to_tensors
from neural_model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'ratings.csv')

#load dataset, prepare it with prepare ids and split to train and test
ratings = pd.read_csv(DATA_PATH)
ratings, num_users, num_movies = prepare_ids(ratings)
train_df, test_df = train_test_split(ratings, test_size = 0.2, random_state = 42)

#convert to tensors both train and test dataframes
users, movies, ratings_t = to_tensors(train_df)
users_test, movies_test, ratings_test = to_tensors(test_df)

model = Model(num_users, num_movies, embedding_dim= 16)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)

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

model.eval()
with torch.no_grad():
    test_predictions = model(users_test, movies_test)
    mae = torch.abs(test_predictions - ratings_test).mean()
    print(f"Test MAE: {mae.item():.4f}")


torch.save(model.state_dict(), 'movie_model.pth')
print("Model saved!")