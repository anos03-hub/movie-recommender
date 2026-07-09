import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super().__init__() #instantiate our nn.Module
        self.em_users = nn.Embedding(num_users, embedding_dim)
        self.em_movies = nn.Embedding(num_movies, embedding_dim)

    def forward(self, user, movie):
        user_vec = self.em_users(user)
        movie_vec = self.em_movies(movie)
        forw = torch.sum(user_vec * movie_vec , dim = 1)
        return forw  