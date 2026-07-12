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
        self.user_bias = nn.Embedding(num_users, 1)    # ένας αριθμός ανά χρήστη
        self.movie_bias = nn.Embedding(num_movies, 1)  # ένας αριθμός ανά ταινία
        self.global_bias = nn.Parameter(torch.zeros(1)) # ένας αριθμός για όλους

    def forward(self, user, movie):
        user_vec = self.em_users(user)
        movie_vec = self.em_movies(movie)
        dot = torch.sum(user_vec * movie_vec, dim=1)
        
        u_bias = self.user_bias(user).squeeze()
        m_bias = self.movie_bias(movie).squeeze()
        
        return dot + u_bias + m_bias + self.global_bias