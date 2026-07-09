import pandas as pd
import numpy as np
import torch
#Χρειάζεται να συνδέσουμε τα user id με δείκτες, οπότε ο user_id = 1 του csv θα αντιστοιχει στον δείκτη 0, κοκ, το ίδιο κάνω και για τις ταινίες. 
#Αυτό διότι το embedding θέλει δείκτες και όχι τα ids.

def prepare_ids(ratings):
    unique_id = ratings['userId'].unique() #get unique values of users from ratings.csv
    #create dic of users
    user_map = {}
    for i, user_id in enumerate(unique_id):
        user_map[user_id] = i
    
    unique_movies = ratings['movieId'].unique()  #get unique values of movies from ratings.csv
    movie_map = {}
    for i, movie_id in enumerate(unique_movies):
         movie_map[movie_id] = i

    ratings['user_idx'] = ratings['userId'].map(user_map)
    ratings['movie_idx'] = ratings['movieId'].map(movie_map)    

    return ratings, len(user_map), len(movie_map)


#TEST THE ABOVE CODE
##if __name__ == '__main__':
    import pandas as pd
    r = pd.read_csv('data/ratings.csv')
    r, nu, nm = prepare_ids(r)
    print(f"Χρήστες: {nu}, Ταινίες: {nm}")
    print(r[['userId', 'user_idx', 'movieId', 'movie_idx']].head())


#convert pandas df -> torch tensor, and also convert the dtypes as they should be
def to_tensors(ratings):
    users = torch.tensor(ratings['user_idx'].values, dtype=torch.long)
    movies = torch.tensor(ratings['movie_idx'].values, dtype = torch.long)
    ratings_t = torch.tensor(ratings['rating'].values, dtype = torch.float) # ; το rating, αλλά τι dtype; (δεκαδικός!
    return users, movies, ratings_t

#Test the above
##if __name__ == '__main__':
    import pandas as pd
    r = pd.read_csv('data/ratings.csv')
    r, nu, nm = prepare_ids(r)
    u, m, rt = to_tensors(r)
    print(u.dtype, m.dtype, rt.dtype)   # θέλουμε: torch.int64, torch.int64, torch.float32
    print(u[:5], rt[:5])