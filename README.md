# Movie Recommender

A movie recommendation system built with the MovieLens dataset.

The project compares two recommendation approaches:

- item-based collaborative filtering using cosine similarity and Pearson correlation
- matrix factorization with user and movie embeddings implemented in PyTorch

The final model is also used in a small Streamlit app where users can select movies they like and receive recommendations.

## Dataset

The project uses the MovieLens Latest Small dataset:

- about 100,000 ratings
- 610 users
- 9,724 movies

Movies with fewer than 10 ratings and users with fewer than 5 ratings are filtered out to reduce sparsity.

## Approach

### Item-based collaborative filtering

The baseline model estimates a user's rating for a movie using similar movies that the user has already rated.

Two similarity measures are tested:

- Cosine similarity
- Pearson correlation

### Matrix factorization

The second approach uses PyTorch to learn user and movie embeddings.

The predicted rating is based on:

    user_embedding · movie_embedding
    + user_bias
    + movie_bias
    + global_bias

The model is trained using MSE loss and the Adam optimizer.

The main hyperparameters tested were:

- embedding dimension
- weight decay
- number of epochs

## Results

| Method | Precision | Recall | MAE |
|---|---:|---:|---:|
| Collaborative filtering, cosine | 0.6643 | 0.7304 | 0.6506 |
| Collaborative filtering, Pearson | 0.6907 | 0.6620 | 0.6481 |
| Matrix factorization | - | - | 0.81 |

The collaborative filtering models performed better on this dataset.

The best matrix factorization configuration used:

    embedding_dim = 16
    weight_decay = 1e-3
    epochs = 400

Adding user and movie bias terms improved the neural model considerably.

## Streamlit App

The repository includes a Streamlit app that generates recommendations for new users.

The user selects a few movies they like and the app uses the learned movie representations to return similar movies.

Run the app locally with:

    streamlit run app.py

Live demo:

https://movie-recommender-ap03.streamlit.app/



## Project Structure

    movie-recommender/
    ├── data/
    │   ├── ratings.csv
    │   └── movies.csv
    ├── src/
    │   ├── data.py
    │   ├── classic_model.py
    │   ├── evaluate.py
    │   ├── neural_data.py
    │   └── neural_model.py
    ├── main.py
    ├── train.py
    ├── recommend.py
    ├── app.py
    ├── movie_model.pth
    ├── requirements.txt
    └── README.md

## How to Run

Install the dependencies:

    pip install -r requirements.txt

Run the collaborative filtering model:

    python main.py

Train the matrix factorization model:

    python train.py

Launch the Streamlit app:

    streamlit run app.py

## Limitations

The dataset is relatively small, which favors simpler collaborative filtering methods.

The recommendation logic is based mainly on rating patterns, so movies that are mathematically similar are not always thematically similar.

Popular movies can also appear frequently in recommendations because of the limited dataset size.

A natural next step would be to combine collaborative filtering with content-based features such as genres.

## Tech Stack

Python · PyTorch · pandas · NumPy · scikit-learn · Streamlit
