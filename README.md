 Recommendation System with a Restricted Boltzmann Machine (RBM)
 Overview

This project implements a Collaborative Filtering based Recommendation System using a Restricted Boltzmann Machine (RBM) in TensorFlow 2.x.

Collaborative Filtering: Recommends items by finding patterns in users’ ratings, assuming that similar users tend to like similar items.

RBM: Learns latent features of users and movies. These features are then used to predict how a user would rate unseen movies, enabling personalized recommendations.

By the end of this project, we can recommend movies to users based on their rating history.

Dataset

We use the MovieLens 1M dataset from GroupLens:

movies.dat → MovieID, Title, Genres

ratings.dat → UserID, MovieID, Rating (1–5), Timestamp

Example:

User 1 rated Toy Story (1995) with 5 stars.

User 2 rated Jurassic Park (1993) with 3 stars.

We convert this into a User-Movie Rating Matrix, where:

Rows = Users

Columns = Movies

Values = Normalized ratings (0–1, with 0 meaning unrated).

 Model Design
 RBM Structure

Visible Layer: One unit per movie (3,883 movies). Each unit stores a user’s rating (0–1).

Hidden Layer: Latent features (50 hidden units).

Weights (W): Connect visible and hidden layers.

Biases: For both visible and hidden layers.

 Training (Contrastive Divergence - CD-1)

Start with real user ratings (v0).

Compute hidden activations (h0).

Reconstruct visible ratings (v1).

Recompute hidden probabilities (h1).

Update weights and biases using difference between real and reconstructed data.

Minimize Reconstruction Error = MSE(v0, v1).

Hyperparameters

Hidden Units: 50

Learning Rate: 0.01

Epochs: 20

Batch Size: 128

 Training

During training, reconstruction error decreases steadily:

Epoch 1/20  - Reconstruction Error: 0.4208
Epoch 10/20 - Reconstruction Error: 0.2180
Epoch 20/20 - Reconstruction Error: 0.1456


This shows the RBM is learning to reconstruct user preferences effectively.

Recommendation Process

Select a user (e.g., User ID 65).

Identify movies they have already rated.

Use RBM to predict ratings for unrated movies.

Sort by predicted rating.

Recommend the Top N movies.

Example output:

==============================================
Top 10 Movie Recommendations for User ID 65
1. The Matrix (1999)
2. Fight Club (1999)
3. The Shawshank Redemption (1994)
4. Pulp Fiction (1994)
...

 Key Insights

RBMs can learn latent factors representing user taste.

Unlike simple matrix factorization, RBMs can capture non-linear interactions between users and items.

This model can be extended to:

Books, songs, e-commerce products.

Larger datasets (Netflix Prize, Amazon Reviews).

Tech Stack

Python 3

TensorFlow 2.x (RBM implementation)

Pandas, NumPy (data preprocessing)

Matplotlib (visualization)

Conclusion

This project demonstrates how a Restricted Boltzmann Machine can be used in a Collaborative Filtering recommendation system.

Learned user preferences from historical ratings.

Successfully generated personalized movie recommendations.

Provides a foundation for applying RBMs to other recommendation tasks.
