"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
#from typing import final
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',delimiter = ',')

collab_df = pd.read_csv('resources/data/collab_df.csv')

collab_df_test = pd.read_csv('resources/data/collab_df_test.csv')
#ratings_df = pd.read_csv('resources/data/ratings.csv')
#ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD_Sml1.pkl', 'rb'))


def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Get the movieIds of the movies that are queried
    m_id = movies_df.loc[movies_df['title'].isin(movie_list), 'movieId'].iloc[0]

    # Get all users that have rated the movies that were queried, sort them in ratings order
    u_id = collab_df[collab_df['movieId']==m_id].sort_values(by='rating', ascending=False)
    # Get the top 50 users
    u_id = u_id['userId'][:50].values

    # Get only those users from the test df
    test_df = collab_df_test[collab_df_test['userId'].isin(u_id)]

    # Make predictions on the smaller test df
    prediction = []
    for i, r in test_df.iterrows():
        pred = (model.predict(r.userId, r.movieId))
        # Ratings prediction
        result = pred[3]
        prediction.append(result)

    # Combine into one df
    final_df = pd.DataFrame({'userId':test_df['userId'], 'movieId':test_df['movieId'], 'rating':prediction})
    final_df = final_df.sort_values(by='rating', ascending=False)

    # Merge finaldf with movies so that we have the movie titles
    final_df = final_df.merge(movies_df, on='movieId', how='left')

    # return top_n
    result = list(final_df.title[:100])
    result = list(dict.fromkeys(result))
    return result[:top_n]
