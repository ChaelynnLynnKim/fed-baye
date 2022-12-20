import numpy as np
import pandas as pd

from typing import Optional, List, Tuple


def load_data() -> Tuple[pd.DataFrame]:
    """Loads rating, movie, and user subsets of MovieLens100k data

    Loads relevant parts of MovieLens100k dataset,
    including user viewing data, movie data, and user
    demographic data, and extracts relevant columns

    Returns
    -------
    Tuple[pd.DataFrame]
        User viewing data, movie data, and demographic
        data stored in three separate pandas dataframes
    """
    # ratings data
    # user id | item id | rating | timestamp
    user_ids = []
    movie_ids = []
    ratings = []
    timestamps = []
    with open('../data/u.data', 'rt') as f_rating:
        for line in f_rating.readlines():
            record = line.split()
            user_ids.append(record[0])
            movie_ids.append(record[1])
            ratings.append(record[2])
            timestamps.append(record[3].strip('\n'))
    f_rating.close()
    
    ratings = pd.DataFrame({
        'user_id': user_ids, 'movie_id': movie_ids,
        'rating': ratings, 'timestamp': timestamps
    })
    
    # movie data
    # movie id | movie title
    movie_ids = []
    movie_titles = []
    with open('../data/u.item', 'rt', encoding='latin-1') as f_movie:
        for line in f_movie.readlines():
            record = line.split('|')
            movie_ids.append(record[0])
            movie_titles.append(record[1])
    f_movie.close()
    
    movies = pd.DataFrame({'movie_id': movie_ids, 'movie_title': movie_titles})
    
    # user data
    # user id | age | gender | occupation | zip code
    user_ids = []
    ages = []
    genders = []
    occupations = []
    zip_codes = []
    with open('../data/u.user', 'rt', encoding='latin-1') as f_user:
        for line in f_user.readlines():
            record = line.split('|')
            user_ids.append(record[0])
            ages.append(record[1])
            genders.append(record[2])
            occupations.append(record[3])
            zip_codes.append(record[4].strip('\n'))
    f_user.close()
    
    users = pd.DataFrame({
        'user_id': user_ids, 'age': ages, 'gender': genders,
        'occupation': occupations, 'zip': zip_codes
    })
    
    return ratings, movies, users


def create_datasets(
    ratings_df: pd.DataFrame,
    max_examples_per_user: Optional[int] = None,
) -> List[pd.DataFrame]:
    """Splits centralized dataset into subsets by user

    Args
    ----
    ratings_df : pd.DataFrame
        Centralized dataset containing user ids, movie ids, and user-movie ratings

    max_examples_per_user : int (optional)
        The maximum number of examples records that should be included in a 
        user subset. If greater than 0, randomly sample max_examples_per_user
        records from user subset and discard remaining examples.

    
    Returns
    -------
    List[pd.DataFrame]
        List of dataframes where each dataframe contains records from only a single user
    """

    if max_examples_per_user is not None:
        sample_frac = min(max_examples_per_user / user_ratings_df.shape[0], 1)
    else:
        sample_frac = 1.0

    datasets = []
    for user_id in ratings_df.user_id.unique():
        user_ratings_df = ratings_df[ratings_df.user_id == user_id]
        user_ratings_df = user_ratings_df.sample(frac=sample_frac, replace=False)
        datasets.append(user_ratings_df)

    return datasets
            

def split_datasets(
    datasets: List[pd.DataFrame],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """Splits a list of user datasets into train/val/test
    
    Splits list of federated datasets into train, validation, and test 
    sets based on `train_fraction` and `val_fraction`. Given user will
    only appear in one of train/val/test

    Args
    ----
    datasets : List[pd.DataFrame]
        List of user datasets containing user id, movie id, and rating information

    train_fraction : float
        Percentage of user datasets to include in training federated dataset

    val_fraction : float
        Percentage of user datasets to include in validation federated dataset


    Returns
    -------
    Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]
        A three-item tuple in which the first item is a list of user datasets for 
        federated training, the second item is a list of user datasets for global
        validation, and the third item is a list of user datasets for testing
    """
    
    # set seed and shuffle list of user datasets
    np.random.seed(42)
    np.random.shuffle(datasets)
            
    # determine sizes of training, validation, and test sets
    # based on `train_fraction` and `val_fraction``
    train_idx = int(len(datasets) * train_fraction)
    val_idx = int(len(datasets) * (train_fraction + val_fraction))
            
    return (
        datasets[:train_idx], datasets[train_idx:val_idx], datasets[val_idx:]
    )
