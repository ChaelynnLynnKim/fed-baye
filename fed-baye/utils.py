import collections
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Optional, List, Tuple


def load_data(dataset: str = 'movie-lens'):
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


def create_tf_datasets(
    ratings_df: pd.DataFrame,
    batch_size: int = 1,
    max_examples_per_user: Optional[int] = None,
    max_clients: Optional[int] = None
) -> List[tf.data.Dataset]:
    """Creates TF Datasets containing the movies and ratings for all users."""
    if max_clients is not None:
        num_users = min(len(ratings_df.user_id.unique()), max_clients)
        
    def rating_batch_map_fn(rating_batch):
        """Maps a rating batch to an OrderedDict with tensor values."""
        return collections.OrderedDict([
            ("x", tf.cast(rating_batch[:, 0:2], tf.float32)),
            ("y", tf.cast(rating_batch[:, 2:3], tf.float32))
        ])

    tf_datasets = []
    for user_id in ratings_df.user_id.unique():
        user_ratings_df = ratings_df[ratings_df.user_id == user_id]

        tf_dataset = tf.data.Dataset.from_tensor_slices(user_ratings_df)
        tf_dataset = (
            tf_dataset.take(max_examples_per_user)
            .shuffle(buffer_size=max_examples_per_user, seed=42)
            .batch(batch_size)
            .map(rating_batch_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        )
        tf_datasets.append(tf_dataset)

    return tf_datasets
            

def split_tf_datasets(
    tf_datasets: List[tf.data.Dataset],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Tuple[List[tf.data.Dataset], List[tf.data.Dataset], List[tf.data.Dataset]]:
    """Splits a list of user TF datasets into train/val/test by user."""
    np.random.seed(42)
    np.random.shuffle(tf_datasets)
            
    train_idx = int(len(tf_datasets) * train_fraction)
    val_idx = int(len(tf_datasets) * (train_fraction + val_fraction))
            
    return (
        tf_datasets[:train_idx], tf_datasets[train_idx:val_idx], tf_datasets[val_idx:]
    )
