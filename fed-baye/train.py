import os
import copy
import time
import logging
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from numba import cuda
from datetime import datetime
from infrastructure.server import Server
from utils import load_data, create_tf_datasets, split_tf_datasets


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger()


def main(args):
    
    if args.dataset == 'movie-lens':
        logger.info('Loading movie-lens data')
        ratings, movies, users = load_data()
        ratings['user_id'] = ratings['user_id'].astype(int)
        ratings['movie_id'] = ratings['movie_id'].astype(int)
        ratings['rating'] = ratings['rating'].astype(float)
        movies['movie_id'] = movies['movie_id'].astype(int)
        movie_ratings = ratings.merge(movies, how='left', on='movie_id')
        data = movie_ratings[['user_id', 'movie_id', 'rating']]
        
        output_directory = './experiment-results/movie-lens'
    else:
        raise NotImplementedError('Please choose one of {movie-lens, }')
        
    logger.info('Building federated datasets')
    tf_data = create_tf_datasets(
        ratings_df=data,
        batch_size=5,
        max_examples_per_user=50,
        max_clients=2000
    )

    train_data, val_data, test_data = split_tf_datasets(tf_data)
    
    if args.bayesian == True:
        from models.bayesian import make_bayesian_recommender
        model_fn = make_bayesian_recommender
    else:
        from models.frequentist import make_frequentist_recommender
        model_fn = make_frequentist_recommender
        
    unique_user_ids = movie_ratings.user_id.unique()
    unique_movie_ids = movie_ratings.movie_id.unique()
    
    logger.info('Initializing client-server infrastructure')
    server = Server(
        args, train_data, val_data, model_fn,
        unique_user_ids=unique_user_ids,
        unique_movie_ids=unique_movie_ids
    )

    acc_log = []
    loss_log = []
    logger.info('Starting training')
    for t in range(server.rounds):
        epoch_start = time.time()
        val_acc, val_loss = server.global_update()
        epoch_duration = time.time() - epoch_start
        acc_log.append(val_acc.numpy())
        loss_log.append(val_loss.numpy())
        msg = (
            f'Training round = {t+1:d}, acc = {acc_log[-1]:.4f}, ' +
            f'loss = {loss_log[-1]:.4f} Round time: {epoch_duration:.2f}s'
        )
        logger.info(msg)
        
    clip = int(args.clip)
    if args.noise_multiplier == 0.0:
        multiplier_str = '0'
    elif args.noise_multiplier == 0.25:
        multiplier_str = '025'
    elif args.noise_multiplier == 0.50:
        multiplier_str = '050'
    elif args.noise_multiplier == 0.75:
        multiplier_str = '075'
    elif args.noise_multiplier == 1.0:
        multiplier_str = '1'
    elif args.noise_multiplier == 2.0:
        multiplier_str = '2'
    timestamp = datetime.now().strftime("%Y%m%d %H-%M-%S")
    model_type = 'bayesian' if args.bayesian else 'frequentist'
    fname = (
        f'results_{model_type}_layers-{args.layers}_units-{args.units}' +
        f'_noise-{multiplier_str}_clip-{clip}_{timestamp}.csv'
    )
    path = os.path.join(output_directory, fname)
    
    logger.info(f'Saving results to {path}')
    results = pd.DataFrame({'Accuracy': acc_log, 'Loss': loss_log})
    results.to_csv(path, index=False, mode='x')
    
    logger.info('Federated training complete')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='movie-lens', type=str)
    parser.add_argument('--bayesian', default=False, type=bool)
    parser.add_argument('--layers', default=3, type=int)
    parser.add_argument('--units', default=64, type=int)
    parser.add_argument('--clients', default=50, type=int)
    parser.add_argument('--clip', default=5.0, type=float)
    parser.add_argument('--noise-multiplier', default=0.0, type=float)
    parser.add_argument('--rounds', default=20, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--learning-rate', default=0.01, type=float)
    args = parser.parse_args()
    
    cuda.get_current_device().reset()
    main(args)
