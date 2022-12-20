import os
import time
import logging
import argparse

import pandas as pd

from numba import cuda
from infrastructure.server import Server
from utils import load_data, create_datasets, split_datasets

# limit TensorFlow logs for more readable output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger()


def main(args):
    """Simulates federated learning of MovieLens100k dataset

    Loads MovieLens100k data, preprocesses it, builds client-server
    infrastructure, and carries out federated learning simulation
    according to command-line arguments.

    Args
    ----
    args : Namespace
        Populated Namespace object containing experiment settings

    Returns
    -------
    CSV file with Accuracy and Loss metrics saved to output directory
    """

    # load and preprocess MovieLens100k dataset 
    logger.info('Loading movie-lens data')
    ratings, movies, _ = load_data()
    ratings['user_id'] = ratings['user_id'].astype(int)
    ratings['movie_id'] = ratings['movie_id'].astype(int)
    ratings['rating'] = ratings['rating'].astype(float)
    movies['movie_id'] = movies['movie_id'].astype(int)
    movie_ratings = ratings.merge(movies, how='left', on='movie_id')
    data = movie_ratings[['user_id', 'movie_id', 'rating']]
        
    # convert centralized datasets into subsets 
    # for federated learning simulation
    logger.info('Building federated datasets')
    federated_data = create_datasets(
        ratings_df=data,
        batch_size=5,
        max_examples_per_user=50,
        max_clients=2000
    )

    # split into train and test sets
    train_data, val_data, _ = split_datasets(federated_data)
    
    # specifying whether recommender is regular neural network
    # or Bayesian neural network and where results should be saved
    if args.bayesian == True:
        from models.bayesian import make_bayesian_recommender
        model_fn = make_bayesian_recommender
        if args.clip == 1.0:
            output_directory = './experiment-results/bayesian/clip-norm-1'
        elif args.clip == 5.0:
            output_directory = './experiment-results/bayesian/clip-norm-5'
    else:
        from models.frequentist import make_frequentist_recommender
        model_fn = make_frequentist_recommender
        if args.clip == 1.0:
            output_directory = './experiment-results/frequentist/clip-norm-1'
        elif args.clip == 5.0:
            output_directory = './experiment-results/frequentist/clip-norm-5'
        
    # Collect unique user ids and movie ids 
    # to build vocabularies
    unique_user_ids = movie_ratings.user_id.unique()
    unique_movie_ids = movie_ratings.movie_id.unique()
    
    # Build centralized server object and global model
    logger.info('Initializing client-server infrastructure')
    server = Server(
        args, train_data, val_data, model_fn,
        unique_user_ids=unique_user_ids,
        unique_movie_ids=unique_movie_ids
    )

    # iterate through federated learning rounds 
    # consisting of local and global model updates
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
        
    # format output CSV file name
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

    model_type = 'bayesian' if args.bayesian else 'frequentist'
    fname = (
        f'results_{model_type}_layers-{args.layers}_units-{args.units}' +
        f'_noise-{multiplier_str}_clip-{clip}.csv'
    )
    path = os.path.join(output_directory, fname)
    
    # aggregate accuracy and loss results in 
    # pandas dataframe and save to CSV
    logger.info(f'Saving results to {path}')
    results = pd.DataFrame({'Accuracy': acc_log, 'Loss': loss_log})
    results.to_csv(path, index=False, mode='x')
    
    logger.info('Federated training complete')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Regular or Bayesian neural network
    parser.add_argument('--bayesian', default=False, type=bool)
    # number of layers in embedding and prediction sub-models
    parser.add_argument('--layers', default=3, type=int)
    # number of units in each sub-model layer
    parser.add_argument('--units', default=64, type=int)
    # number of clients to randomly sample during each training round
    parser.add_argument('--clients', default=50, type=int)
    # Maximum L2 norm for gradients
    parser.add_argument('--clip', default=5.0, type=float)
    # Multiplier to determine standard deviation of noise distribution
    parser.add_argument('--noise-multiplier', default=0.0, type=float)
    # Number of rounds of federated learning updates  
    parser.add_argument('--rounds', default=20, type=int)
    # Number of epochs to train each client model
    parser.add_argument('--epochs', default=1, type=int)
    # Batch size for client model training
    parser.add_argument('--batch-size', default=5, type=int)
    # Learning rate for client model training
    parser.add_argument('--learning-rate', default=0.01, type=float)
    args = parser.parse_args()
    
    # Clear GPU memory from previous experiments
    cuda.get_current_device().reset()
    main(args)
