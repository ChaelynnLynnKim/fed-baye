import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm
from .client import Client
from .metrics import RatingAccuracy
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Server:
    def __init__(
        self, args, train_data, val_data, model_fn, **kwargs
    ):
        super(Server, self).__init__()
        self.n_clients = args.clients
        self.clip = args.clip
        self.noise_multiplier = args.noise_multiplier
        self.sigma = self.clip * self.noise_multiplier
        self.rounds = args.rounds
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.layers = args.layers
        self.units = args.units
        self.model_fn = model_fn
        
        self.train_data = train_data
        self.raw_val_data = pd.concat(val_data, axis=0)
        self.val_data = (
            tf.data.Dataset.from_tensor_slices((
                tf.convert_to_tensor(self.raw_val_data[['user_id','movie_id']].values, dtype=tf.float32),
                tf.convert_to_tensor(self.raw_val_data['rating'].values, dtype=tf.float32),
            ))
            .batch(self.batch_size)
        )
        # self.input_size = int(self.data[0].shape[1])
        self.learning_rate = args.learning_rate
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        
        if args.dataset == 'movie-lens':
            self.unique_user_ids = kwargs['unique_user_ids']
            self.unique_movie_ids = kwargs['unique_movie_ids']
            self.global_model = model_fn(
                self.unique_user_ids, self.unique_movie_ids,
                args.layers, args.units
            )
        
        self.val_acc_metric = RatingAccuracy()
        self.clients = []
        
    def federated_averaging(self, user_indices):
        client_weights = [self.clients[idx].model.get_weights() for idx in range(len(user_indices))]
        mean_client_weights = np.array(client_weights, dtype=object).mean(axis=0)
        self.global_model.set_weights(mean_client_weights)

        return copy.deepcopy(self.global_model.get_weights())
    
    def broadcast_model(self, weight_updates):
        for client in self.clients:
            client.receive_global_weights(weight_updates)
            
    @tf.function
    def test_step(self, x, y):
        val_logits = self.global_model(x, training=False)
        loss_value = self.loss_fn(y, val_logits)
        self.val_acc_metric.update_state(y, val_logits)
        return loss_value
            
    def evaluate(self):
        val_loss = 0
        for i, (x_batch_val, y_batch_val) in enumerate(self.val_data):
            val_loss += self.test_step(x_batch_val, y_batch_val)            

        val_acc = self.val_acc_metric.result()
        self.val_acc_metric.reset_states()
        
        val_loss /= (i + 1)
        return val_acc, val_loss
    
    def global_update(self):
        user_indices = np.random.choice(range(len(self.train_data)), int(self.n_clients), replace=False)
        for idx in tqdm(user_indices, desc='Building client models'):
            self.clients.append(
                Client(
                    self.model_fn(
                        self.unique_user_ids, self.unique_movie_ids,
                        self.layers, self.units
                    ),
                    self.train_data[idx], 
                    self.learning_rate,
                    self.epochs,
                    self.batch_size,
                    self.clip,
                    self.noise_multiplier
                )
            )
        weight_updates = self.global_model.get_weights()
        self.broadcast_model(weight_updates)

        for client in tqdm(self.clients, desc='Updating client models'):
            client.update()
        self.federated_averaging(user_indices)
        acc, loss = self.evaluate()

        self.clients = []

        return acc, loss