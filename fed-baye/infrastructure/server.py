import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from tqdm import tqdm
from .client import Client
from .metrics import RatingAccuracy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Server:
    """
    Class representation of a federated learning centralized server

    Attributes
    ----------
    batch_size : int
        Batch size used for training client models

    clients : List
        List for storing `Client` objects

    clip : float
        Maximum L2 norm for gradients, used to clip large gradient values

    epochs : int
        Number of epochs to train client models
    
    layeers : int
        Number of layers used in global and client sub-model blocks

    learning_rate : float
        Learning rate used for training client models

    loss_fn : tf.keras.losses.MeanSquaredError
        Loss function object used for training client models
    
    model_fn : <class 'function'>
        Function used to create global and local models

    n_clients : int
        Number of clients sampled for inclusion in training

    noise_multiplier : float
        Hyperparameter controlling standard deviation of noise distribution

    rounds : int
        Number of rounds of federated learning

    sigma : float
        Standard deviation of noise distribution

    units : int
        Number of units in each sub-model layer

    unique_movie_ids : npt.NDArray[int]
        NumPy array of unique movie ids in dataset
    
    unique_user_ids : npt.NDArray[int]
        NumPy array of unique user ids in dataset

    val_acc_metric : RatingAccuracy
        RatingAccuracy object used in global model evaluation on validation dataset

    train_data : list[pd.DataFrame]
        List of dataframes representing all client data avilable for training

    raw_val_data : pd.DataFrame
        pandas DataFrame representing validation dataset

    val_data : tf.data.Dataset
        `raw_val_data` converted into TensorFlow Dataset for downstream use

    Methods
    -------
    federated_averaging(user_indices)
        Clients and averages client weights tied to `user_indices`
        to update global model

    broadcast_model(weight_updates)
        Broadcasts current global model weights to client models
        for client model weight initialization

    test_step(x, y)
        Performs evaluation step of global model using 
        inputs `x` and ground truths `y`

    evaluate()
        Calculates loss and accuracy metrics for global model on validation data

    global_update()
        Performs round of federated learning by sampling client data, creating
        client models, broadcasting global model weights, training client models,
        aggregating client model weights, and updating global model weights
    """
    def __init__(
        self, args, train_data, val_data, model_fn, **kwargs
    ):
        """
        Constructs class attributes needed to simulate server activity 
        in federated learning

        Args
        ----
        args : Namespace
            Populated Namespace object containing experiment settings

        train_data : List[pd.DataFrame]
            List of client datasets for training client models

        val_data : List[pd.DataFrame]
            List of client datasets for evaluating global model

        model_fn : <class 'function'>
            Function for creating recommender system
        """
        super(Server, self).__init__()
        self.batch_size = args.batch_size
        self.clients = []
        self.clip = args.clip
        self.epochs = args.epochs
        self.global_model = model_fn(
            self.unique_user_ids, self.unique_movie_ids,
            args.layers, args.units
        )
        self.layers = args.layers
        self.learning_rate = args.learning_rate
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.model_fn = model_fn
        self.n_clients = args.clients
        self.noise_multiplier = args.noise_multiplier
        self.rounds = args.rounds
        self.sigma = self.clip * self.noise_multiplier
        self.units = args.units
        self.unique_movie_ids = kwargs['unique_movie_ids']
        self.unique_user_ids = kwargs['unique_user_ids']
        self.val_acc_metric = RatingAccuracy()
        
        self.train_data = train_data
        self.raw_val_data = pd.concat(val_data, axis=0)
        self.val_data = (
            tf.data.Dataset.from_tensor_slices((
                tf.convert_to_tensor(self.raw_val_data[['user_id','movie_id']].values, dtype=tf.float32),
                tf.convert_to_tensor(self.raw_val_data['rating'].values, dtype=tf.float32),
            ))
            .batch(self.batch_size)
        )
        
    def federated_averaging(self, user_indices):
        """Aggregates client model weights and updates global model

        Aggregates client model weights from `self.clients`, takes
        mean of across all client weights, and replaces global model
        weights with mean client weights.

        Args
        ----
        user_indices : npt.NDArray[int]
            NumPy array of indices pointing to client datasets
            used in current training round

        Returns
        -------
        npt.NDArray
            NumPy array containing updated global model weights
        
        """
        client_weights = [
            self.clients[idx].model.get_weights() for idx in range(len(user_indices))
        ]
        mean_client_weights = np.array(client_weights, dtype=object).mean(axis=0)
        self.global_model.set_weights(mean_client_weights)

        return copy.deepcopy(self.global_model.get_weights())
    
    def broadcast_model(self, weight_updates):
        """Broadcasts updated global model weights to client models"""
        for client in self.clients:
            client.receive_global_weights(weight_updates)
            
    @tf.function
    def test_step(self, x, y):
        """Performs evaluation step for global model
        
        Performs evluation step for global model. Calculates
        logits and loss. Updates internal validation
        accuracy object
        """
        val_logits = self.global_model(x, training=False)
        loss_value = self.loss_fn(y, val_logits)
        self.val_acc_metric.update_state(y, val_logits)
        return loss_value
            
    def evaluate(self):
        """Calculates validation loss and accuracy metrics for global model"""
        val_loss = 0
        for i, (x_batch_val, y_batch_val) in enumerate(self.val_data):
            val_loss += self.test_step(x_batch_val, y_batch_val)            

        val_acc = self.val_acc_metric.result()
        self.val_acc_metric.reset_states()
        
        val_loss /= (i + 1)
        return val_acc, val_loss
    
    def global_update(self):
        """Performs round of federated learning

        Samples `self.n_clients` clients for training and build
        `self.n_clients` Client objects stored in `self.clients`.
        Gets current global model weights and initializes client 
        model weights with them. Updates each client model, conducts
        federated averaging, and evaluates updated global model.
        Resets `self.clients` to empty List to ensure that not all
        available memory is consumed and training fails.
        """
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
        