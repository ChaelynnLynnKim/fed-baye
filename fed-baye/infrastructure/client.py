import copy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Client:
    """
    Class representation of a federated learning client 

    Attributes
    ----------
    batch_size : int
        Batch size used for training client model

    clip : float
        Maximum L2 norm for gradients, used to clip large gradient values

    dataset : tf.data.Dataset
        Local client data

    epochs : int
        Number of epochs to train client model
    
    learning_rate : float
        Learning rate used for training client model

    loss_fn : tf.keras.losses.MeanSquaredError
        Loss function object used for training client model
    
    model : tf.keras.Model
        Local client copy of global model

    noise_multiplier : float
        Hyperparameter controlling standard deviation of noise distribution

    optimizer : tf.keras.optimizers.SGD
        Optimizer used for training client model

    sigma : float
        Standard deviation of noise distribution

    Methods
    -------
    train_step(x, y)
        Performs training step of client model using 
        inputs `x` and ground truths `y`

    receive_global_weights(global_weights)
        Replaces local model weights with global weights provided by server

    update()
        Performs iterations of training steps for `self.epochs` epochs
    """
    def __init__(
        self, batch_size, clip, data, epochs,
        model, learning_rate, noise_multiplier
    ):
        """
        Constructs class attributes needed to simulate client activity 
        in federated learning

        Args
        ----
        batch_size : int 
            Batch size used during training

        clip : float 
            Maximum L2 norm for gradients, used to clip large gradient values

        data : tf.data.Dataset
            Client's data

        epochs : int
            Number of epochs to train
        
        model : tf.keras.Model
            Local client copy of global model

        learning_rate : float
            Learning rate used for training

        noise_multiplier : float
            Hyperparameter controlling standard deviation of noise distribution
        """
        super(Client, self).__init__()
        self.batch_size = batch_size
        self.clip = clip 
        self.dataset = (
            tf.data.Dataset.from_tensor_slices((
                data[['user_id','movie_id']].values, data['rating'].values
            ))
            .batch(self.batch_size)
        )
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.optimizer = tf.keras.optimizers.SGD(self.learning_rate, momentum=0.9)
        self.sigma = self.clip * self.noise_multiplier
        
    @tf.function
    def train_step(self, x, y):
        """Performs training step for local client model
        
        Performs training step for client model. Calculates
        logits, loss, and gradients. Clips gradients based on
        `self.clip` and adds random sample for normal distribution
        to each gradients. Applies clipped and noisy gradients to
        client model's weights via `self.optimizer`.
        """

        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        
        # clip gradients
        grads = [tf.clip_by_norm(g, self.clip) for g in grads]

        # add noise by drawing from normal distribution with 
        # mean 0 and standard deviation `self.sigma`
        grads = [
            grad +
            tf.keras.backend.random_normal(
                tf.keras.backend.shape(grad),
                mean=0,
                stddev=self.sigma,
                dtype=tf.keras.backend.dtype(grad)
            )
            for grad in grads
        ]
        
        # update model weights with clipped, noisy gradients
        # and stochastic gradient descent
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value
        
    def receive_global_weights(self, global_weights):
        """Receives global weights from server and replaces local weights with them"""
        self.model.set_weights(global_weights)
        
    def update(self):
        """Trains client model for `self.epochs` epochs"""
        for _ in range(self.epochs):
            total_loss = 0.
            for _, (x_batch_train, y_batch_train) in enumerate(self.dataset):
                loss_value = self.train_step(x_batch_train, y_batch_train)
                total_loss += loss_value
