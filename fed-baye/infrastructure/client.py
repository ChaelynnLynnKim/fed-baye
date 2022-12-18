import copy
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Client:
    def __init__(
        self, model, data, learning_rate,
        epochs, batch_size, clip, noise_multiplier
    ):
        super(Client, self).__init__()
        self.batch_size = batch_size # int
        self.dataset = (
            tf.data.Dataset.from_tensor_slices((
                data[['user_id','movie_id']].values, data['rating'].values
            ))
            .batch(self.batch_size)
        )# tensorflow dataset
        self.data_size = len(self.dataset) # int
        self.clip = clip # float, probably 1
        self.noise_multiplier = noise_multiplier # noise multiplier between 0 and 1 
        self.sigma = self.clip * self.noise_multiplier
        self.learning_rate = learning_rate # float
        self.epochs = epochs # int 
        self.model = model # tensorflow model
        self.grads = None
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.SGD(0.01, momentum=0.9)
        
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        
        # clip gradients
        grads = [tf.clip_by_norm(g, self.clip) for g in grads]

        # add noise
        grads = [
            grad +
            tf.keras.backend.random_normal(
                tf.keras.backend.shape(grad), mean=0, stddev=self.sigma, dtype=tf.keras.backend.dtype(grad)
            )
            for grad in grads
        ]
        
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss_value
        
    def receive_global_weights(self, global_weights):
        '''
        Receive global model updates from server and
        update local model
        '''
        self.model.set_weights(global_weights)
        
    def update(self):
        for epoch in range(self.epochs):
            total_loss = 0.
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(self.dataset):
                loss_value = self.train_step(x_batch_train, y_batch_train)
                total_loss += loss_value