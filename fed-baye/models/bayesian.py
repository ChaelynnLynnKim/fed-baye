import tensorflow as tf
import tensorflow_probability as tfp

import numpy.typing as npt


def make_bayesian_recommender(
    unique_user_ids: npt.NDArray[int],
    unique_movie_ids: npt.NDArray[int],
    layers: int,
    units: int
):
    """
    Builds recommender system with Bayesian neural network sub-models

    Builds recommender system with Bayesian sub-models using
    TensorFlow Probability and TensorFlow's functional API.
    Three separate blocks in the model - one for user embeddings,
    one for movie embeddings, and one for prediction of user-movie ratings.

    Args
    ----
    unique_user_ids : npt.NDArray[int]
        NumPy array of unique user ids in federated dataset

    unique_movie_ids : npt.NDArray[int]
        NumPy array of unique movie ids in federated dataset

    layers : int
        Number of layers for each of the sub-models

    units : int
        Number of units for each layer in the sub-models

    Returns
    -------
    tf.keras.Model
        Keras model that takes in user and movie ids and 
        predicts user-movie ratings
    """
    def prior(kernel_size, bias_size, dtype=None):
        """Defines the fixed prior distribution for the models' weights"""
        n = kernel_size + bias_size
        prior_model = tf.keras.Sequential([
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ])

        return prior_model

    def posterior(kernel_size, bias_size, dtype=None):
        """Defines the posterior for the models' weights to be updated during training"""
        n = kernel_size + bias_size
        posterior_model = tf.keras.Sequential([
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ])

        return posterior_model
    
    # Input layer defining the shape of the recommender's inputs
    inputs = tf.keras.Input(shape=(2,))
    
    # User embeddings sub-model with an initial embedding layer, followed by 
    # batch normalization and blocks of Bayesian dense layers, each with its
    # own batch normalization
    user_embedding = tf.keras.layers.Embedding(len(unique_user_ids) + 1, units)(inputs[:,0])
    user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    for _ in range(layers):    
        user_embedding = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            activation='sigmoid'
        )(user_embedding)
        user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    
    # Movie embeddings sub-model with an initial embedding layer, followed by 
    # batch normalization and blocks of Bayesian dense layers, each with its
    # own batch normalization
    movie_embedding = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, units)(inputs[:,1])
    movie_embedding = tf.keras.layers.BatchNormalization()(movie_embedding)
    for _ in range(layers):    
        movie_embedding = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            activation='sigmoid'
        )(movie_embedding)
        movie_embedding = tf.keras.layers.BatchNormalization()(movie_embedding)
    
    # Final model block that concatenates outputs from user and movie
    # sub-models, adds additional Bayesian layers with batch normalization,
    # and returns point estimate of user-movie rating
    embedding = tf.concat([user_embedding, movie_embedding], axis=1)
    for _ in range(layers):
        embedding = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            activation='sigmoid'
        )(embedding)
        embedding = tf.keras.layers.BatchNormalization()(embedding)
    prediction = tf.keras.layers.Dense(1)(embedding)

    # Above sub-models packaged as a single Keras model
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
    return model
    