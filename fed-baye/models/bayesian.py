import tensorflow as tf
import tensorflow_probability as tfp

def make_bayesian_recommender(unique_user_ids, unique_movie_ids, layers, units):
    def prior(kernel_size, bias_size, dtype=None):
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
        n = kernel_size + bias_size
        posterior_model = tf.keras.Sequential([
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ])

        return posterior_model
    
    inputs = tf.keras.Input(shape=(2,))
    
    # user embeddings
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
    
    # movie embeddings
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
    
    # concatenated embedding
    embedding = tf.concat([user_embedding, movie_embedding], axis=1)
    for _ in range(layers):
        embedding = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            activation='sigmoid'
        )(embedding)
        embedding = tf.keras.layers.BatchNormalization()(embedding)
    
    # prediction head
    prediction = tf.keras.layers.Dense(1)(embedding)
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
    
    return model
    