import tensorflow as tf

    
def recommender_model_fn():
    inputs = tf.keras.Input(shape=(2,))
    user_embedding = tf.keras.layers.Embedding(len(unique_user_ids) + 1, 256)(inputs[:,0])
    user_embedding = tf.keras.layers.Dense(128, activation='relu')(user_embedding)
    user_embedding = tf.keras.layers.Dense(64, activation='relu')(user_embedding)
    user_embedding = tf.keras.layers.Dense(32, activation='relu')(user_embedding)

    movie_embedding = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, 256)(inputs[:,1])
    movie_embedding = tf.keras.layers.Dense(128, activation='relu')(movie_embedding)
    movie_embedding = tf.keras.layers.Dense(64, activation='relu')(movie_embedding)
    movie_embedding = tf.keras.layers.Dense(32, activation='relu')(movie_embedding)

    prediction = tf.keras.layers.Dot(1, normalize=False)([user_embedding, movie_embedding])
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
    
    return tff.learning.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.MeanSquaredError(),
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
        ),
        metrics=[RatingAccuracy()]
    )


def bayesian_recommender_fn():
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
        
        posterior_model
    
    inputs = tf.keras.Input(shape=(2,))
    user_embedding = tf.keras.layers.Embedding(len(unique_user_ids) + 1, 256)(inputs[:,0])
    user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    user_embedding = tfp.layers.DenseVariational(
        units=64,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight = 1 / train_size,
        activation='sigmoid'
    )(user_embedding)
    user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    user_embedding = tfp.layers.DenseVariational(
        units=64,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight = 1 / train_size,
        activation='sigmoid'
    )(user_embedding)
    user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    
    movie_embedding = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, 256)(inputs[:,1])
    movie_embedding = tf.keras.layers.BatchNormalization()(movie_embedding)
    movie_embedding = tfp.layers.DenseVariational(
        units=64,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight = 1 / train_size,
        activation='sigmoid'
    )(movie_embedding)
    movie_embedding = tf.keras.layers.BatchNormalization()(movie_embedding)
    movie_embedding = tfp.layers.DenseVariational(
        units=64,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight = 1 / train_size,
        activation='sigmoid'
    )(movie_embedding)
    movie_embedding = tf.keras.layers.BatchNormalization()(movie_embedding)
    
    embedding = tf.concat([user_embedding, movie_embedding], axis=1)
    embedding = tf.keras.layers.Dense(64, activation='relu')(embedding)
    embedding = tf.keras.layers.Dense(32, activation='relu')(embedding)
    prediction = tf.keras.layers.Dense(1)(embedding)
    
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
    
    return tff.learning.from_keras_model(
        keras_model=model,
        loss=tf.keras.losses.MeanSquaredError(),
        input_spec=collections.OrderedDict(
            x=tf.TensorSpec(shape=[None, 2], dtype=tf.float32),
            y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
        ),
        metrics=[RatingAccuracy()]
    )
