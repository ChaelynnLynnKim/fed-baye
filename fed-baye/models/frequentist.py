import tensorflow as tf

def make_frequentist_recommender(unique_user_ids, unique_movie_ids, layers, units):
    inputs = tf.keras.Input(shape=(2,))
    user_embedding = tf.keras.layers.Embedding(len(unique_user_ids) + 1, 256)(inputs[:,0])
    user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)
    for _ in range(layers):
        user_embedding = tf.keras.layers.Dense(units, activation='relu')(user_embedding)
        user_embedding = tf.keras.layers.BatchNormalization()(user_embedding)

    movie_embedding = tf.keras.layers.Embedding(len(unique_movie_ids) + 1, 256)(inputs[:,1])
    movie_embedding = tf.keras.layers.BatchNormalization()(movie_embedding)
    for _ in range(layers):
        movie_embedding = tf.keras.layers.Dense(units, activation='relu')(movie_embedding)
        movie_embedding = tf.keras.layers.BatchNormalization()(movie_embedding)
    
    embedding = tf.concat([user_embedding, movie_embedding], axis=1)
    for _ in range(layers):
        embedding = tf.keras.layers.Dense(units, activation='relu')(embedding)
        
    prediction = tf.keras.layers.Dense(1)(embedding)
    model = tf.keras.Model(inputs=inputs, outputs=prediction)
    
    return model
