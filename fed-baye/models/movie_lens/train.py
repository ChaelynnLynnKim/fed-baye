import os
import argparse
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tff.backends.native.set_local_python_execution_context(clients_per_thread=5)

total_clients = len(unique_user_ids)

def train(rounds, noise_multiplier, clients_per_round, data_frame):
    aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
        noise_multiplier, clients_per_round
    )
    
    # sampling_prob = clients_per_round / total_clients
    
    learning_process = tff.learning.algorithms.build_unweighted_fed_avg(
        recommender_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0, momentum=0.9),
        model_aggregator=aggregation_factory
    )
    
    eval_process = tff.learning.build_federated_evaluation(recommender_model_fn)
    
    state = learning_process.initialize()
    for round in range(rounds):
        model_weights = learning_process.get_model_weights(state)
        metrics = eval_process(model_weights, val_data)['eval']
        data_frame = data_frame.append(
            {
                'Round': round,
                'NoiseMultiplier': noise_multiplier,
                **metrics
            },
            ignore_index=True
        )
        if round % 5 == 0:
            if round < 25 or round % 25 == 0:
                print(f'Round {round:3d}: {metrics}')
                
        # x = np.random.uniform(size=total_clients)
        # sampled_clients = [
        #     train_data.client_ids[i] for i in range(total_clients)
        #     if x[i] < sampling_prob
        # ]
        # sampled_train_data = [
        #     train_data.create_tf_dataset_for_client(client)
        #     for client in sampled_clients
        # ]
        
        federated_train_data = (
            np.random.choice(train_data, size=clients_per_round, replace=False)
            .tolist()
        )
        result = learning_process.next(state, federated_train_data)
        state = result.state
        metrics = result.metrics

    model_weights = learning_process.get_model_weights(state)
    metrics = eval_process(model_weights, val_data)['eval']
    print(f'Round {rounds:3d}: {metrics}')
    data_frame = data_frame.append(
        {
            'Round': rounds,
            'NoiseMultiplier': noise_multiplier,
            **metrics
        },
        ignore_index=True
    )

    return data_frame


def main(args):
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', default=100)
    parser.add_argument('--clients-per-round', default=50)
    parser.add_argument('--embedding-dim', default=32)
    parser.add_argument('--layers', default=3)
    parser.add_argument('--units', default=64)
    parser.add_argument('--bayes', default=True)
    parser.add_argument('--output-directory', default='./output')
    args = parser.parse_args()
    
    main(args)