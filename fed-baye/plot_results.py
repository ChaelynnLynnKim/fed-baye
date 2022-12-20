import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional


def get_noise_multiplier(fpath):
    """Get noise multiplier value from file name"""

    multiplier = fpath.split('_')[-2].split('-')[-1]
    if multiplier == '0':
        return 0.0
    elif multiplier == '025':
        return 0.25
    elif multiplier == '050':
        return 0.50
    elif multiplier == '075':
        return 0.75
    elif multiplier == '1':
        return 1.0

def load_plot_data(fpath):
    """Load data for single experiment run"""

    df = pd.read_csv(fpath)
    df['Noise Multiplier'] = get_noise_multiplier(fpath)
    df['Training Round'] = np.arange(1,31)
    df['Trailing Loss'] = df.Loss.ewm(3).mean()
    df['Trailing Accuracy'] = df.Accuracy.ewm(3).mean()
    return df


def make_eval_plots(
    data: pd.DataFrame, metric: str ='Loss',
    clip: float = 1.0, bayesian: bool = True,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    smoothed: bool = True
):
    """Plots results for combination of clipping norm and model type"""

    model_type = 'Bayesian' if bayesian else 'Frequentist'
    plot_title = (
        f'Evaluation {metric} by Federated Training Round | ' +
        f'Max Gradient Norm = {clip} | Model Type = {model_type}'
    )

    if smoothed:
        metric = 'Trailing ' + metric
        plot_title = 'Smoothed ' + plot_title
    
    plt.figure(figsize=(14,8))
    sns.lineplot(x='Training Round', y=metric, hue='Noise Multiplier', data=data)
    plt.ylim([y_min, y_max])
    plt.title(plot_title)
    plt.show();


def main(args):
    """Load data for combination of model type and clipping norm and plot results"""

    # Choose directory where results are stored
    if args.bayesian == True:
        if args.clip == 1.0:
            data_dir = './experiment-results/bayesian/clip-norm-1'
        elif args.clip == 5.0:
            data_dir = './experiment-results/bayesian/clip-norm-5'
    else:
        if args.clip == 1.0:
            data_dir = './experiment-results/frequentist/clip-norm-1'
        elif args.clip == 5.0:
            data_dir = './experiment-results/frequentist/clip-norm-5'

    # Combine results for different noise multiplier values
    results = (
        pd.concat(
            [
                load_plot_data(os.path.join(data_dir, fname))
                for fname in os.listdir(data_dir)
            ],
            axis=0
        )
        .reset_index(drop=True)
    )

    make_eval_plots(
        results, args.metric, args.clip, args.bayesian,
        args.y_min, args.y_max, args.smoothed
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bayesian', default=False, type=bool)
    parser.add_argument('--clip', default=5.0, type=float)
    parser.add_argument('--metric', default='Loss', type=str)
    parser.add_argument('--y-min', default=None, type=float)
    parser.add_argument('--y-max', default=None, type=float)
    parser.add_argument('--smoothed', default=False, type=bool)
    args = parser.parse_args()
    
    main(args)