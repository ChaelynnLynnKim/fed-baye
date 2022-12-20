import tensorflow as tf
from typing import Optional


class RatingAccuracy(tf.keras.metrics.Mean):
    """
    Class for tracking rating accuracy during training and evaluation.
    Single method `update_state` takes absolute difference of point 
    estimate and ground truth and counts point estimate as correct label
    if it is within 0.5 of the ground truth. Custom accuracy function
    needed because models trained with mean squared error loss.
    """
    def __init__(self, name: str = 'rating_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        
    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ):
        absolute_diffs = tf.abs(y_true - y_pred)
        example_accuracies = tf.less_equal(absolute_diffs, 0.5)
        super().update_state(example_accuracies, sample_weight=sample_weight)