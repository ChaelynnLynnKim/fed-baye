import tensorflow as tf

from typing import Optional


class RatingAccuracy(tf.keras.metrics.Mean):
    
    def __init__(self, name: str = 'rating_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        
    def update_state(
        self, y_true: tf.Tensor, y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None
    ):
        absolute_diffs = tf.abs(y_true - y_pred)
        example_accuracies = tf.less_equal(absolute_diffs, 0.5)
        super().update_state(example_accuracies, sample_weight=sample_weight)