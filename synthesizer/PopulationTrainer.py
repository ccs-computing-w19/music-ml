import tensorflow as tf
from absl import logging

from trainer import Trainer


class PopulationTrainer(Trainer):

  def __init__(self, hyperaprameters, model_init, metric_fn, number_models):
    self._metric_fn = metric_fn
    self._models = []
    for i in range(number_models):
      self._models.append(model_init(name="Model {}".format(i)))

  def train(self, train_step_fn, iterations=None):
    hooks = []
    if iterations:
      hooks.append(tf.train.StopAtStepHook(last_step=iterations))
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      while not sess.should_stop():
        pass
