from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import logging

# from synthesizer import models


# TODO(adam): Is a class even necessary or can it be a standalone function?
class Trainer():

  def __init__(self):
    raise NotImplementedError

  # TODO(adam): Include Pop based training (And other training algorithms?)
  # TODO(adam): Distributed? Need chief, workers, and ps
  def train(self, train_step_fn, iterations):
    hooks = []
    hooks.append(tf.train.StopAtStepHook(last_step=iterations))
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      while not sess.should_stop():
        train_step_fn(sess)
