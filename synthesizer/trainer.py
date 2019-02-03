from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from absl import logging


class Trainer():

  def __init__(self):
    pass

  def train(self, train_step_fn, iterations):
    hooks = []
    hooks.append(tf.train.StopAtStepHook(last_step=iterations))
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      while not sess.should_stop():
        train_step_fn(sess)
