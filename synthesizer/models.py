from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


# TODO(adam): Should split audio in chunks for training more easily.
class AudioEncoderDecoder(tf.keras.models.Model):

  def __init__(self, hidden_layers, name='AudioEncoderDecoder'):
    super(AudioEncoderDecoder, self).__init__(name=name)

    dilation = []
    filters = []
    for i in range(hidden_layers):
      dilation.append(2**i)
      filters.append(4)

    self.dconvs = []
    self.norms = []
    for i in range(hidden_layers):
      with tf.name_scope("convolutions_{}".format(i)):
        self.dconvs.append(
            tf.keras.layers.Conv1D(
                filters=filters[i],
                kernel_size=2,
                dilation_rate=dilation[i],
                padding='same',
                activation=tf.nn.leaky_relu))
        self.norms.append(tf.keras.layers.BatchNormalization())

    with tf.name_scope('output'):
      self.out = tf.keras.layers.Conv1D(
          filters=1, kernel_size=5, padding='same', activation=None)

  def call(self, inputs):
    net = inputs

    skip = []
    for i in range(len(self.dconvs)):
      resid = net
      net = self.dconvs[i](net)
      net = self.norms[i](net)
      net = net + resid
      skip.append(net)

    net = tf.reduce_sum(skip, axis=0)
    net = self.out(net)

    return net


# TODO(adam): Should split audio in chunks for training more easily.
class AudioClassifier(tf.keras.models.Model):

  def __init__(self, name='AudioClassifier'):
    super(AudioClassifier, self).__init__(name=name)
    filters = [3, 5, 10]
    kernel_sizes = [10, 8, 5]
    pool_sizes = [10, 10, 10]

    self.convs = []
    self.pools = []
    self.norms = []

    for i in range(len(filters)):
      self.convs.append(
          tf.keras.layers.Conv1D(
              filters=filters[i],
              kernel_size=kernel_sizes[i],
              padding='same',
              activation=tf.nn.leaky_relu))
      self.pools.append(
          tf.keras.layers.AveragePooling1D(
              pool_size=pool_sizes[i], padding='same'))
      self.norm1 = tf.keras.layers.BatchNormalization()

    with tf.name_scope("output"):
      self.fc = tf.keras.layers.Dense(units=1)

  def call(self, inputs):
    net = inputs
    net = tf.squeeze(net, axis=-1)

    for i in range(len(self.convs)):
      net = self.convs[i](net)
      net = self.pools[i](net)
      net = self.norms[i](net)

    net = tf.reduce_max(net, axis=[1])
    net = self.fc(net)

    return net


if __name__ == '__main__':
  tf.enable_eager_execution()

  import numpy as np

  input_data = np.random.normal(size=[20, 6000, 1])
  model = AudioEncoderDecoder(10)

  print(input_data)
  print(input_data.shape)
  out = model(tf.constant(input_data, dtype=tf.float32)).numpy()
  print(out)
  print(out.shape)
