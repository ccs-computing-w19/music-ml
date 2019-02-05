from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Conv1DTranspose(tf.keras.layers.Conv2DTranspose):

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(Conv1DTranspose, self).__init__(
        filters, (kernel_size, 1), (strides, 1),
        data_format='channels_last',
        padding=padding,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)

  def build(self, input_shape):
    super(Conv1DTranspose, self).build(input_shape)

  def call(self, inputs):
    return tf.squeeze(super(Conv1DTranspose, self).call(inputs), -2)

  def __call__(self, inputs):
    return super(Conv1DTranspose, self).__call__(tf.expand_dims(inputs, -2))

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    # Shape should be (Width, 1, channels)
    input_shape.insert(-1, 1)
    input_shape = tf.TensorShape(input_shape)
    output_shape = super(Conv1DTranspose,
                         self).compute_output_shape(input_shape).as_list()
    output_shape.pop(-2)
    return tf.TensorShape(output_shape)


def pad_up_to(t, max_in_dims, constant_values=0):
  s = t.shape
  paddings = [[0, m - s[i]] for (i, m) in enumerate(max_in_dims)]
  return tf.pad(t, paddings, 'CONSTANT', constant_values=constant_values)


# TODO(adam): Should split audio in chunks for training more easily.
# TODO(adam): Maybe use spectrogram as input instead.
# TODO(adam): Pad input to get correct decoder step shape.
class AudioEncoderDecoder(tf.keras.models.Model):

  def __init__(self, name='AudioEncoderDecoder'):
    super(AudioEncoderDecoder, self).__init__(name=name)

    kernels = [10, 10, 5, 5, 2, 2]
    pool = [5, 5, 3, 3, 2, 2]
    filters = [5, 5, 10, 10, 20, 20]

    self.econvs = []
    self.epools = []
    self.enorms = []

    self.dconvs = []
    self.dnorms = []

    with tf.name_scope("encoder"):
      for i in range(len(kernels)):
        self.econvs.append(
            tf.keras.layers.Conv2D(
                filters=filters[i],
                kernel_size=kernels[i],
                padding='same',
                activation=tf.nn.leaky_relu))
        self.epools.append(
            tf.keras.layers.AveragePooling2D(pool_size=pool[i], padding='same'))
        self.enorms.append(tf.keras.layers.BatchNormalization())

    with tf.name_scope("decoder"):
      for i in range(len(kernels)):
        i = len(kernels) - i - 1
        self.dconvs.append(
            tf.keras.layers.Conv2DTranspose(
                filters=filters[i],
                kernel_size=kernels[i],
                strides=pool[i],
                padding='same',
                activation=tf.nn.leaky_relu))
        self.dnorms.append(tf.keras.layers.BatchNormalization())

    with tf.name_scope('output'):

      self.out = tf.keras.layers.Conv2D(
          filters=2, kernel_size=10, padding='same', activation=None)

  def call(self, inputs):
    encoded = []
    net = inputs
    net = tf.squeeze(net, axis=-1)
    print(net.shape)

    net = tf.contrib.signal.stft(net, frame_length=1024, frame_step=1024)
    net = tf.stack([tf.real(net), tf.imag(net)], axis=-1)

    for i in range(len(self.econvs)):
      net = self.econvs[i](net)
      encoded.append(net)
      net = self.epools[i](net)
      net = self.enorms[i](net)

    for i in range(len(self.dconvs)):
      net = self.dconvs[i](net)
      net = self.dnorms[i](net)
      reshaped_encoded = encoded[-1]
      reshaped_encoded = pad_up_to(reshaped_encoded, net.shape)
      net = tf.concat([net, reshaped_encoded], axis=-1)
      encoded.pop()

    net = self.out(net)
    net = tf.contrib.signal.inverse_stft(
        tf.complex(net[..., 0], net[..., 1]),
        frame_length=1024,
        frame_step=1024)

    return net


# TODO(adam): Should split audio in chunks for training more easily.
# TODO(adam): Maybe use spectrogram as input instead.
class AudioClassifier(tf.keras.models.Model):

  def __init__(self, name='AudioClassifier'):
    super(AudioClassifier, self).__init__(name=name)

    self.conv1 = tf.keras.layers.Conv2D(
        filters=3, kernel_size=10, padding='same', activation=tf.nn.leaky_relu)
    self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=10, padding='same')
    self.norm1 = tf.keras.layers.BatchNormalization()

    self.conv2 = tf.keras.layers.Conv2D(
        filters=5, kernel_size=8, padding='same', activation=tf.nn.leaky_relu)
    self.pool2 = tf.keras.layers.AveragePooling2D(pool_size=10, padding='same')
    self.norm2 = tf.keras.layers.BatchNormalization()

    self.conv3 = tf.keras.layers.Conv2D(
        filters=10, kernel_size=5, padding='same', activation=tf.nn.leaky_relu)
    self.pool3 = tf.keras.layers.AveragePooling2D(pool_size=10, padding='same')
    self.norm3 = tf.keras.layers.BatchNormalization()

    with tf.name_scope("output"):
      self.fc = tf.keras.layers.Dense(units=1)

  def call(self, inputs):
    net = inputs
    net = tf.squeeze(net, axis=-1)

    net = tf.contrib.signal.stft(net, frame_length=1024, frame_step=1024)
    net = tf.stack([tf.real(net), tf.imag(net)], axis=-1)

    net = self.conv1(net)
    net = self.pool1(net)
    net = self.norm1(net)
    net = self.conv2(net)
    net = self.pool2(net)
    net = self.norm2(net)
    net = self.conv3(net)
    net = self.pool3(net)
    net = self.norm3(net)

    net = tf.reduce_max(net, axis=[1, 2])
    net = self.fc(net)

    return net


if __name__ == '__main__':
  tf.enable_eager_execution()

  import numpy as np

  input_data = np.random.normal(size=[20, 6000, 1])
  model = AudioEncoderDecoder()

  print(input_data)
  print(input_data.shape)
  out = model(tf.constant(input_data, dtype=tf.float32)).numpy()
  print(out)
  print(out.shape)
