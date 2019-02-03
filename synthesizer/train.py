from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib import eager as tfe

from absl import app
from absl import flags
from absl import logging

import models
import trainer

from scipy.io.wavfile import write as write_wav

tf.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string("instrument", None,
                    "Instrument from which to get the files.")
flags.DEFINE_string("data_path", "data", "Data root path")

flags.DEFINE_integer("batch_size", 1, "Batch size to use.")
flags.DEFINE_integer("steps", None, "Number of steps.")


def main(argv):
  del argv

  MAIN_PATH = os.path.join(FLAGS.data_path, 'synthesis', FLAGS.instrument)
  ORIGINAL_PATH = os.path.join(MAIN_PATH, 'original.tfrecords')
  MIDI_PATH = os.path.join(MAIN_PATH, 'midi.tfrecords')

  def _parse_fn(example_proto):
    features = {
        'song': tf.FixedLenFeature((), tf.string, default_value=''),
        'sample_rate': tf.FixedLenFeature((), tf.int64, default_value=0)
    }

    parsed_features = tf.parse_single_example(example_proto, features)
    audio = parsed_features['song']
    sample_rate = parsed_features['sample_rate']

    audio = tf.decode_raw(audio, tf.int16)
    audio = (tf.to_float(audio) + 0.5) / 32767.5
    audio = tf.reshape(audio, [-1, 1])
    audio = tf.cast(audio, tf.float32)

    sample_rate = tf.cast(sample_rate, tf.float32)

    return sample_rate, audio

  original_ds = tf.data.TFRecordDataset(ORIGINAL_PATH)
  original_ds = original_ds.map(_parse_fn)
  # original_ds = original_ds.shuffle(100)
  original_ds = original_ds.batch(FLAGS.batch_size)
  original_ds = original_ds.repeat()

  midi_ds = tf.data.TFRecordDataset(MIDI_PATH)
  midi_ds = midi_ds.map(_parse_fn)
  # midi_ds = midi_ds.shuffle(100)
  midi_ds = midi_ds.batch(FLAGS.batch_size)
  midi_ds = midi_ds.repeat()

  midi2original = models.AudioEncoderDecoder()
  original2midi = models.AudioEncoderDecoder()

  midi_disc = models.AudioClassifier()
  original_disc = models.AudioClassifier()

  for step, element in enumerate(tf.data.Dataset.zip((original_ds, midi_ds))):
    original = element[0]
    midi = element[1]

    print(original[1].numpy())
    write_wav("/tmp/test.wav", int(original[0].numpy()), original[1].numpy()[0])
    break


if __name__ == '__main__':
  app.run(main)
