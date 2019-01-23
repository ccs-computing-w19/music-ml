from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import models
import trainer

FLAGS = flags.FLAGS

flags.DEFINE_string("instrument", None,
                    "Instrument from which to get the files.")
flags.DEFINE_string("data_path", "data", "Data root path")

flags.DEFINE_integer("batch_size", 1, "Batch size to use.")


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
    audio = tf.reshape(audio, [-1, 1])
    audio = tf.cast(audio, tf.float32)

    sample_rate = tf.cast(sample_rate, tf.float32)

    return sample_rate, audio

  original_ds = tf.data.TFRecordDataset(ORIGINAL_PATH)
  original_ds = original_ds.map(_parse_fn)
  original_ds = original_ds.shuffle(100)
  original_ds = original_ds.batch(FLAGS.batch_size)
  original_iterator = original_ds.make_one_shot_iterator()
  original_sr, original_song = original_iterator.get_next()

  midi_ds = tf.data.TFRecordDataset(MIDI_PATH)
  midi_ds = midi_ds.map(_parse_fn)
  midi_ds = midi_ds.shuffle(100)
  midi_ds = midi_ds.batch(FLAGS.batch_size)
  midi_iterator = midi_ds.make_one_shot_iterator()
  midi_sr, midi_song = midi_iterator.get_next()

  midi2original = models.AudioEncoderDecoder()
  original2midi = models.AudioEncoderDecoder()

  midi_disc = models.AudioClassifier()
  original_disc = models.AudioClassifier()

  #TODO (adam): Finish this

  # def step_fn(session):

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    orig_song, orig_sr = sess.run([original_song, original_sr])
    midi_song, midi_sr = sess.run([midi_song, midi_sr])

    print(orig_song)
    print(midi_song)


if __name__ == '__main__':
  app.run(main)
