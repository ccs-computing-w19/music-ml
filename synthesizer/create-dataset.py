from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import wave

import numpy as np
import samplerate
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from scipy.io import wavfile


def read_wave_files(input_dir):
  out = []
  for filename in os.listdir(input_dir):
    filepath = os.path.join(input_dir, filename)
    song = wavfile.read(filepath)
    out.append(song)
  return out


def split(array, window_size, step_size, padding=True):
  """Split array into sub arrays of size "window_size" for every step of "step_size" """

  steps = int(len(array) / step_size)
  pad = (window_size - (len(array) - steps * step_size)) % window_size
  # TODO(Adam): Can do by padding less and decreasing number of steps.
  if padding:
    array = np.pad(array, (0, pad), mode='constant')
  else:
    raise NotImplementedError("padding must be True")

  logging.debug("Steps: {}, padding: {}".format(steps, pad))

  return [
      array[step * step_size:(step * step_size) + window_size, ...]
      for step in range(steps)
  ]


def resample(audio, sample_rate, new_sample_rate):
  ratio = new_sample_rate / sample_rate
  output = samplerate.resample(audio, ratio, 'sinc_best')
  return new_sample_rate, output


def to_chunks(audio, sample_rate, step_secs, window_secs):
  signals = split(audio, window_secs * sample_rate, step_secs * sample_rate)
  return [(sample_rate, signal) for signal in signals]


def process_audio(song):
  audio = song[1]
  sample_rate = song[0]
  logging.debug("Processing to mono")
  audio = np.mean(audio, axis=1)
  logging.debug("Trimming zeros")
  audio = np.trim_zeros(audio)

  logging.debug("Resampling")
  sample_rate, audio = resample(audio, sample_rate, 16000)
  logging.debug("Chunking")
  chunks = to_chunks(audio, sample_rate, 1, 7)

  return chunks


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_records(songs, out_file):
  with tf.python_io.TFRecordWriter(out_file) as writer:
    for i, song in enumerate(songs):
      if i % 30 == 0:
        logging.debug('Processesed: {}/{}'.format(i, len(songs)))

      chunks = process_audio(song)
      logging.debug('Processing chunks: {}'.format(len(chunks)))
      for sample_rate, song in chunks:
        feature = {
            'song': bytes_feature(tf.compat.as_bytes(song.tostring())),
            'sample_rate': int64_feature(sample_rate)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


DATA_PATH = "data/synthesis"
MIDI_GENERATED = "midi/wave"
ORIGINAL = "original/wave"

MODEL_INPUT_PATH = "preprocessed"


def main(argv):
  del argv
  for instrument in os.listdir(DATA_PATH):
    midi_dir = os.path.join(DATA_PATH, instrument, MIDI_GENERATED)
    original_dir = os.path.join(DATA_PATH, instrument, ORIGINAL)
    midi_songs = read_wave_files(midi_dir)
    original_songs = read_wave_files(original_dir)

    OUT_DIR = os.path.join(DATA_PATH, instrument)

    write_records(midi_songs, os.path.join(OUT_DIR, "midi.tfrecords"))
    write_records(original_songs, os.path.join(OUT_DIR, "original.tfrecords"))


if __name__ == '__main__':
  app.run(main)
