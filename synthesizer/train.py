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

flags.DEFINE_float('cycle_weight', 10.0, 'Lambda value for the cycle loss.')
flags.DEFINE_float('dlr', 0.001, "Discriminator Learning rate")
flags.DEFINE_float('glr', 0.001, "Generator Learning rate")


def main(argv):
  del argv

  logdir = "./tb/"
  writer = tf.contrib.summary.create_file_writer(logdir)
  writer.set_as_default()

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

    audio = tf.decode_raw(audio, tf.float64)
    audio = (tf.to_float(audio) + 0.5) / 32767.5
    audio = tf.reshape(audio, [-1, 1])
    audio = tf.cast(audio, tf.float32)

    #     # Pad to multiple of pools
    # shape = tf.shape(audio)
    # div = 900
    # mod = tf.mod(shape[0], div)
    # extra = tf.mod(div - mod, div)
    #     pad = tf.zeros(dtype=tf.float32, shape=[extra, shape[-1]])
    # audio = tf.concat([audio, pad], axis=0)

    sample_rate = tf.cast(sample_rate, tf.float32)

    return sample_rate, audio

  original_ds = tf.data.TFRecordDataset(ORIGINAL_PATH)
  original_ds = original_ds.map(_parse_fn)
  # original_ds = original_ds.shuffle(100)
  original_ds = original_ds.padded_batch(
      FLAGS.batch_size, padded_shapes=([], [None, 1]))
  original_ds = original_ds.repeat()

  midi_ds = tf.data.TFRecordDataset(MIDI_PATH)
  midi_ds = midi_ds.map(_parse_fn)
  # midi_ds = midi_ds.shuffle(100)
  midi_ds = midi_ds.padded_batch(
      FLAGS.batch_size, padded_shapes=([], [None, 1]))
  midi_ds = midi_ds.repeat()

  midi2original = models.AudioEncoderDecoder()
  original2midi = models.AudioEncoderDecoder()

  midi_disc = models.AudioClassifier()
  original_disc = models.AudioClassifier()

  g_opt = tf.train.AdamOptimizer(FLAGS.glr)
  d_opt = tf.train.AdamOptimizer(FLAGS.dlr)

  for step, element in enumerate(tf.data.Dataset.zip((original_ds, midi_ds))):
    original_sr, original = element[0]
    midi_sr, midi = element[1]

    with tf.device('/cpu:0'):
      with tf.GradientTape(persistent=True) as tape:
        gen_original = midi2original(midi)
        gen_midi = original2midi(original)

        real_midi_logits = midi_disc(midi)
        fake_midi_logits = midi_disc(gen_midi)

        real_original_logits = original_disc(original)
        fake_original_logits = original_disc(gen_original)

        g_midi_loss = tf.reduce_mean(tf.square(fake_midi_logits))
        g_original_loss = tf.reduce_mean(tf.square(fake_original_logits))

        d_midi_loss = tf.losses.mean_squared_error(
            tf.ones_like(real_midi_logits),
            real_midi_logits) + tf.losses.mean_squared_error(
                tf.zeros_like(fake_midi_logits), fake_midi_logits)

        d_original_loss = tf.losses.mean_squared_error(
            tf.ones_like(real_original_logits),
            real_original_logits) + tf.losses.mean_squared_error(
                tf.zeros_like(fake_original_logits), fake_original_logits)

        cycle_midi = original2midi(gen_original)
        cycle_original = midi2original(gen_midi)

        cycle_midi_loss = tf.reduce_mean(tf.abs(cycle_midi - midi))
        cycle_original_loss = tf.reduce_mean(tf.abs(cycle_original - original))

        g_midi_loss = g_midi_loss + FLAGS.cycle_weight * (
            cycle_midi_loss + cycle_original_loss)
        g_original_loss = g_original_loss + FLAGS.cycle_weight * (
            cycle_midi_loss + cycle_original_loss)

      update_ops = midi2original.updates + original2midi.updates + midi_disc.updates + original_disc.updates
      with tf.control_dependencies(update_ops):
        g_opt.apply_gradients(
            zip(
                tape.gradient(g_midi_loss, midi2original.trainable_variables),
                midi2original.trainable_variables))
        g_opt.apply_gradients(
            zip(
                tape.gradient(g_original_loss,
                              original2midi.trainable_variables),
                original2midi.trainable_variables))
        d_opt.apply_gradients(
            zip(
                tape.gradient(d_midi_loss, midi_disc.trainable_variables),
                midi_disc.trainable_variables))
        d_opt.apply_gradients(
            zip(
                tape.gradient(d_original_loss,
                              original_disc.trainable_variables),
                original_disc.trainable_variables))
      del tape

    if step % 1 == 0:
      logging.info(
          "Step: {} \t Midi Loss: {:.3E} \t Original Loss: {:.3E} \t MidiDiscLoss: {:.3E} \t OriginalDiscLoss: {:.3E}"
          .format(step, g_midi_loss, g_original_loss, d_midi_loss,
                  d_original_loss))
    if step % 100 == 0:

      saver = tfe.Saver(midi2original.variables + original2midi.variables +
                        midi_disc.variables + original_disc.variables)
      saver.save('checkpoints/checkpoint.ckpt', global_step=step)

    with tf.contrib.summary.record_summaries_every_n_global_steps(100):
      tf.contrib.summary.scalar('d_midi_loss', d_midi_loss)
      tf.contrib.summary.scalar('midi2original_loss', g_midi_loss)
      tf.contrib.summary.scalar('d_original_loss', d_original_loss)
      tf.contrib.summary.scalar('g_original2midi_loss', g_original_loss)

      tf.contrib.summary.audio(
          'midi_song', tf.expand_dims(midi[0], 0), midi_sr[0], max_outputs=1)
      tf.contrib.summary.audio(
          'original_song',
          tf.expand_dims(original[0], 0),
          original_sr[0],
          max_outputs=1)

      tf.contrib.summary.audio(
          'gen_midi',
          tf.expand_dims(gen_midi[0], 0),
          original_sr[0],
          max_outputs=1)
      tf.contrib.summary.audio(
          'gen_original',
          tf.expand_dims(gen_original[0], 0),
          midi_sr[0],
          max_outputs=1)

      tf.contrib.summary.audio(
          'cycle_midi',
          tf.expand_dims(cycle_midi[0], 0),
          midi_sr[0],
          max_outputs=1)
      tf.contrib.summary.audio(
          'cycle_original',
          tf.expand_dims(cycle_original[0], 0),
          original_sr[0],
          max_outputs=1)


if __name__ == '__main__':
  app.run(main)
