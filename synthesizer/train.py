from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from scipy.io.wavfile import write as write_wav
from tensorflow.contrib import eager as tfe

import models
import trainer

tf.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string("instrument", None,
                    "Instrument from which to get the files.")
flags.DEFINE_string("data_path", "data", "Data root path")
flags.DEFINE_string("exp_name", None, "Experiment name to save checkpoints and summaries")

flags.DEFINE_bool("restore", True, "If a checkpoint exists and this flag is set to true, then restore parameters from checkpoint.")

flags.DEFINE_integer("batch_size", 1, "Batch size to use.")
flags.DEFINE_integer("steps", None, "Number of steps.")

flags.DEFINE_float('cycle_weight', 10.0, 'Lambda value for the cycle loss.')
flags.DEFINE_float('dlr', 0.001, "Discriminator Learning rate")
flags.DEFINE_float('glr', 0.001, "Generator Learning rate")


def main(argv):
  del argv

  logdir = "gs://music-generation-bucket/experiments/{}/tb".format(FLAGS.exp_name)
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

    audio = tf.decode_raw(audio, tf.float32)
    audio = (tf.to_float(audio) + 0.5) / 32767.5
    audio = tf.reshape(audio, [-1, 1])
    sample_rate = tf.cast(sample_rate, tf.float32)

    return sample_rate, audio

  original_ds = tf.data.TFRecordDataset(ORIGINAL_PATH)
  original_ds = original_ds.map(_parse_fn)
  original_ds = original_ds.shuffle(100)
  original_ds = original_ds.batch(FLAGS.batch_size)
  original_ds = original_ds.repeat()

  midi_ds = tf.data.TFRecordDataset(MIDI_PATH)
  midi_ds = midi_ds.map(_parse_fn)
  midi_ds = midi_ds.shuffle(100)
  midi_ds = midi_ds.batch(FLAGS.batch_size)
  midi_ds = midi_ds.repeat()

  strategy = tf.contrib.distribute.MirroredStrategy()

  with strategy.scope():
    midi2original = models.AudioEncoderDecoder(5)
    original2midi = models.AudioEncoderDecoder(5)

    midi_disc = models.AudioClassifier()
    original_disc = models.AudioClassifier()

    midi2original.build((None, None, 1))
    original2midi.build((None, None, 1))
    midi_disc.build((None, None, 1))
    original_disc.build((None, None, 1))

  g_opt = tf.train.AdamOptimizer(FLAGS.glr)
  d_opt = tf.train.AdamOptimizer(FLAGS.dlr)

  step = tf.train.get_or_create_global_step()

  checkpoint_dir = 'gs://music-generation-bucket/experiments/{}/checkpoints'.format(FLAGS.exp_name)
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  if FLAGS.restore and latest_checkpoint:
    print (midi2original.variables)
    print (midi2original.weights)
    logging.info("Restoring from {}".format(latest_checkpoint))
    saver = tfe.Saver(midi2original.variables + original2midi.variables +
                      midi_disc.variables + original_disc.variables)
    saver.restore(latest_checkpoint)

  print (step.numpy())
  exit()

  for _, element in enumerate(tf.data.Dataset.zip((original_ds, midi_ds))):
    original_sr, original = element[0]
    midi_sr, midi = element[1]

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
          tf.sigmoid(real_midi_logits)) + tf.losses.mean_squared_error(
              tf.zeros_like(fake_midi_logits), tf.sigmoid(fake_midi_logits))

      d_original_loss = tf.losses.mean_squared_error(
          tf.ones_like(real_original_logits),
          tf.sigmoid(real_original_logits)) + tf.losses.mean_squared_error(
              tf.zeros_like(fake_original_logits),
              tf.sigmoid(fake_original_logits))

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
              tape.gradient(g_original_loss, original2midi.trainable_variables),
              original2midi.trainable_variables))
      d_opt.apply_gradients(
          zip(
              tape.gradient(d_midi_loss, midi_disc.trainable_variables),
              midi_disc.trainable_variables))
      d_opt.apply_gradients(
          zip(
              tape.gradient(d_original_loss, original_disc.trainable_variables),
              original_disc.trainable_variables))
    del tape

    if step.numpy() % 10 == 0:
      logging.debug(
          "Step: {} \t Midi Loss: {:.3E} \t Original Loss: {:.3E} \t MidiDiscLoss: {:.3E} \t OriginalDiscLoss: {:.3E}"
          .format(step.numpy(), g_midi_loss, g_original_loss, d_midi_loss,
                  d_original_loss))

    if step.numpy() % 100 == 0:
      checkpoint_prefix = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
      saver = tfe.Saver(midi2original.variables + original2midi.variables +
                        midi_disc.variables + original_disc.variables)
      saver.save(checkpoint_prefix, global_step=step)
    with tf.contrib.summary.record_summaries_every_n_global_steps(50):
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
    step.assign_add(1)


if __name__ == '__main__':
  flags.mark_flags_as_required(['exp_name', 'instrument'])
  app.run(main)
