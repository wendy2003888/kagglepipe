import absl
import tensorflow as tf
from tensorflow import keras
from typing import Text
import tensorflow_transform as tft
import os
from kerastuner.tuners import RandomSearch

def gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')

def input_fn(file_pattern: Text,
             tf_transform_output: tft.TFTransformOutput,
             batch_size: int = 200) -> tf.data.Dataset:

  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=gzip_reader_fn,
      label_key='target_tft')

  return dataset

def raw_column_name(idx):
  return 'var_{0}'.format(idx)

def tft_column_name(idx):
  return 'var_{0}_tft'.format(idx)

def build_keras_model(hp) -> tf.keras.Model:
  inputs = [
      keras.layers.Input(shape=(1,), name=tft_column_name(key))
      for key in range(0, 200)
  ]
  # feature crossing in TFT
  inputs.append(keras.layers.Input(shape=(1,), name='fc1'))
  # feature crossing in model
  fc2 = inputs[2] * inputs[3]

  all = inputs + [fc2]
  d = keras.layers.concatenate(all)
  if hp != None:
    d = keras.layers.Dense(hp.Int('n_1', min_value=32, max_value=512, step=64),
                           activation='relu')(d)
    d = keras.layers.Dense(hp.Int('n_2', min_value=8, max_value=64, step=16),
                           activation='relu')(d)
  else:
    d = keras.layers.Dense(50, activation='relu')(d)
    d = keras.layers.Dense(25, activation='relu')(d)    
  outputs = keras.layers.Dense(1, activation='sigmoid')(d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  lr = 1e-2 if hp == None else hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
  model.compile(
      optimizer=keras.optimizers.Adam(lr=lr),
      loss='binary_crossentropy',
      metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy")])

  # model.summary(print_fn=absl.logging.info)
  return model

def get_serve_tf_examples_fn(model, tf_transform_output):

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop('target', None)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    # or don't set tft_layer but
    # transformed_features = tf_transform_output.transform_raw_features(parsed_features)
    transformed_features = model.tft_layer(parsed_features)
    # can be removed after TFT 0.22.2
    transformed_features.pop('target_tft', None)

    return model(transformed_features)

  return serve_tf_examples_fn

# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """

  outputs = {}
  for key in range(0, 200):
    feature_key = raw_column_name(key)
    outputs[tft_column_name(key)] = inputs[feature_key] # tft.scale_to_z_score(inputs[feature_key])
  outputs['target_tft'] = inputs['target']
  outputs['fc1'] = inputs['var_0'] * inputs['var_1']
  return outputs

# TFX Trainer will call this function.
def run_fn(fn_args):

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = input_fn(fn_args.train_files, tf_transform_output, batch_size=100)
  eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, batch_size=100)

  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')

  if True:
    print("Use normal Keras model")
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model = build_keras_model(None)
    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback])
  else:
    print("Use normal Keras Tuner")
    tuner = RandomSearch(
        build_keras_model,
        objective='val_binary_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory=fn_args.serving_model_dir,
        project_name='tuner')
    tuner.search(
        train_dataset,
        epochs=1,
        steps_per_epoch=fn_args.train_steps, # or few steps to get best HP and then well fit
        validation_steps=fn_args.eval_steps,
        validation_data=eval_dataset,
        callbacks=[tensorboard_callback, tf.keras.callbacks.EarlyStopping()])
    tuner.search_space_summary()
    tuner.results_summary()
    best_hparams = tuner.oracle.get_best_trials(1)[0].hyperparameters.get_config()
    model = tuner.get_best_models(1)[0]

  signatures = {
      'serving_default': get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
          tf.TensorSpec(shape=[None],
                        dtype=tf.string,
                        name='input_example_tensor')),
  }

  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
