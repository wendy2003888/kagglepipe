# Lint as: python3
"""The utils/functions user defined to process data and train the model."""
import os

import absl
import tensorflow as tf
import tensorflow_transform as tft
from kerastuner.tuners import RandomSearch
from kerastuner import HyperModel


NUMERIC_FEATURE_KEYS = [
    'Age',
    'SibSp',
    'Parch',
    'Fare',
]

CATEGORICAL_FEATURE_KEYS = [
    'Pclass',
    'Sex',
    'Embarked',
    'NotAlone',
]
# Categorical features are assumed to each have a maximum value in the dataset.
CATEGORICAL_FEATURE_BUCKETS = {
    'Pclass': 3,
    'Sex': 2,
    'Embarked': 4,
    'NotAlone': 2
}

LABEL_KEY = 'Survived'


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def sparse_2_dense(x):
  default_value = '' if x.dtype == tf.string else -1
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def pack_numeric_features(features, numeric_cols):
  numeric_features = [features.pop(col) for col in numeric_cols]
  numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
  numeric_features = tf.stack(numeric_features, axis=-1)
  features['numeric'] = numeric_features
  return features


def preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns."""
  outputs = {}
  for key in NUMERIC_FEATURE_KEYS:
    outputs[key] = sparse_2_dense(inputs[key])

  for key in CATEGORICAL_FEATURE_KEYS:
    outputs[key] = sparse_2_dense(inputs[key])

  outputs[LABEL_KEY] = sparse_2_dense(inputs[LABEL_KEY])

  return outputs


def input_fn(filenames, tf_transform_output, batch_size=200):
  """Generates features and labels for training or evaluation.

  Args:
    filenames: [str] list of CSV files to read data from.
    tf_transform_output: A TFTransformOutput.
    batch_size: int First dimension size of the Tensors returned by input_fn.

  Returns:
    A (features, indices) tuple where features is a dictionary of
      Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      filenames,
      batch_size,
      transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=LABEL_KEY,
      shuffle=True)
  return dataset

class KerasModel(HyperModel):

    def __init__(self, feature_columns):
        self.feature_columns = feature_columns

    def build(self, hp):
        feature_input_layers = {
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
            for colname in NUMERIC_FEATURE_KEYS
        }
        feature_input_layers.update({
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.string)
            for colname in CATEGORICAL_FEATURE_KEYS
        })
        # Makes Model Reader Happy.
        # DenseFeatures V2 doesn't have partitioner right now. down grade to v1 api.
        # https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/feature_column/dense_features_v2.py#L39
        inputs = tf.keras.layers.DenseFeatures(self.feature_columns, partitioner=partitioner)(feature_input_layers)
        fc1 = tf.keras.layers.Dense(units=hp.Int('units',
                                            min_value=32,
                                            max_value=128,
                                            step=32), activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(32, activation='relu')(fc1)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(fc2)
        flatten_outputs = tf.keras.layers.Reshape((-1,))(outputs)
        model = tf.keras.Model(inputs=feature_input_layers, outputs=flatten_outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                  hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['BinaryAccuracy'])
        absl.logging.info(model.summary())
        return model



def get_serving_receiver_fn(model, tf_transform_output):
  """Identifies and exports the saved model for serving(also used in evaluator)."""
  # This layer is needed as an attribute to the model in order to make sure that
  # the model assets are handled correctly when exporting
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serving_input_receiver_fn(serialized_tf_examples):

    # run_fn doesn't have schema, so we use tft.output.raw_feature_spec
    raw_feature_spec = tf_transform_output.raw_feature_spec()
    raw_feature_spec.pop(LABEL_KEY)

    parsed_features = tf.io.parse_example(serialized_tf_examples,
                                          raw_feature_spec)
    transformed_features = tf_transform_output.transform_raw_features(
        parsed_features)
    # Remember also pop label from features.
    transformed_features.pop(LABEL_KEY)
    outputs = model(transformed_features)
    return outputs

  return serving_input_receiver_fn


# GenericExecutor is using run_fn rather than trainer_fn
def run_fn(fn_args):
  """Build the estimator using the high level API.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.

  Returns:
    A dict of the following:
      - estimator: The estimator that will be used for training and eval.
      - train_spec: Spec for training.
      - eval_spec: Spec for eval.
      - eval_input_receiver_fn: Input function for eval.
  """

  train_batch_size = 100
  eval_batch_size = 100

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
  # numeric_columns =  [tf.feature_column.numeric_column('packed_numeric')]
  numeric_columns = [
      tf.feature_column.numeric_column(key) for key in NUMERIC_FEATURE_KEYS
  ]
  categorical_columns = [
      tf.feature_column.indicator_column(  # pylint: disable=g-complex-comprehension
          tf.feature_column.categorical_column_with_hash_bucket(
              key, hash_bucket_size=CATEGORICAL_FEATURE_BUCKETS[key]))
      for key in CATEGORICAL_FEATURE_KEYS
  ]

  train_data = input_fn(  # pylint: disable=g-long-lambda
      fn_args.train_files,
      tf_transform_output,
      batch_size=train_batch_size)

  eval_data = input_fn(  # pylint: disable=g-long-lambda
      fn_args.eval_files,
      tf_transform_output,
      batch_size=eval_batch_size)

  feature_columns = numeric_columns + categorical_columns

  
  model = KerasModel(feature_columns)
  
  tuner = RandomSearch(
      model,
      objective='val_BinaryAccuracy',
      max_trials=10,
      # Separate tunner files with model files, so that pusher can work properly when version <= 0.21.4.
      directory=os.path.dirname(fn_args.serving_model_dir),
      project_name='keras_tuner')

  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')
  # When passing an infinitely repeating dataset, you must specify the `steps_per_epoch` argument.
  tuner.search(train_data,
               epochs=10,
               steps_per_epoch=20,
               validation_steps=fn_args.eval_steps,
               validation_data=eval_data,
               callbacks=[tensorboard_callback])
  tuner.search_space_summary()
  best_model = tuner.get_best_models(1)[0]
  signatures = {
      'serving_default':
          get_serving_receiver_fn(best_model,
                                  tf_transform_output).get_concrete_function(
                                      tf.TensorSpec(
                                          shape=[None],
                                          dtype=tf.string,
                                          name='examples'))
  }
  # More about signatures:
  # https://www.tensorflow.org/api_docs/python/tf/saved_model/save?hl=en
  best_model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
