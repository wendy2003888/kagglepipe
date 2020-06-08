# Lint as: python3
r"""Convert csv raw data to features and write features to a tfrecord file.

Example command to run:
python3 data_pipeline.py --input_file=/tmp/titanic/test.csv \
--output_file=/tmp/titanic/data/test/test.tfrecord
"""

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', '', 'File path of the input raw data.')
flags.DEFINE_string('output_file', '', 'File path of the output tfrecord.')

INDEX_COL = ['PassengerId']

NUMERIC_COLS = ['Age', 'SibSp', 'Parch', 'Fare']

CATEGORICAL_COLS = ['Sex', 'Pclass', 'Embarked', 'NotAlone']

DROPED_COLS = ['Name', 'Ticket', 'Cabin']


def check_nan(data):
  return set([col for col in data.columns if data[col].isna().any()])


def replace_nan_to_unknown_str(data, col):
  data[col] = data[col].fillna('Unknown')
  return data


def normalize_numeric_data(data, mn, mx):
  # Center the data
  return (data - mn) / (mx - mn)


def fill_fare(data):
  for idx in data.index:
    if pd.notna(data.loc[idx]['Fare']):
      continue
    pclass = data.loc[idx]['Pclass']
    data.loc[idx, 'Fare'] = data.loc[data['Pclass'] == pclass]['Fare'].mean()
  return data


def generate_not_alone(data):
  tmp = data['SibSp'] + data['Parch']
  tmp[tmp != 0] = 1
  data['NotAlone'] = tmp
  return data


def serialize_one_row(x):
  """Serialize data to a binary-string using tf.train.Example.SerializeToString."""

  def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # if isinstance(value, type(pd.DataFrame)):
    #     value = value.numpy().astype('S')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

  def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  def _create_feature(value):
    if isinstance(value, str):
      value = value.encode('utf-8')
      return _bytes_feature(value)
    elif isinstance(value, float):
      return _float_feature(value)
    else:
      return _int64_feature(value)

  feature_dict = {col: _create_feature(x[col]) for col in x.index}
  return tf.train.Example(features=tf.train.Features(
      feature=feature_dict)).SerializeToString()


def data_pipeline(data, numeric_cols=None, cate_cols=None, drop_cols=None):
  """Process raw data to features used in a model. return pandas.DataFrame."""
  if drop_cols:
    data = data.drop(drop_cols, axis=1)
  age_mean = data['Age'].mean()
  media_age = data['Age'].quantile(q=0.50)
  data['Age'] = data['Age'].fillna(media_age)
  media_fare = data['Fare'].quantile(q=0.50)
  data['Fare'] = data['Fare'].fillna(media_fare)
  data = generate_not_alone(data)
  data['NumFamilyMember'] = data['SibSp'] + data['Parch']

  for col in cate_cols:
    if data[col].isna().any():
      data = replace_nan_to_unknown_str(data, col)
    data[col] = data[col].astype('str')

  print('nan columns', check_nan(data))
  return data


def main(unused_argv):
  # Read raw data.
  data_origin = pd.read_csv(FLAGS.input_file)

  # Do feature engineering. (e.g. drop data, fill missing data, normalize, etc.)
  data = data_pipeline(
      data_origin,
      numeric_cols=NUMERIC_COLS,
      cate_cols=CATEGORICAL_COLS,
      drop_cols=DROPED_COLS)
  # Put data to tf.Examples and write data to a tfrecord file.
  writer = tf.io.TFRecordWriter(FLAGS.output_file)
  for unused_idx, row in data.iterrows():
    serialized_features = serialize_one_row(row)
    writer.write(serialized_features)


if __name__ == '__main__':
  app.run(main)
