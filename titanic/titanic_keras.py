# Lint as: python3
r"""TFX Beam implementation to solve titanic problem.

Input: tfrecord file.
Output: servable model.

Please run this file after you have train.tfrecord under
/tmp/titanic/data/train/.
Example command to run:
python3 titanic_keras.py --project_root=/tmp/titanic/ \
--data_root=/tmp/titanic/data/train/
"""
import os
from typing import Text

from absl import app
from absl import flags
import tensorflow_model_analysis as tfma
from tfx.components import Evaluator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.example_gen.import_example_gen.component import ImportExampleGen
from tfx.components.trainer.executor import GenericExecutor
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import tfrecord_input
from custom_component import component as custom_component

FLAGS = flags.FLAGS

flags.DEFINE_string('pipeline_name', 'titanic', 'Name of pipeline')
flags.DEFINE_string('project_root', '~/titanic',
                    'Root directory of this project.')
flags.DEFINE_string(
    'data_root', '~/titanic/data/train',
    'Directory of training data. No test.tfrecord in the folder.')

def create_tfx_pipeline(pipeline_name: Text, pipeline_root: Text,
                        data_root: Text, module_file: Text,
                        serving_model_dir: Text, metadata_path: Text,
                        direct_num_workers: int) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""

  # Brings data into the pipeline or otherwise joins/converts training data.
  train_data_path = os.path.join(data_root, 'train')
  test_data_path = os.path.join(data_root, 'test')
  train_examples = tfrecord_input(train_data_path)
  train_example_gen = ImportExampleGen(input=train_examples, instance_name='train_example_gen')
  test_examples = tfrecord_input(test_data_path)
  test_example_gen = ImportExampleGen(input=test_examples, instance_name='test_example_gen')

  # Computes statistics over data for visualization and example validation.
  train_statistics_gen = StatisticsGen(examples=train_example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  train_infer_schema = SchemaGen(
      statistics=train_statistics_gen.outputs['statistics'],
      infer_feature_shape=False)

  train_transform = Transform(
      examples=train_example_gen.outputs['examples'],
      schema=train_infer_schema.outputs['schema'],
      module_file=module_file,
      instance_name='train_transformer')

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=module_file,
      # need to use custom executor spec
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      transformed_examples=train_transform.outputs['transformed_examples'],
      transform_graph=train_transform.outputs['transform_graph'],
      schema=train_infer_schema.outputs['schema'],
      train_args=trainer_pb2.TrainArgs(num_steps=20),
      eval_args=trainer_pb2.EvalArgs(num_steps=10))
  
  test_pred = custom_component.TestPredComponent(
      examples=test_example_gen.outputs['examples'],
      model=trainer.outputs['model']
  )

  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='Survived')],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  'BinaryAccuracy':
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.6}))
              })
      ])

  evaluator = Evaluator(
      examples=train_example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))
  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          train_example_gen, train_statistics_gen, train_infer_schema, train_transform, trainer,
          test_example_gen, test_pred,
          # evaluator, pusher
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=['--direct_num_workers=%d' % direct_num_workers])


def main(unused_argv):
  serving_model_dir = os.path.join(FLAGS.project_root, 'serving_model',
                                   FLAGS.pipeline_name)

  module_file = os.path.join(FLAGS.project_root, 'titanic_keras_utils.py')
  # Root directory to store pipeline artifacts.
  pipeline_root = os.path.join(FLAGS.project_root, 'pipeline')
  # Sqlite ML-metadata db path.
  metadata_path = os.path.join(pipeline_root, 'metadata', FLAGS.pipeline_name,
                               'metadata.db')

  BeamDagRunner().run(
      create_tfx_pipeline(
          pipeline_name=FLAGS.pipeline_name,
          pipeline_root=pipeline_root,
          data_root=FLAGS.data_root,
          module_file=module_file,
          serving_model_dir=serving_model_dir,
          metadata_path=metadata_path,
          # 0 means auto-detect based on on the number of CPUs available during
          # execution time.
          direct_num_workers=0))


if __name__ == '__main__':
  app.run(main)
