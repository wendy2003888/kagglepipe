from absl import app
from absl import flags
from typing import Text

import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, Trainer, Transform, Evaluator, Pusher, ResolverNode, BulkInferrer
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.proto import bulk_inferrer_pb2
from tfx.utils.dsl_utils import external_input
from tfx.orchestration import pipeline
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from hello_component import component
from tfx.orchestration.kubeflow import kubeflow_dag_runner

FLAGS = flags.FLAGS

def generate_pipeline(pipeline_name, pipeline_root, train_data, test_data, train_steps, eval_steps, pusher_target, runner):
  module_file = 'util.py' # util.py is a file in the same folder

  # RuntimeParameter is only supported on KubeflowDagRunner currently
  if runner == 'kubeflow':
    pipeline_root_param = os.path.join('gs://{{kfp-default-bucket}}', pipeline_name, '{{workflow.uid}}')
    train_data_param = data_types.RuntimeParameter(name='train-data', default='gs://renming-mlpipeline-kubeflowpipelines-default/kaggle/santander/train', ptype=Text)
    test_data_param = data_types.RuntimeParameter(name='test-data', default='gs://renming-mlpipeline-kubeflowpipelines-default/kaggle/santander/test', ptype=Text)
    pusher_target_param = data_types.RuntimeParameter(name='pusher-destination', default='gs://renming-mlpipeline-kubeflowpipelines-default/kaggle/santander/serving', ptype=Text)
  else:
    pipeline_root_param = pipeline_root
    train_data_param = train_data
    test_data_param = test_data
    pusher_target_param = pusher_target

  examples = external_input(train_data_param)
  example_gen = CsvExampleGen(input=examples, instance_name="train")

  test_examples = external_input(test_data_param)
  test_example_gen = CsvExampleGen(input=test_examples, output_config={'split_config': {'splits': [{'name':'test', 'hash_buckets':1}]}}, instance_name="test")

  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True) # infer_feature_shape controls sparse or dense

  # Transform is too slow in my side.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  trainer = Trainer(
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file,
      train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
      eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps),
      instance_name="train",
      enable_cache=False)

  # Get the latest blessed model for model validation.
  model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='target')],
      # tfma.SlicingSpec(feature_keys=['var_0', 'var_1']) when add more, Evaluator can't ouptput BLESSED status. It should be a bug in TFMA.
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  'binary_accuracy':
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.4}),
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ])
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      # baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config,
      instance_name="eval5")

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination={'filesystem': {
          'base_directory': pusher_target_param}})

  bulk_inferrer = BulkInferrer(
      examples=test_example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      # model_blessing=evaluator.outputs['blessing'],
      data_spec=bulk_inferrer_pb2.DataSpec(),
      model_spec=bulk_inferrer_pb2.ModelSpec(),
      instance_name="bulkInferrer"
      )

  hello = component.HelloComponent(
      input_data=bulk_inferrer.outputs['inference_result'], instance_name='csvGen')

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root_param,
      components=[
          example_gen, statistics_gen, schema_gen, transform, trainer,
          model_resolver, evaluator, pusher, hello, test_example_gen, bulk_inferrer
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          os.path.join(pipeline_root, 'metadata.sqlite')),
      beam_pipeline_args=['--direct_num_workers=0'])

def main(_):
  pipeline = generate_pipeline(
      flags.FLAGS.pipeline_name,
      flags.FLAGS.pipeline_root,
      flags.FLAGS.train_data,
      flags.FLAGS.test_data,
      flags.FLAGS.train_steps,
      flags.FLAGS.eval_steps,
      flags.FLAGS.pusher_target,
      flags.FLAGS.runner)

  if flags.FLAGS.runner == 'local':
    BeamDagRunner().run(pipeline)
  #elif flags.FLAGS.runner == 'flink':
    # need to slightly change TFX codes to support other Beam-runners
    # BeamDagRunner(pipelineOptions).run(pipeline)
  elif flags.FLAGS.runner == 'kubeflow':
    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    tfx_image = os.environ.get('KUBEFLOW_TFX_IMAGE', None)
    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        tfx_image=tfx_image)
    kubeflow_dag_runner.KubeflowDagRunner(config=runner_config).run(
        pipeline)
  else:
    exit(1)

if __name__ == '__main__':
  flags.DEFINE_string(
      name="pipeline_name", default="santander",
      help="pipeline name used to identity different pipelines")
  flags.DEFINE_string(
      name="pipeline_root", default="/var/tmp/santander/keras-tft/",
      help="pipeline root for storing artifacts, it's not used in KFP runner")
  flags.DEFINE_string(
      name="train_data", default="/var/tmp/santander/data/train",
      help="Folder for Kaggle train.csv. No test.csv in the folder, it's not used in KFP runner")
  flags.DEFINE_string(
      name="test_data", default="/var/tmp/santander/data/test",
      help="Folder for Kaggle test.csv. No train.csv in the folder, it's not used in KFP runner")
  flags.DEFINE_integer(
      name="train_steps", default=10000,
      help="Steps to train a model")
  flags.DEFINE_integer(
      name="eval_steps", default=1000,
      help="Steps to train a model")
  flags.DEFINE_string(
      name="pusher_target", default="/var/tmp/santander/pusher",
      help="Pusher can't create this folder for you, it's not used in KFP runner")
  flags.DEFINE_enum(
      name="runner", default="kubeflow",
      enum_values=['local', 'kubeflow'],
      help="Pusher can't create this folder for you")

  app.run(main)
