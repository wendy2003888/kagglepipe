# Lint as: python3

import os
from typing import Any, Dict, List, Text
import absl
import apache_beam as beam
import tensorflow as tf
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils

class RunModel(beam.DoFn):
  def __init__(self, model_path, signature_name, id_label):
    self.model_path = model_path
    self.signature_name = signature_name
    self.tags = [tf.saved_model.SERVING] # == 'serve' => tensorflow/python/saved_model/tag_constants.py
    self.model = None
    self.id_label = id_label
    
  def setup(self):
    self.model = tf.keras.models.load_model(self.model_path) 
    
  def process(self, elements):
    examples = tf.constant([elements.SerializeToString()])
    outputs = self.model.signatures[self.signature_name](examples=examples)
    return [{'id': elements.features.feature[self.id_label], 'pred': outputs}]

# Not running
class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    self._log_startup(input_dict, output_dict, exec_properties)

    example_uris = {}
    for example in input_dict['examples']:
      for split in artifact_utils.decode_split_names(example.split_names):
        example_uris[split] = os.path.join(example.uri, split)
               
    model = artifact_utils.get_single_instance(input_dict['model'])
    model_path = path_utils.serving_model_path(model.uri)
    absl.logging.info('Using {} as current model.'.format(model_path))
    
    output_uri = os.path.join(
        artifact_utils.get_single_uri(output_dict['output_data']), 'pred.csv')
    with self._make_beam_pipeline() as pipeline:
      test_data = []
      for split, example_uri in example_uris.items():
        test_data.append(pipeline | 'ReadFromTFRecord_{}'.format(split) >> beam.io.ReadFromTFRecord(file_pattern=io_utils.all_files_pattern(example_uri)))
      
      (test_data | 'Flattern' >> beam.Flatten()
      | 'ParseToExample' >> beam.Map(tf.train.Example.FromString)
      | 'Prediction' >> beam.ParDo(RunModel(model_path, 'serving_default', 'PassengerId'))
      | 'ParseToKVPair' >> beam.Map(lambda x: ParseResultToKV(x))
      | 'AddSameKey' >> beam.Map(lambda x: (1, x))
      | 'Window' >> beam.WindowInto(beam.window.GlobalWindows())
      | 'GroupByKey' >> beam.GroupByKey()
      | 'Sort' >> beam.Map(lambda group_data :  sorted(group_data[1],key=lambda x: x[0]))
      | 'Flatten' >> beam.FlatMap(lambda x : x)
      | 'ToStr' >> beam.Map(lambda x : '{},{}'.format(x[0], '0' if x[1] < 0.5 else '1'))
      | 'WriteToFile' >> beam.io.WriteToText(output_uri, num_shards=1, shard_name_template='', header='PassengerId,Survived'))
    absl.logging.info('TestPredComponent result written to %s.', output_uri)

def ParseResultToKV(outputs):
  # res = {}
  # if len(outputs['id'].int64_list.value):
  #   res['id'] = list(outputs['id'].int64_list.value)
  # # 'output_0' is the outputs key in SignatureDef
  # res['pred'] = outputs['pred']['output_0'].numpy()
  id_val = None
  if len(outputs['id'].int64_list.value):
    id_val = outputs['id'].int64_list.value[0]
  pred = outputs['pred']['output_0'].numpy().ravel()
  label = 0 if pred[0] < 0.5 else 1
  return (id_val, label)