import json
import os
from typing import Any, Dict, List, Text
import absl

import tensorflow as tf

from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils


class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    self._log_startup(input_dict, output_dict, exec_properties)

    absl.logging.info('Hello Component - Executor - Do Start')

    split_to_instance = {}
    for artifact in input_dict['input_data']:
      for split in json.loads(artifact.split_names):
        uri = os.path.join(artifact.uri, split)
        split_to_instance[split] = uri

    for split, instance in split_to_instance.items():
      input_dir = instance
      output_dir = artifact_utils.get_split_uri(
          output_dict['output_data'], split)
      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)
        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)

    absl.logging.info('Hello Component - Executor - Do End')