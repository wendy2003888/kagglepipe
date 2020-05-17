# Lint as: python3
from typing import Optional, Text
from custom_component import executor
from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter


class TestPredComponentSpec(types.ComponentSpec):
  """ComponentSpec for Custom TFX Hello World Component."""

  PARAMETERS = {
      # These are parameters that will be passed in the call to
      # create an instance of this component.
  }
  INPUTS = {
      # This will be a dictionary with input artifacts, including URIs
      'examples': ChannelParameter(type=standard_artifacts.Examples),
      
      'model': ChannelParameter(type=standard_artifacts.Model),
  }
  OUTPUTS = {
      # This will be a dictionary which this component will populate
      'output_data': ChannelParameter(type=standard_artifacts.Examples),
  }


class TestPredComponent(base_component.BaseComponent):
  """Custom TFX Hello World Component.

  This custom component class consists of only a constructor.
  """

  SPEC_CLASS = TestPredComponentSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               examples: types.Channel = None,
               model: types.Channel = None,
               output_data: types.Channel = None):
    """Construct a HelloComponent.

    Args:
      input_data: A Channel of type `standard_artifacts.String`.
      output_data: A Channel of type `standard_artifacts.String`.
      name: Optional unique name. Necessary if multiple Hello components are
        declared in the same pipeline.
    """
    if not output_data:
        examples_artifact = standard_artifacts.Examples()
        output_data = channel_utils.as_channel([examples_artifact])

    spec = TestPredComponentSpec(
        examples=examples, model=model, output_data=output_data)
    super(TestPredComponent, self).__init__(spec=spec)
