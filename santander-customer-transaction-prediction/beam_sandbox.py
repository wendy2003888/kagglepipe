from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tensorflow_serving.apis import prediction_log_pb2
import apache_beam as beam
import tensorflow as tf


def print_item(item, file):
    example_bytes = item.predict_log.request.inputs['input_example_tensor'].string_val[0]
    
    # parsed = tf.train.Example.FromString(example_bytes)
    # parsed is tf.Example (list of feature)

    features = {
        'ID_code': tf.io.FixedLenFeature((), tf.string)
    }
    parsed = tf.io.parse_single_example(example_bytes, features=features)
    # parsed['ID_code'] is a Tensor with string value, .numpy() can gets the value like b'id1'
    id_string = parsed['ID_code'].numpy().decode()
    output = item.predict_log.response.outputs['output_0'].float_val[0]
    file.write('{0},{1}\n'.format(id_string, 1 if output >= 0.5 else 0))
    
input_dir = '/var/tmp/santander/keras-tft/HelloComponent.HelloWorld/output_data/10'
input_uri = io_utils.all_files_pattern(input_dir)
with tf.io.gfile.GFile('/var/tmp/output.csv', 'w') as file:
    file.write('ID_code,target\n')

    p = beam.Pipeline()
    out = p | 'ReadExamples' >> beam.io.ReadFromTFRecord(file_pattern=input_uri, coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog))
    out = out | 'Print' >> beam.Map(lambda item: print_item(item, file))
    result = p.run()

print('done')