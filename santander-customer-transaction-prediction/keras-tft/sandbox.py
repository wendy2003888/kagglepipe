import apache_beam as beam
from apache_beam.transforms.userstate import CombiningValueStateSpec

class IndexAssigningStatefulDoFn(beam.DoFn):
  INDEX_STATE = CombiningValueStateSpec('index', sum)

  def process(self, element, index=beam.DoFn.StateParam(INDEX_STATE)):
    current_index = index.read()
    index.add(1)
    yield (element, current_index)

with beam.Pipeline() as pipeline:
  intrim = pipeline | 'Data' >> beam.Create([
          ('p', 1),
          ('a', 2),
          ('z', 3),
          ('m', 2),])
  intrim = intrim | beam.ParDo(IndexAssigningStatefulDoFn())
  intrim = intrim | beam.Map(print)