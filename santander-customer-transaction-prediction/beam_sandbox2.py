import apache_beam as beam

data = []
with beam.Pipeline() as pipeline:
  intrim = pipeline | 'Data' >> beam.Create([
          ('p', 1),
          ('a', 2),
          ('z', 3),
          ('m', 2),])
  intrim = intrim | 'Sink' >> beam.Map(lambda item: data.append(item))

print(data)
data.sort(key = lambda item: item[0] )
print(data)
