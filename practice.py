import numpy

from func.utils import randomly_missing_trace

data = numpy.ones(((1, 5, 1, 5)))
print(data)
result = randomly_missing_trace(data)
print(result)