__author__ = 'Andrei'

import numpy as np
import random
import scipy.sparse
from bulbs.neo4jserver import Graph

# we might need to create a model if this does not terminate the problem

g = Graph()

random_node_names = ['%030x' % random.randrange(16**30) for _ in range(0, 1000)]
for node in random_node_names:
    g.vertices.create(name=node)

# add cross-links if this still doesn't help