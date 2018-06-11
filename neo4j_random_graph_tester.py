__author__ = 'Andrei'

import random
from bulbs.neo4jserver import Graph

# we might need to create a model if this does not terminate the problem

g = Graph()

random_node_names = ['%030x' % random.randrange(16**30) for _ in range(0, 10000)]
for i, node in enumerate(random_node_names):
    print 'inserting: %.2f %%' % (i/100.)
    g.vertices.create(name=node)

# add cross-links if this still doesn't help