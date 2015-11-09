__author__ = 'Andrei'

from py2neo import Graph
from py2neo import Node, Relationship


graph = Graph('http://neo4j:sergvcx@localhost:7474/db/data')
alice = Node("Person", name="Alice")
bob = Node("Person", name="Bob")
alice_knows_bob = Relationship(alice, "KNOWS", "bob")
graph.create(alice_knows_bob)

alice.properties["age"] = 33
bob.properties["age"] = 44

graph.push(alice, bob)