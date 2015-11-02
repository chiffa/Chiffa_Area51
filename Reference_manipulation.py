__author__ = 'Andrei'

import nltk
from nltk.corpus import treebank
from pprint import pprint



raw = []

with open("C:/Users/Andrei/Desktop/test_biblio.txt") as of:
    for line in of:
        raw.append(line)

tokens = []
tags = []
entities = []

for line in raw:
    token = nltk.word_tokenize(line)
    tag = nltk.pos_tag(token)
    entity = nltk.chunk.ne_chunk(tag)
    entity.draw()
    tokens.append(token)
    tags.append(tag)
    entities.append(entity)

if __name__ == "__main__":
    pprint(raw)
    pprint(tags)
    pprint(entities)

