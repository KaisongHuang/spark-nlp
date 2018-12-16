#!/bin/bash

ANSERINI="git/Anserini-Spark/target/anserini-spark-0.0.1-SNAPSHOT-fatjar.jar"
INDEX="lucene-index.core18.pos+docvectors+rawdocs-10"
PY_FILES="src/spacy_ner.py,src/allen_ner.py"

spark-2.4.0-bin-hadoop2.7/bin/spark-submit \
	--master "local[*]" \
	--py-files $PY_FILES --jars $ANSERINI src/main.py --index $INDEX