#!/bin/bash

# The Anserini-Spark JAR
ANSERINI="git/Anserini-Spark/target/anserini-spark-0.0.1-SNAPSHOT-fatjar.jar"

# The Lucene index directory
INDEX="/home/ryan1clancy/lucene-index.core18.pos+docvectors+rawdocs-1"

# The files to include in the PYPATH
PY_FILES="src/spacy_ner.py,src/allen_ner.py"

# Run the code...
spark-2.4.0-bin-hadoop2.7/bin/spark-submit \
	--master "local[2]" --executor-memory 8G --driver-memory 8G \
	--py-files $PY_FILES --jars $ANSERINI src/main.py --index $INDEX --library spacy --num 100
