#!/bin/bash

# The Anserini-Spark JAR
ANSERINI="/home/k86huang/anserini-spark-0.0.1-SNAPSHOT-fatjar.jar"

# The Stanford CoreNLP JAR
CORENLP="/home/k86huang/stanford-corenlp-3.9.2.jar"

# The Stanford CoreNLP English Model JAR
COREMODEL="/home/k86huang/stanford-corenlp-3.9.2-models.jar"

# The Lucene index directory
INDEX="hdfs://node-master:9000/indexes/core18-1000"

# The files to include in the PYPATH
PY_FILES="libs.zip"

export PYSPARK_PYTHON="venv/bin/python"
export PYSPARK_DRIVER_PYTHON="venv/bin/python"

# Run the code...
spark-submit \
	--num-executors 9 --executor-cores 16 --executor-memory 32G --driver-memory 32G \
	--conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
	--archives "venv.zip#venv" \
	--py-files $PY_FILES --jars $ANSERINI,$CORENLP,$COREMODEL main.py --index $INDEX --library spacy --task pos --num 10