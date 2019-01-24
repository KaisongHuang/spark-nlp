#!/bin/bash

# The Anserini-Spark JAR
ANSERINI="git/Anserini-Spark/target/anserini-spark-0.0.1-SNAPSHOT-fatjar.jar"

# The Lucene index directory
INDEX="hdfs://node-master:9000/indexes/core17"

# The files to include in the PYPATH
PY_FILES="src/libs/spacy/ner.py"

export PYSPARK_PYTHON="venv/bin/python"
export PYSPARK_DRIVER_PYTHON="venv/bin/python"

# Run the code...
spark-submit \
	--num-executors 9 --executor-cores 4 --executor-memory 8G --driver-memory 8G \
	--conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
	--archives "venv.zip#venv" \
	--py-files $PY_FILES --jars $ANSERINI src/main.py --index $INDEX --library spacy --task ner