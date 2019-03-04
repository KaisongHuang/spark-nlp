#!/bin/bash

# The Stanford CoreNLP JAR
# CORENLP="/<somewhere>/stanford-corenlp-3.9.2.jar"

# The Stanford CoreNLP English Model JAR
# COREMODEL="/<somewhere>/stanford-corenlp-3.9.2-models.jar"

# The WashingtonPost collection
COLLECTION="/home/kaisong1huang/TREC_Washington_Post_collection.v2.jl"

# The files to include in the PYPATH
PY_FILES="libs.zip"

export PYSPARK_PYTHON="venv/bin/python"
export PYSPARK_DRIVER_PYTHON="venv/bin/python"

# Run the code...
spark-submit \
        --master "local[8]" --executor-memory 4G --driver-memory 8G \
        --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
        --archives "venv.zip#venv" \
        --py-files $PY_FILES main.py --collection $COLLECTION --library spacy --task pos --sample 0.0001 --gpu 1
