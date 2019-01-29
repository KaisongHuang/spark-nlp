#!/bin/bash

# The WashingtonPost collection
COLLECTION="/home/ryan1clancy/ir/collections/WashingtonPost.v2/data/TREC_Washington_Post_collection.v2.jl"

# The files to include in the PYPATH
PY_FILES="libs.zip"

export PYSPARK_PYTHON="venv/bin/python"
export PYSPARK_DRIVER_PYTHON="venv/bin/python"

# Run the code...
spark-submit \
        --num-executors 8 --executor-cores 1 --executor-memory 4G --driver-memory 8G \
        --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
        --archives "venv.zip#venv" \
        --py-files $PY_FILES main.py --collection $COLLECTION --library spacy --task ner --sample 0.0001