#!/bin/bash

# The WashingtonPost collection
COLLECTION="/home/kaisong1huang/TREC_Washington_Post_collection.v2.jl"

# The files to include in the PYPATH
PY_FILES="libs.zip"

export PYSPARK_PYTHON="venv/bin/python"
export PYSPARK_DRIVER_PYTHON="venv/bin/python"

# Run the code...
spark-submit \
        --master "local[2]" --executor-memory 4G --driver-memory 8G \
        --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
        --archives "venv.zip#venv" \
        --py-files $PY_FILES main.py --collection $COLLECTION --library stanfordnlp --task pos --sample 0.0001 --gpu 0
