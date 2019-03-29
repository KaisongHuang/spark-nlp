Setup
-

The following script will setup the virtualenv + download Anserini-Spark and any other required software:

`./setup.sh`

Running
-

We need to setup our environment with the following:
```
# Required for CUDA
export CUDA_HOME=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64:/usr/local/cudnn7/lib64
export PATH=/usr/local/cuda-9.2/bin/:$PATH

# Required on hydra
export SPARK_LOCAL_IP="127.0.0.1"
```

We're ready to run now (after changing the parameters in the script for index location, etc.):

`./run.sh`

```
#!/bin/bash

COLLECTION="/home/kaisong1huang/git/spark-nlp/testdoc.txt"

# The files to include in the PYPATH
PY_FILES="libs.zip"

export SPARK_LOCAL_IP="127.0.0.1"
export PYSPARK_PYTHON="venv/bin/python"
export PYSPARK_DRIVER_PYTHON="venv/bin/python"

# Run the code...
spark-submit \
        --master "local[2]" --executor-memory 4G --driver-memory 8G \
        --conf "spark.yarn.appMasterEnv.PYSPARK_PYTHON=$PYSPARK_PYTHON" \
        --archives "venv.zip#venv" \
        --py-files $PY_FILES main.py --collection $COLLECTION --library allennlp --sample 1 --task pos --gpu 0
```
