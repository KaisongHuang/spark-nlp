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