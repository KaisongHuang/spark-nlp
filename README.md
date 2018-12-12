# Setup

The following script will setup the Conda environment (and install the dependencies) + download Anserini-Spark, Spark, and any other required software:
`./setup.sh`

After setup is complete, we need to setup our bash session:
```
# If Conda isn't already sourced...
. /home/ralph/anaconda3/etc/profile.d/conda.sh

# Required for CUDA
export CUDA_HOME=/usr/local/cuda-9.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64:/usr/local/cudnn7/lib64
export PATH=/usr/local/cuda-9.2/bin/:$PATH

# Required on hydra
export SPARK_LOCAL_IP="127.0.0.1"
```

We're ready to run now:
`python run.py --index <index_location> [--gpu]`