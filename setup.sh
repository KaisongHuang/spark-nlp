#!/bin/bash

# Environment name
ENV_NAME="bigdata"

# Base directory
BASE=`pwd`

###
# Setup Python Environment
###

# Source Anaconda
. /home/ryan1clancy/.miniconda/etc/profile.d/conda.sh

# Create conda environment
conda create --name $ENV_NAME python=3.7

# Activate the enviroment
conda activate $ENV_NAME

# Install spaCy
conda install -c conda-forge spacy

# Download spaCy English model
python -m spacy download en

# Download AllenNLP
pip install allennlp

###
# CUDA - Disabled for now
###

# Create git directory
mkdir git && cd git

# Install spaCy CUDA
#pip install spacy[cuda92]

###
# Anserini + Spark
###

# Setup Anserini-Spark
git clone https://github.com/castorini/Anserini-Spark.git && cd Anserini-Spark && mvn clean package

# Back to the base...
cd $BASE

# Setup Spark
wget -qO- http://mirror.csclub.uwaterloo.ca/apache/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz | tar -zxv