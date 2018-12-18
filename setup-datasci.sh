#!/bin/bash

# Base directory
BASE=`pwd`

###
# Setup Python Environment
###

# Create venv
virtualenv -p /usr/bin/python3 venv

# Source venv
. venv/bin/activate

# Download spaCy
pip install spacy

# Download spaCy English model
python -m spacy download en

# Download AllenNLP
pip install allennlp

cd venv

# Dependencies for Spark workers
zip -qr ../venv.zip .

cd $BASE

# Create git directory
mkdir git && cd git

###
# Anserini + Spark
###

# Setup Anserini-Spark
git clone https://github.com/castorini/Anserini-Spark.git && cd Anserini-Spark && mvn clean package

# Back to the base...
cd $BASE

# Setup Spark
wget -qO- http://mirror.csclub.uwaterloo.ca/apache/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz | tar -zxv
