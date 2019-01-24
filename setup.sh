#!/bin/bash

###
# Setup Python Environment
###

# Create venv
virtualenv -p /usr/bin/python3 venv

# Source venv
source venv/bin/activate

# Update setuptools
pip install -U setuptools

# Install NLTK
pip install -U nltk

# Install spaCy
pip install -U spacy

# Download spaCy English model
python -m spacy download en

# Dependencies for Spark workers
zip -qr venv.zip venv/*

###
# Anserini-Spark
###

# Create git directory
mkdir git && cd git

# Setup Anserini-Spark
git clone https://github.com/castorini/Anserini-Spark.git && cd Anserini-Spark && mvn clean package
