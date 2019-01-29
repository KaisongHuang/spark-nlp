#!/bin/bash

BASE=`pwd`

###
# Setup Python Environment
###

# Create venv
virtualenv -p /usr/bin/python3 venv

# Source venv
source venv/bin/activate

# Update setuptools
pip install -U pip setuptools

# Install AllenNLP
pip install -U allennlp

# Install NLTK
pip install -U nltk

# Install spaCy
pip install -U spacy spacy[cuda92]

# Download spaCy English model
python -m spacy download en

# Out libs folder for the Spark workers (via --py-files)
zip -qr libs.zip libs/

# Dependencies for Spark workers (via --archives)
cd venv
zip -qr ../venv.zip .