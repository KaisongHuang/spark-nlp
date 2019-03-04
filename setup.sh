#!/bin/bash

###
# Setup Python Environment
###

# Create venv
virtualenv -p /usr/bin/python3 venv

# Source venv
source venv/bin/activate

# Install AllenNLP
pip install -U allennlp

# Install NLTK
pip install -U nltk

# Install spaCy
pip install -U spacy

# Install spaCy GPU support
pip install -U spacy[cuda92]

# Install stanfordnlp
pip install -U stanfordnlp

# Download spaCy English model
python -m spacy download en

# Out libs folder for the Spark workers (via --py-files)
zip -qr libs.zip libs/

# Dependencies for Spark workers (via --archives)
cd venv
zip -qr ../venv.zip .
