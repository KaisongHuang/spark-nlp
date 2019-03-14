import os

import stanfordnlp
import torch

from ..task import Task

torch.backends.cudnn.enabled = False


class StanfordNLPPartOfSpeechTagger(Task):

    def __init__(self, gpu):
        use_gpu = False
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            use_gpu = True

        # Download english model
        stanfordnlp.download('en', 'venv/share', confirm_if_exists=True)

        # Specify the local dir of the model and pipeline
        self.nlp = stanfordnlp.Pipeline(lang='en', models_dir='venv/share', processors='tokenize,pos', use_gpu=use_gpu)

    def run(self, data):
        paragraphs = []
        words = 0

        # TODO

        return paragraphs, words
