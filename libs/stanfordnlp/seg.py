from ..task import Task
import torch
torch.backends.cudnn.enabled = False
import stanfordnlp
import os

class StanfordNLPSentenceSegmenter(Task):

    def __init__(self, gpu):
        use_gpu = False
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            use_gpu = True

        # Download english model
        # stanfordnlp.download('en','venv/share',confirm_if_exists=True)

        # Specify the local dir of the model and pipeline
        self.nlp = stanfordnlp.Pipeline(lang='en', models_dir='stanfordnlp_resources', processors='tokenize', use_gpu=use_gpu)

    def run(self, data):
        paragraphs = []
        words = 0

        for paragraph in data:
            sentences = []
            doc = self.nlp(paragraph)

            for sentence in doc.sentences:
                sentences.append(sentence.words)
                words += len(sentence.words)

            paragraphs.append(sentences)

        return paragraphs, words
