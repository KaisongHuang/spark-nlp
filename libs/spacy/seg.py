import os
import spacy

from ..task import Task


class SpacySentenceSegmenter(Task):

    def __init__(self, gpu):
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            spacy.require_gpu()
        self.nlp = spacy.load("en")

    def run(self, data):
        paragraphs = []
        words = 0
        for doc in self.nlp.pipe(data):
            sentences = []
            for sentence in doc.sents:
                sentences.append(str(sentence))
                words += len(sentence)
            paragraphs.append(sentences)
        return paragraphs, words
