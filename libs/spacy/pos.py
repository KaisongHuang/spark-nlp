import os

import spacy

from ..task import Task


class SpacyPartOfSpeechTagger(Task):

    def __init__(self, gpu):
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            spacy.require_gpu()
        self.nlp = spacy.load("en", disable=["ner"])

    def run(self, data):
        paragraphs = []
        words = 0
        for paragraph in self.nlp.pipe(data):
            sentences = []
            for sentence in paragraph.sents:
                tokens = []
                for token in sentence:
                    tokens.append("{}[{}]".format(token.text, token.pos_))
                sentences.append(" ".join(str(t) for t in tokens))
                words += len(sentence)
            paragraphs.append(sentences)
        return paragraphs, words
