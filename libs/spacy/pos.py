import os
import spacy
from spacy.lang.en import English
from ..task import Task


class SpacyPartOfSpeechTagger(Task):

    def __init__(self, gpu):
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            spacy.require_gpu()
        self.nlp_sent = English()
        self.sentencizer = self.nlp_sent.create_pipe("sentencizer")
        self.nlp_sent.add_pipe(self.sentencizer)
        self.nlp = spacy.load("en", disable=["ner"])

    def run(self, data):
        paragraphs = []
        words = 0

        for paragraph in data:
            par = []

            doc = self.nlp_sent(paragraph)
            for sentence in doc.sents:
                sent = []
                tokens = self.nlp(str(sentence))
                for token in tokens:
                    if token.text_ == ' ':
                        continue
                    sent.append((token.text, token.tag_))
                par.append(sent)
                words += len(tokens)
            paragraphs.append(par)
        return paragraphs, words
