import nltk

from ..task import Task


class NLTKSentenceSegmenter(Task):

    def __init__(self, config):
        self.config = config
        nltk.download("punkt")

    def run(self, data):
        paragraphs = []
        words = 0
        for paragraph in data:
            paragraphs.append(nltk.sent_tokenize(paragraph))
            words += len(nltk.word_tokenize(paragraph))
        return paragraphs, words
