import nltk

from ..task import Task


class NLTKPartOfSpeechTagger(Task):

    def __init__(self, config):
        self.config = config
        nltk.download('averaged_perceptron_tagger')

    def run(self, data):
        paragraphs = []
        words = 0
        for paragraph in data:
            sentences = []
            for sentence in nltk.sent_tokenize(paragraph):
                tokens = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)
                sentences.append(" ".join("{}[{}]".format(t[0], t[1]) for t in tagged))
                words += len(tokens)
            paragraphs.append(sentences)
        return paragraphs, words
