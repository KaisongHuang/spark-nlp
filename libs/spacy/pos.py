import spacy

from ..task import Task


class SpacyPartOfSpeechTagger(Task):

    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("en")

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
            paragraphs.append(sentences)
        return paragraphs, words
