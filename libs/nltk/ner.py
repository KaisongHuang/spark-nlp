import nltk

from ..task import Task


class NLTKNamedEntityRecognition(Task):

    def __init__(self, config):
        self.config = config
        nltk.download("punkt")
        nltk.download('averaged_perceptron_tagger')
        nltk.download("maxent_ne_chunker")
        nltk.download("words")

    def run(self, data):
        paragraphs = []
        words = 0
        for paragraph in data:
            sentences = []
            for sentence in nltk.sent_tokenize(paragraph):
                tokens = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(tokens)
                chunked = nltk.ne_chunk(tagged)
                sentences.append(chunked)
                words += len(tokens)
            paragraphs.append(sentences)
        return paragraphs, words
