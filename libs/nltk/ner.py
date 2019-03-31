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
                sent = []
                tokens = nltk.word_tokenize(sentence)
                tags = nltk.pos_tag(tokens)
                chunks = nltk.ne_chunk(tags)
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        ner_tag = chunk.label()
                        for c in chunk:
                            sent.append((c[0] ,ner_tag))
                    else:
                        sent.append((chunk[0], '0'))
                sentences.append(sent)
                words += len(tokens)
            paragraphs.append(sentences)
        return paragraphs, words
