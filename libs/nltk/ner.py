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
                print(sentence)
                sent = []
                tokens = nltk.word_tokenize(sentence)
                tags = nltk.pos_tag(tokens)
                chunks = nltk.ne_chunk(tags)
                for chunk in chunks:
                    # BIO Scheme
                    # B - 'beginning'
                    # I - 'inside'
                    # O - 'outside'
                    if hasattr(chunk, 'label'):
                        ner_tag = chunk.label()
                        length = len(chunk)
                        for i in range(length):
                            if i == 0:
                                sent.append((chunk[i][0], 'B-' + ner_tag))
                            else:
                                sent.append((chunk[i][0], 'I-' + ner_tag))
                            # sent.append((c[0] ,ner_tag))
                    else:
                        sent.append((chunk[0], 'O'))
                sentences.append(sent)
                words += len(tokens)
            paragraphs.append(sentences)
        return paragraphs, words
