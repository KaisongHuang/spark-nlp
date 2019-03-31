import os
import spacy
from spacy.lang.en import English
from ..task import Task


class SpacyNamedEntityRecognition(Task):

    def __init__(self, gpu):
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            spacy.require_gpu()
        self.nlp_sent = English()
        self.sentencizer = self.nlp_sent.create_pipe("sentencizer")
        self.nlp_sent.add_pipe(self.sentencizer)
        self.nlp = spacy.load("en")

    def run(self, data):
        results = []
        words = 0

        for paragraph in data:
            # Keep track of sentences in the paragraph
            par = []

            # For each sentence in the paragraph, fetch the entities
            doc = self.nlp_sent(paragraph)
            for sentence in doc.sents:
                sent = []
                tokens = self.nlp(str(sentence))
                for token in tokens:
                    if token.text == ' ':
                        continue

                    # IOB Scheme (token.ent_iob_)
                    # I - Token is inside an entity
                    # O - Token is outside an entity
                    # B - Token is the beginning of an entity
                    if token.ent_iob_ == 'O':
                        sent.append((token.text, token.ent_iob_ + token.ent_type_))
                    else:
                        sent.append((token.text, token.ent_iob_ + '-' + token.ent_type_))

                par.append(sent)
                words += len(tokens)
            results.append(par)

        return results, words
