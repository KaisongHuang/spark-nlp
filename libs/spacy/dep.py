import os
import spacy

from ..task import Task


class SpacyDependencyParser(Task):

    def __init__(self, gpu):
        if gpu >= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            spacy.require_gpu()
        self.nlp = spacy.load("en")

    def run(self, data):
        results = []
        words = 0

        # For each parsed doc...
        for doc in self.nlp.pipe(data):

            # Keep track of sentences in the paragraph
            par = []

            # For each sentence in the paragraph, fetch the entities
            for sentence in doc.sents:
                words += len(sentence)
                par.append({
                    "text": sentence,
                    "deps": self.get_deps(sentence)
                })

            # Add the paragraph to our results
            results.append(par)

        return results, words

    def get_deps(self, sent):
        return [(c.text, c.root.text, c.root.dep_, c.root.head.text) for c in sent.noun_chunks]
