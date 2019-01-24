from ..task import Task

import spacy


class SpacyNamedEntityRecognition(Task):

    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("en", disable=["tagger"])

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
                    "entities": self.get_entities(sentence)
                })

            # Add the paragraph to our results
            results.append(par)

        return results, words

    def get_entities(self, sent):
        return [{e.text: (e.label_, e.start_char - e.sent.start_char, e.end_char - e.sent.start_char)} for e in
                sent.ents]
